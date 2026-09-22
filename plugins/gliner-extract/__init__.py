"""Bounded GLiNER semantic evidence for Hermes' System-1 router.

GLiNER is used as an extractor, not treated as an infallible intent classifier.
It identifies action targets, investigation/design signals, constraints, and risk.
The sibling ``glm-codex-effort`` plugin combines that evidence with live harness
state before selecting a fast or full execution lane.

The model is pre-warmed in the background. Classification is cached, bounded by
a short timeout, and fail-open: GLiNER can improve routing but can never prevent a
normal Hermes request from reaching the model.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_LOADING = False
_MODEL_LOCK = threading.Lock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-gliner")
_BUSY_UNTIL = 0.0
_EVIDENCE: ContextVar[dict[str, Any] | None] = ContextVar(
    "hermes_gliner_evidence", default=None
)
_CACHE: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CTX = None

_SYNTHETIC_USER_FLAGS = (
    "_empty_recovery_synthetic",
    "_empty_terminal_sentinel",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_kanban_stop_synthetic",
    "_dropped_toolcall_nudge",
    "_todo_snapshot_synthetic",
)
_SYNTHETIC_DISPLAY_KINDS = {
    "auto_continue",
    "model_switch",
    "personality_switch",
    "process_complete",
    "hidden",
}
_AUTO_CONTINUE_PREFIX = "[System note: Your previous turn was interrupted mid-run"
_EMPTY_RECOVERY_TEXT = (
    "You just executed tool calls but returned an empty response. "
    "Please process the tool results above and continue with the task."
)

# Labels describe spans GLiNER can extract reliably. They are semantic evidence,
# not a keyword list of commands that happen to be common today.
_LABELS = (
    "requested software operation",
    "software target",
    "request for explanation",
    "debugging symptom",
    "root cause investigation",
    "architecture or system design",
    "irreversible destructive operation",
    "password API key or private credential",
    "specific constraint",
)
_FAST_LABELS = {
    "requested software operation",
    "software target",
    "request for explanation",
    "specific constraint",
}
_COMPLEX_LABELS = {
    "debugging symptom",
    "root cause investigation",
}
_RISK_LABELS = {
    "irreversible destructive operation",
    "password API key or private credential",
}

# Deterministic checks are guardrails only. They can prevent an unsafe fast
# route; they never create one based on a product name such as Vite/localhost.
_IRREVERSIBLE_RE = re.compile(
    r"\b(drop\s+(?:database|table)|delete\s+(?:database|account|production)|"
    r"reset\s+production|rm\s+-rf|force\s+push|reboot|kill\s+-9)\b",
    re.I,
)
_COMPLEX_GUARD_RE = re.compile(
    r"\b(root\s+cause|race\s+condition|deadlock|architecture|architectural|"
    r"migration\s+plan|(?:database|schema)\s+migrat\w*|security\s+audit|"
    r"audit\b.{0,80}\bsecurity|security\b.{0,80}\baudit|vulnerabilit(?:y|ies)|"
    r"threat\s+model|refactor(?:ing)?|oauth|from\s+scratch)\b",
    re.I,
)
_DEICTIC_RE = re.compile(
    r"\b(it|that|this|those|these|continue|again)\b", re.I
)
_STRONG_DEICTIC_RE = re.compile(
    r"\b(same|above|earlier|previous(?:ly)?)\b", re.I
)
_SOCIAL_RE = re.compile(
    r"\s*(?:hi|hello|hey|hey\s+there|good\s+(?:morning|afternoon|evening))"
    r"(?:\s+(?:there|again))?[!.?\s]*$",
    re.I,
)
_ACTION_REQUEST_RE = re.compile(
    r"\s*(?:(?:can|could|would|will)\s+you\s+|please\s+)?"
    r"(?:start|run|stop|restart|launch|open|close|read|inspect|check|find|search|"
    r"change|edit|update|add|remove|fix|build|test|verify|deploy|create|implement|"
    r"continue|resume|show|list|write|patch)\b",
    re.I,
)
_PROSE_REQUEST_RE = re.compile(
    r"\s*(?:(?:what|why|how|when|where|who|which)\b|"
    r"(?:(?:can|could|would)\s+you\s+)?(?:explain|describe|clarify|summarize)\b|"
    r"tell\s+me\s+about\b)",
    re.I,
)


def _setting(key: str, default: Any) -> Any:
    if _CTX is None:
        return default
    try:
        return _CTX.get_config(key, default)
    except Exception:
        return default


def _model_name() -> str:
    return str(_setting("model_name", "gliner-community/gliner_small-v2.5"))


def _timeout_seconds() -> float:
    try:
        return max(0.025, min(0.75, float(_setting("timeout_ms", 250)) / 1000.0))
    except (TypeError, ValueError):
        return 0.25


def _enabled() -> bool:
    return bool(_setting("enabled", True))


def _load_model():
    global _MODEL, _MODEL_LOADING
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        _MODEL_LOADING = True
        try:
            from gliner import GLiNER

            model = GLiNER.from_pretrained(_model_name())
            try:
                model.eval()
            except Exception:
                pass
            # Torch performs additional setup on the first prediction. Keep it
            # inside startup prewarm so the first user turn sees steady-state
            # classifier latency rather than a one-off timeout.
            model.predict_entities("warmup", list(_LABELS), threshold=0.99)
            _MODEL = model
            logger.info("GLiNER ready: %s", _model_name())
            return _MODEL
        finally:
            _MODEL_LOADING = False


def _prewarm() -> None:
    try:
        started = time.perf_counter()
        _load_model()
        logger.info("GLiNER prewarm completed in %.1fms", (time.perf_counter() - started) * 1000)
    except Exception:
        logger.warning("GLiNER prewarm failed; Hermes will use its normal router")
        logger.debug("GLiNER prewarm failure", exc_info=True)


def _latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages or []):
        if (
            not isinstance(message, dict)
            or message.get("role") != "user"
            or _is_synthetic_user(message)
        ):
            continue
        content = message.get("content")
        if isinstance(content, str):
            return _strip_attached_context(content)
        if isinstance(content, list):
            return _strip_attached_context("\n".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
            ))
    return ""


def _strip_attached_context(text: str) -> str:
    """Classify user instructions without Desktop's expanded URL/file payload.

    The full attachment remains in the model request. Only local routing sees
    the concise user-authored prefix, keeping GLiNER inside its bounded input.
    """
    for marker in ("\n\n--- Attached Context ---", "\n--- Attached Context ---"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return text.strip()


def _is_synthetic_user(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return True
    if str(message.get("display_kind") or "") in _SYNTHETIC_DISPLAY_KINDS:
        return True
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else ""
    return text.startswith(_AUTO_CONTINUE_PREFIX) or text == _EMPTY_RECOVERY_TEXT


def _should_probe(text: str) -> bool:
    # All reasonably bounded turns are eligible. This is intentionally not tied
    # to words such as start/run/Vite/localhost.
    try:
        max_chars = int(_setting("max_input_chars", 1200))
    except (TypeError, ValueError):
        max_chars = 1200
    return bool(text and len(text) <= max(160, max_chars))


def _cache_get(text: str) -> dict[str, Any] | None:
    now = time.monotonic()
    with _CACHE_LOCK:
        item = _CACHE.get(text)
        if item is None:
            return None
        created, evidence = item
        if now - created > 900:
            _CACHE.pop(text, None)
            return None
        _CACHE.move_to_end(text)
        cached = dict(evidence)
        cached["cached"] = True
        cached["latency_ms"] = 0.0
        return cached


def _cache_put(text: str, evidence: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _CACHE[text] = (time.monotonic(), dict(evidence))
        _CACHE.move_to_end(text)
        while len(_CACHE) > 256:
            _CACHE.popitem(last=False)


def _extract(text: str) -> dict[str, Any]:
    cached = _cache_get(text)
    if cached is not None:
        return cached

    started = time.perf_counter()
    model = _load_model()
    try:
        threshold = float(_setting("threshold", 0.25))
    except (TypeError, ValueError):
        threshold = 0.25
    entities = model.predict_entities(text, list(_LABELS), threshold=threshold)
    slots: dict[str, list[dict[str, Any]]] = {label: [] for label in _LABELS}
    for item in entities or []:
        label = str(item.get("label") or "")
        if label not in slots:
            continue
        slots[label].append(
            {
                "text": str(item.get("text") or "")[:240],
                "score": round(float(item.get("score") or 0.0), 3),
            }
        )
    slots = {key: value for key, value in slots.items() if value}

    def strongest(labels: set[str]) -> float:
        return max(
            (float(item["score"]) for label in labels for item in slots.get(label, [])),
            default=0.0,
        )

    operation_score = strongest({"requested software operation", "specific constraint"})
    explanation_score = strongest({"request for explanation"})
    target_score = strongest({"software target"})
    fast_score = max(operation_score, explanation_score, target_score)
    complex_score = strongest(_COMPLEX_LABELS)
    risk_score = strongest(_RISK_LABELS)
    target_count = len(slots.get("software target", []))
    sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])
    destructive = risk_score >= 0.45 or bool(_IRREVERSIBLE_RE.search(text))
    complex_request = (
        complex_score >= 0.34
        or bool(_COMPLEX_GUARD_RE.search(text))
        or target_count >= 4
        or sentence_count >= 4
    )
    social_greeting = bool(_SOCIAL_RE.fullmatch(text))
    action_request = bool(_ACTION_REQUEST_RE.match(text))
    prose_request = bool(_PROSE_REQUEST_RE.match(text))
    strong_reference = bool(_STRONG_DEICTIC_RE.search(text))
    unresolved_reference = not social_greeting and (
        strong_reference
        or (bool(_DEICTIC_RE.search(text)) and fast_score < 0.30)
    )
    short_safe_response = (
        len(text) <= 180
        and "```" not in text
        and not unresolved_reference
        and not destructive
        and not complex_request
    )

    if destructive or complex_request:
        decision = "full"
        category = "risk" if destructive else "complex"
        confidence = max(
            risk_score,
            complex_score,
            0.7 if _COMPLEX_GUARD_RE.search(text) else 0.0,
        )
    elif social_greeting:
        decision = "fast"
        category = "quick_response"
        confidence = 0.95
    elif action_request and not strong_reference:
        decision = "fast"
        category = "bounded_operation"
        confidence = max(fast_score, 0.72)
    elif prose_request and not strong_reference:
        decision = "fast"
        category = "quick_response"
        confidence = max(explanation_score, target_score, 0.72)
    elif explanation_score >= 0.30 and operation_score < 0.30 and not unresolved_reference:
        decision = "fast"
        category = "quick_response"
        confidence = max(explanation_score, target_score)
    elif fast_score >= 0.30 and not unresolved_reference:
        decision = "fast"
        category = "bounded_operation"
        confidence = fast_score
    elif short_safe_response:
        decision = "fast"
        category = "quick_response"
        confidence = 0.62
    else:
        decision = "unknown"
        category = "ambiguous"
        confidence = max(fast_score, complex_score, risk_score)

    evidence = {
        "decision": decision,
        "category": category,
        "destructive": destructive,
        "complex": complex_request,
        "unresolved_reference": unresolved_reference,
        "slots": slots,
        "confidence": round(confidence, 3),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "cached": False,
    }
    _cache_put(text, evidence)
    return evidence


def _signal(text: str) -> dict[str, Any] | None:
    global _BUSY_UNTIL
    now = time.monotonic()
    if now < _BUSY_UNTIL or _MODEL_LOADING:
        return None
    future = _EXECUTOR.submit(_extract, text)
    try:
        return future.result(timeout=_timeout_seconds())
    except FutureTimeout:
        _BUSY_UNTIL = now + 1.0
        logger.warning(
            "GLiNER exceeded %.0fms budget; using normal Hermes routing",
            _timeout_seconds() * 1000,
        )
        return None
    except Exception:
        _BUSY_UNTIL = now + 5.0
        logger.warning("GLiNER unavailable; using normal Hermes routing")
        logger.debug("GLiNER extraction failure", exc_info=True)
        return None


def consume_evidence() -> dict[str, Any] | None:
    evidence = _EVIDENCE.get()
    _EVIDENCE.set(None)
    return evidence


def on_llm_request(**kwargs: Any):
    _EVIDENCE.set(None)
    request = kwargs.get("request")
    model = str(kwargs.get("model") or (request or {}).get("model") or "")
    if not isinstance(request, dict) or not _enabled() or "glm-5.3" not in model.lower():
        return None
    messages = request.get("messages")
    if not isinstance(messages, list):
        messages = request.get("input")
    text = _latest_user_text(messages or [])
    if not _should_probe(text):
        return None
    evidence = _signal(text)
    if evidence is None:
        return None
    _EVIDENCE.set(evidence)
    logger.info(
        "gliner-route: decision=%s category=%s confidence=%.3f latency_ms=%.1f cached=%s",
        evidence["decision"],
        evidence["category"],
        evidence["confidence"],
        evidence["latency_ms"],
        evidence["cached"],
    )
    # Evidence remains process-local and never enters the provider payload.
    return {
        "request": dict(request),
        "source": "gliner-extract",
        "reason": f"{evidence['decision']}:{evidence['category']}",
    }


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    ctx.register_middleware("llm_request", on_llm_request)
    if _enabled() and bool(_setting("prewarm", True)):
        threading.Thread(target=_prewarm, name="hermes-gliner-prewarm", daemon=True).start()
