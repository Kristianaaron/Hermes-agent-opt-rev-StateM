"""Adaptive GLM 5.3 reasoning: keep thinking on, switch low vs high like Codex.

Spark's chat template only honors ``low`` and ``high`` (anything else becomes
max). This plugin rewrites extra_body after the transport builds the request:

- high: plan, repair, escalate, or a clearly hard first turn
- low: simple edits and routine tool hops (localize/execute/verify)

Thinking is never turned off. Prior CoT is cleared so GLM does not re-deliberate
the last think block on every tool hop.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_COMPLEX = re.compile(
    r"\b(why|debug|investigat|root cause|hung|stall|deadlock|race|oom|"
    r"architect|refactor|migrat|design|systematically|"
    r"from scratch|multi-file|regress|implement (a|the|new)|trace)\b",
    re.I,
)
_SIMPLE = re.compile(
    r"\b(rename|typo|tweak|nit|wording|comment)\b|"
    r"\bchange\s+\S+\s+to\s+\S+\b|"
    r"\bset\s+\S+\s+to\s+\S+\b",
    re.I,
)
_GLM = ("glm-5.3", "glm53")
_HIGH_PHASES = {"plan", "repair", "escalate"}
_VALID_EFFORT = {"low", "high"}

_PROMPT = """\
GLM / Codex-style execution:
- Simple, local edits: brief think, then tools. Do not write a long plan.
- Ambiguous, multi-file, failing, or architectural work: reason thoroughly before more tools.
- After a clear tool result, act. Do not re-derive the whole strategy.
- Hidden reasoning is not a user reply. After tools, one short status then the next tool batch.
"""


def _is_glm(model: str) -> bool:
    lowered = (model or "").lower()
    return any(token in lowered for token in _GLM)


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"text", "input_text"}:
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts).strip()
    return ""


def _frontier_phase(messages: list) -> str:
    try:
        from agent.frontier_harness import _analyze, _load_policy

        state = _analyze(messages, _load_policy())
        return str(state.get("budget_phase") or state.get("phase") or "")
    except Exception:
        return ""


def choose_effort(messages: list, api_call_count: int = 1) -> tuple[str, str]:
    """Return (low|high, reason). Thinking stays on either way."""
    user = _latest_user_text(messages)
    phase = _frontier_phase(messages)
    complex_ask = bool(_COMPLEX.search(user)) or len(user) > 900
    simple_ask = bool(_SIMPLE.search(user)) and len(user) < 280 and not complex_ask

    if complex_ask:
        return "high", f"complex ask phase={phase or 'none'}"
    if phase in _HIGH_PHASES and not simple_ask:
        return "high", f"phase={phase}"
    if simple_ask:
        return "low", "simple local edit"
    if api_call_count > 1 and phase in {"localize", "execute", "verify", "final"}:
        return "low", f"tool-hop phase={phase}"
    if api_call_count <= 1:
        return "low", "default first turn"
    return "low", "default"


def chat_template_kwargs(effort: str) -> dict[str, Any]:
    effort = effort if effort in _VALID_EFFORT else "low"
    return {
        "enable_thinking": True,
        "reasoning_effort": effort,
        "clear_thinking": True,
    }


def _apply_effort(request: dict[str, Any], effort: str) -> dict[str, Any]:
    extra = dict(request.get("extra_body") or {})
    ctk = dict(extra.get("chat_template_kwargs") or {})
    ctk.update(chat_template_kwargs(effort))
    extra["chat_template_kwargs"] = ctk
    updated = dict(request)
    updated["extra_body"] = extra
    return updated


def on_llm_request(**kwargs: Any):
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None
    model = str(kwargs.get("model") or request.get("model") or "")
    if not _is_glm(model):
        return None
    messages = request.get("messages") if isinstance(request.get("messages"), list) else []
    try:
        api_call_count = int(kwargs.get("api_call_count") or 1)
    except (TypeError, ValueError):
        api_call_count = 1
    effort, reason = choose_effort(messages, api_call_count)
    logger.info("glm-codex-effort: %s (%s) model=%s call=%s", effort, reason, model, api_call_count)
    return {
        "request": _apply_effort(request, effort),
        "source": "glm-codex-effort",
        "reason": f"{effort}:{reason}",
    }


def glm_prompt_section(session_info) -> str:
    model = ""
    if isinstance(session_info, dict):
        model = str(session_info.get("model") or "")
    if not _is_glm(model):
        return ""
    return _PROMPT


def register(ctx) -> None:
    ctx.register_middleware("llm_request", on_llm_request)
    register_section = getattr(ctx, "register_system_prompt_section", None)
    if callable(register_section):
        register_section(
            "glm-codex-effort.steer",
            glm_prompt_section,
            position="after_memory",
            max_chars=800,
        )
