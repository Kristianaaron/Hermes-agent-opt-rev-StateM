"""Adaptive GLM execution lanes for fast, reliable Hermes turns.

The router has three normally monotonic lanes:

* ``fast``: compact current-turn context, bounded tools, thinking disabled.
* ``standard``: the normal Hermes prompt/tools with low reasoning.
* ``full``: the normal Hermes prompt/tools with high reasoning.

GLiNER supplies semantic evidence. Harness failure/repetition state, destructive
risk, scope, and an explicit High picker selection can only promote a turn.
Bounded policy clamps may narrow an accidentally broad low-reasoning lane, but
never a genuine full/high lane. All mutations are request local, so durable chat
history and the installed Hermes source remain untouched.

Action budgets only apply to bounded (System-1) tasks. When a bounded task outgrows
its budget it escalates to the normal harness instead of being cut off; the no-tool
``finalize`` lane is a late safety net, not the normal end of a long task.
"""

from __future__ import annotations

import hashlib
import logging
import json
import re
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_CTX = None
_GLM = ("glm-5.3", "glm53")
_VALID_EFFORT = {"low", "high"}
_TURN_LANES: dict[str, tuple[str, float]] = {}
_TURN_LOCK = threading.Lock()

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
_SYSTEM_CONTINUE_PREFIXES = (
    "[system: continue now",
    "[system: continue",
    "[system: resume now",
)
_EMPTY_RECOVERY_TEXT = (
    "You just executed tool calls but returned an empty response. "
    "Please process the tool results above and continue with the task."
)
_DISCOVERY_TOOLS = {"tool_search", "tool_describe"}

_HARD_COMPLEX_GUARD = re.compile(
    r"\b(root\s+cause|race\s+condition|deadlock|architecture|architectural|"
    r"security\s+audit|threat\s+model|migration\s+plan|distributed\s+transaction)\b",
    re.I,
)
_BOUNDED_ACTION = re.compile(
    r"\b(?:add|adjust|apply|change|check|create|delete|disable|enable|fix|implement|"
    r"inspect|make|move|open|remove|rename|replace|restart|run|set|start|stop|test|"
    r"update|verify)\b",
    re.I,
)
_MUTATION_REQUEST = re.compile(
    r"\b(?:add|adjust|apply|change|create|delete|fix|implement|make|move|remove|"
    r"rename|replace|set|update|write|needs?\s+to|should)\b",
    re.I,
)
_READ_ONLY_TOOLS = {
    "read_file", "read_preview", "search_files", "web_search", "web_extract",
    "browser_snapshot", "browser_console", "browser_get_images", "session_search",
    "skill_view", "skills_list", "tool_search", "tool_describe",
}
_DIRECT_MUTATION_TOOLS = {
    "patch", "write_file", "browser_click", "browser_type", "browser_press",
}
_EXPLICIT_FAILURE_TEXT = re.compile(
    r"^\s*(?:traceback\s*\(|(?:tool\s+)?errors?\s*:|failed\s*:|failure\s*:|"
    r"command\s+(?:failed|timed?\s*out)|\[command\s+timed?\s*out|blocked\s*:|"
    r"permission\s+denied\b|unauthorized\b|http\s+(?:401|403|429|5\d\d)\b|"
    r"exit\s+code\s*[1-9]\d*\b)",
    re.I,
)
_CONTEXTUAL_FOLLOWUP_ACTIONS = {"continue", "proceed", "resume", "retry", "rety", "try"}
_CONTEXTUAL_FOLLOWUP_FILLER = {
    "a", "able", "again", "ahead", "and", "approach", "are", "be", "can", "change",
    "could", "different", "do", "fix", "from", "go", "implementation", "is", "it",
    "just", "last", "now", "off", "on", "please", "request", "same", "so", "still",
    "task", "that", "the", "then", "this", "to", "we", "where", "with", "work",
    "working", "you",
}
_RECENT_EXCHANGE_REFERENCE = re.compile(
    r"\b(?:it|that|this|those|these|same|above|earlier|previous(?:ly)?|"
    r"continue|resume|again)\b",
    re.I,
)

_FAST_TOOLS = {
    "terminal",
    "read_terminal",
    "process",
    "process_manage",
    "read_file",
    "read_preview",
    "write_file",
    "patch",
    "search_files",
    "desktop_project",
    "desktop_preview",
    "open_preview",
    "close_preview",
    "close_terminal",
    "browser_exec",
    "web_search",
    "web_extract",
}

_FAST_SYSTEM_PROMPT = """\
You are Hermes in System 1: act quickly on the user's current bounded request.
- If the user is chatting or asking for a simple explanation, answer directly with no tools.
- If the user asks to inspect, change, run, or verify something, use the smallest direct tool immediately.
- Do not announce a plan, inspect skills, or search for more tools in the System-1 lane.
- Use the provided tools directly and follow each schema exactly.
- For a local dev server, use the terminal background option; never shell '&' or nohup.
- For a small edit, inspect only the named/obvious file, patch it, and verify proportionally.
- Batch independent reads. Do not repeat an unchanged failed action.
- If a tool fails, the request becomes risky/ambiguous, or scope expands, stop guessing; the
  next model call will automatically receive the full Hermes harness.
- After success, answer concisely with the result and any address/path the user needs.
Never expose hidden reasoning or private credentials.
"""

_MUTATION_GATE_PROMPT = """\
You have enough inspection evidence for this bounded edit. Do not read or search again.
Make the smallest concrete change now with patch/write_file (or the relevant direct UI action),
then return a concise result. If the evidence is genuinely insufficient, state the exact blocker
instead of exploring further.
"""

_FINALIZE_PROMPT = """\
Tools are no longer available for this turn. Do not call or describe tool calls.
Write the final answer now from the evidence above: what was done, what was verified,
and anything still incomplete with the specific next step.
"""

_FULL_STEERING = """\
GLM adaptive execution:
- Routine work: act with tools promptly and keep visible narration short.
- Complex, ambiguous, failing, multi-file, or architectural work: reason thoroughly.
- After a clear tool result, continue from that evidence; do not re-derive the whole strategy.
- Verification must be proportional to the requested change.
- Never expose private chain-of-thought; provide concise outcomes and evidence.
"""


def _setting(key: str, default: Any) -> Any:
    if _CTX is None:
        return default
    try:
        return _CTX.get_config(key, default)
    except Exception:
        return default


def _gliner_module() -> Any:
    module = sys.modules.get("hermes_gliner_extract")
    if module is not None:
        return module
    for module in tuple(sys.modules.values()):
        if module is not None and str(getattr(module, "__name__", "")).endswith(".gliner_extract"):
            return module
    return None


def _consume_gliner_evidence() -> dict[str, Any] | None:
    """Read the sibling plugin's request-local signal without a hard dependency."""
    consume = getattr(_gliner_module(), "consume_evidence", None)
    if callable(consume):
        try:
            return consume()
        except Exception:
            logger.debug("GLiNER evidence consume failed", exc_info=True)
    return None


def _gliner_evidence(text: str) -> dict[str, Any] | None:
    """Evidence from middleware state, else a direct (cached, time-boxed) classify.

    Hermes runs middleware in plugin load order, so the GLiNER middleware may run
    after this router. Pulling directly keeps GLiNER effective in either order.
    """
    evidence = _consume_gliner_evidence()
    if evidence is not None or not text:
        return evidence
    classify = getattr(_gliner_module(), "classify", None)
    if not callable(classify):
        return None
    try:
        return classify(text)
    except Exception:
        logger.debug("GLiNER direct classify failed", exc_info=True)
        return None


def _is_glm(model: str) -> bool:
    lowered = (model or "").lower()
    return any(token in lowered for token in _GLM)


def _request_messages(request: dict[str, Any]) -> tuple[str, list[Any]]:
    messages = request.get("messages")
    if isinstance(messages, list):
        return "messages", messages
    messages = request.get("input")
    if isinstance(messages, list):
        return "input", messages
    return "", []


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
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
            ).strip()
    return ""


def _routing_user_text(messages: list[Any]) -> str:
    """Return only user-authored instructions, excluding injected attachment payloads.

    Desktop URL/file context can be thousands of characters even when the user's request
    is a tiny bounded edit. Routing on that payload length incorrectly turns System-1 work
    into full/high reasoning.
    """
    text, _ = _routing_user_context(messages)
    return text


def _strip_attached_context(text: str) -> str:
    for marker in ("\n\n--- Attached Context ---", "\n--- Attached Context ---"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return text.strip()


def _is_contextual_followup(text: str) -> bool:
    """Recognize a short discourse-only continuation without task details."""
    if not text or len(text) > 120:
        return False
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words or not (_CONTEXTUAL_FOLLOWUP_ACTIONS & set(words)):
        return False
    return all(
        word in _CONTEXTUAL_FOLLOWUP_ACTIONS or word in _CONTEXTUAL_FOLLOWUP_FILLER
        for word in words
    )


def _routing_user_context(messages: list[Any]) -> tuple[str, bool]:
    """Return effective intent and whether it came from the prior task.

    A real user message such as ``retry and continue`` starts a fresh failure
    scope, but it does not replace the task contract. The model still receives
    the latest message unchanged; only the local routing decision inherits.
    """
    texts: list[str] = []
    for message in messages or []:
        if (
            not isinstance(message, dict)
            or message.get("role") != "user"
            or _is_synthetic_user(message)
        ):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
            )
        else:
            text = ""
        text = _strip_attached_context(text)
        if text:
            texts.append(text)
    if not texts:
        return "", False
    latest = texts[-1]
    if not _is_contextual_followup(latest):
        return latest, False
    for previous in reversed(texts[:-1]):
        if not _is_contextual_followup(previous):
            return previous, True
    return latest, False


def _is_synthetic_user(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return True
    if str(message.get("display_kind") or "") in _SYNTHETIC_DISPLAY_KINDS:
        return True
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else ""
    lowered = text.lower()
    return (
        text.startswith(_AUTO_CONTINUE_PREFIX)
        or text == _EMPTY_RECOVERY_TEXT
        or any(lowered.startswith(prefix) for prefix in _SYSTEM_CONTINUE_PREFIXES)
    )


def _latest_is_system_continuation(messages: list[Any]) -> bool:
    """Detect an internal recovery user row that must not re-route the turn.

    The gateway inserts ``[System: Continue now...]`` after a reasoning-only or
    interrupted response.  Treating that row as a new bounded user request lets
    GLiNER demote an already-reasoning turn to ``fast``/``finalize`` and can cause
    another empty response.  It is synthetic control flow, not new intent.
    """
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        return any(text.lower().startswith(prefix) for prefix in _SYSTEM_CONTINUE_PREFIXES)
    return False


def _current_turn_messages(messages: list[Any]) -> list[Any]:
    """Return the active user turn, excluding failures from older turns.

    Failure evidence is intentionally monotonic within a turn, but must not
    poison the next user request in a long-lived session.
    """
    for idx in range(len(messages or []) - 1, -1, -1):
        message = messages[idx]
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and not _is_synthetic_user(message)
        ):
            return messages[idx:]
    return messages or []


def _frontier_state(messages: list[Any]) -> dict[str, Any]:
    try:
        from agent.frontier_harness import _analyze, _load_policy

        state = _analyze(_current_turn_messages(messages), _load_policy())
        if isinstance(state, dict) and "failure_count" in state:
            return state
    except Exception:
        pass
    # The overlay is missing or its internals changed after an upstream update.
    # Escalation on repeated failure must not silently disappear with it.
    return _local_failure_state(messages)


def _local_failure_state(messages: list[Any]) -> dict[str, Any]:
    """Current-turn failure and repeated-call counts from the transcript alone."""
    signatures: dict[str, int] = {}
    failures = 0
    for message in _current_turn_messages(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str(function.get("name") or call.get("name") or "")
                signature = name + "|" + str(function.get("arguments") or "")
                signatures[signature] = signatures.get(signature, 0) + 1
        elif message.get("role") == "tool" and _single_result_failed(message.get("content")):
            failures += 1
    return {
        "failure_count": failures,
        "max_repeated_signature": max(signatures.values(), default=0),
    }


def _single_result_failed(content: Any) -> bool:
    try:
        from agent.frontier_harness import _failed_result

        return bool(_failed_result(content))
    except Exception:
        pass
    parsed = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = None
    if isinstance(parsed, dict):
        status = str(parsed.get("status") or "").lower()
        success = (
            parsed.get("ok") is True
            or parsed.get("success") is True
            or parsed.get("isError") is False
            or status in {"ok", "success", "completed", "complete"}
        )
        exit_code = parsed.get("exit_code", parsed.get("returncode"))
        if (
            parsed.get("ok") is False
            or parsed.get("success") is False
            or parsed.get("isError") is True
            or status in {"error", "failed", "failure"}
        ):
            return True
        if isinstance(exit_code, int):
            return exit_code != 0
        return bool(parsed.get("error")) and not success
    text = content if isinstance(content, str) else str(content or "")
    return bool(_EXPLICIT_FAILURE_TEXT.match(text))


def _tool_result_failed(messages: list[Any]) -> bool:
    current = _current_turn_messages(messages)
    for message in reversed(current[-6:]):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            break
        if role == "tool" and _single_result_failed(message.get("content")):
            return True
    return False


def choose_effort(messages: list[Any], api_call_count: int = 1) -> tuple[str, str]:
    """Compatibility helper: return low/high using harness evidence only."""
    user = _routing_user_text(messages)
    state = _frontier_state(messages)
    failures = int(state.get("failure_count") or 0)
    repeated = int(state.get("max_repeated_signature") or 0)
    if failures >= 1 or repeated >= 2 or _tool_result_failed(messages):
        return "high", f"failure={failures} repeated={repeated}"
    if len(user) > int(_setting("full_request_chars", 1800)) or _HARD_COMPLEX_GUARD.search(user):
        return "high", "complex request"
    return "low", "routine"


def chat_template_kwargs(effort: str, *, thinking: bool = True) -> dict[str, Any]:
    if not thinking:
        return {"enable_thinking": False, "clear_thinking": True}
    effort = effort if effort in _VALID_EFFORT else "low"
    return {
        "enable_thinking": True,
        "reasoning_effort": effort,
        "clear_thinking": True,
    }


def _apply_effort(request: dict[str, Any], effort: str, *, thinking: bool) -> dict[str, Any]:
    extra = dict(request.get("extra_body") or {})
    ctk = dict(extra.get("chat_template_kwargs") or {})
    ctk.update(chat_template_kwargs(effort, thinking=thinking))
    if not thinking:
        ctk.pop("reasoning_effort", None)
    extra["chat_template_kwargs"] = ctk
    updated = dict(request)
    updated["extra_body"] = extra
    return updated


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


def _forced_tool_name(request: dict[str, Any]) -> str:
    choice = request.get("tool_choice")
    if not isinstance(choice, dict):
        return ""
    function = choice.get("function")
    return str(function.get("name") or "") if isinstance(function, dict) else ""


def _bounded_tools(
    tools: Any, *, forced_tool: str = "", required: bool = False,
    used: set[str] | frozenset[str] = frozenset(),
) -> list[Any]:
    if not isinstance(tools, list):
        return []
    configured = _setting("fast_tools", None)
    # Always a fresh set: adding to the module default would leak a forced tool
    # into every later fast request.
    allowed = set(configured) if isinstance(configured, list) and configured else set(_FAST_TOOLS)
    if forced_tool:
        allowed.add(forced_tool)
    # Tools already called this turn stay declared, so the replayed calls and
    # results still match a schema the model can see.
    allowed |= set(used)
    selected = [tool for tool in tools if _tool_name(tool) in allowed]
    # An explicit required-tools request with no matching fast tool must retain
    # its original surface instead of becoming an invalid provider request.
    return tools if required and not selected else selected


def _used_tool_names(messages: list[Any]) -> set[str]:
    names: set[str] = set()
    for message in _current_turn_messages(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            name = (function.get("name") if isinstance(function, dict) else None) or call.get("name")
            if name:
                names.add(str(name))
    return names


def _compact_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    keep = {
        key: value
        for key, value in message.items()
        if key in {"role", "content", "tool_calls", "tool_call_id", "name"}
    }
    if keep.get("role") == "tool" and isinstance(keep.get("content"), str):
        limit = int(_setting("fast_tool_result_chars", 8000))
        content = keep["content"]
        if len(content) > limit:
            keep["content"] = content[:limit] + "\n[tool output truncated by System-1 lane]"
    return keep


def _exchange_indices(messages: list[Any], user_idx: int) -> tuple[int, int]:
    """Return (previous real user index, its final text response index) before ``user_idx``."""
    assistant_idx = -1
    for idx in range(user_idx - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and message.get("content") and not message.get("tool_calls"):
            assistant_idx = idx
            break
    if assistant_idx < 0:
        return -1, -1
    for idx in range(assistant_idx - 1, -1, -1):
        message = messages[idx]
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and not _is_synthetic_user(message)
        ):
            return idx, assistant_idx
    return -1, assistant_idx


def _truncate_exchange(selected: list[Any]) -> list[Any]:
    limit = max(160, int(_setting("fast_recent_context_chars", 1200)))
    compacted: list[Any] = []
    for message in selected:
        item = _compact_message(message)
        if isinstance(item, dict) and isinstance(item.get("content"), str) and len(item["content"]) > limit:
            item = dict(item)
            item["content"] = item["content"][:limit] + "\n[recent context truncated]"
        compacted.append(item)
    return compacted


def _recent_exchange(messages: list[Any], user_idx: int) -> list[Any]:
    """Keep only the endpoints of one completed exchange.

    A completed agent turn may contain dozens of assistant tool calls and tool
    results between its user request and final response.  Those intermediate
    messages are execution trace, not conversational continuity.  Replaying
    them in the next fast request defeats System-1 compaction and makes remote
    prefill dominate latency, so retain only the prior real user message and
    the final text response.
    """
    return _recent_exchanges(messages, user_idx, 1)


def _recent_exchanges(messages: list[Any], user_idx: int, count: int) -> list[Any]:
    """Endpoints of up to ``count`` completed exchanges before ``user_idx``, oldest first."""
    if user_idx <= 0:
        return []
    selected: list[Any] = []
    end = user_idx
    for _ in range(max(1, count)):
        previous_user_idx, assistant_idx = _exchange_indices(messages, end)
        if assistant_idx < 0:
            break
        pair = [messages[assistant_idx]]
        if previous_user_idx >= 0:
            pair.insert(0, messages[previous_user_idx])
        selected[:0] = pair
        if previous_user_idx <= 0:
            break
        end = previous_user_idx
    return _truncate_exchange(selected)


def _needs_recent_exchange(messages: list[Any], user_idx: int) -> bool:
    """Return whether the current request depends on prior conversational text."""
    if user_idx < 0 or user_idx >= len(messages):
        return False
    message = messages[user_idx]
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, list):
        text = "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
        )
    else:
        text = str(content or "")
    text = _strip_attached_context(text)
    return _is_contextual_followup(text) or bool(_RECENT_EXCHANGE_REFERENCE.search(text))


def _compact_current_turn(messages: list[Any]) -> list[Any]:
    user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and not _is_synthetic_user(message)
        ):
            user_idx = idx
            break
    tail = messages[user_idx:] if user_idx >= 0 else messages[-1:]
    # Explicit self-contained requests do not need the previous exchange.  This
    # keeps the stable System-1 prefix directly adjacent to the new task, which
    # improves prefix-cache reuse as well as reducing raw prefill work.
    recent = _recent_exchange(messages, user_idx) if _needs_recent_exchange(messages, user_idx) else []
    return [
        {"role": "system", "content": _FAST_SYSTEM_PROMPT},
        *recent,
        *map(_compact_message, tail),
    ]


def _compact_harness_context(messages: list[Any]) -> list[Any]:
    """Bound durable-session history without replacing Hermes' system prompt.

    Full/high reasoning should change reasoning depth, not resend an unbounded
    transcript on every tool hop. Preserve the original control messages, one
    compact completed exchange, the current user request, and the most recent
    complete tool rounds. The frontier state packet already carries the recent
    failure/progress digest for older rounds.
    """
    try:
        compact_after = max(8, int(_setting("history_compact_after_messages", 24)))
    except (TypeError, ValueError):
        compact_after = 24
    if len(messages) <= compact_after:
        return list(map(_compact_message, messages))

    user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and not _is_synthetic_user(message)
        ):
            user_idx = idx
            break
    if user_idx < 0:
        return list(map(_compact_message, messages[-compact_after:]))

    controls = [
        _compact_message(message)
        for message in messages[:user_idx]
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}
    ]
    # Normal/high work keeps a few prior exchanges so references to earlier
    # decisions survive; System-1 keeps only one (see _compact_current_turn).
    try:
        exchanges = max(1, int(_setting("history_recent_exchanges", 3)))
    except (TypeError, ValueError):
        exchanges = 3
    recent = _recent_exchanges(messages, user_idx, exchanges)
    tail = messages[user_idx:]
    try:
        keep_rounds = max(2, int(_setting("history_recent_tool_rounds", 6)))
    except (TypeError, ValueError):
        keep_rounds = 6
    starts = [
        idx for idx, message in enumerate(tail)
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and message.get("tool_calls")
    ]
    if len(starts) > keep_rounds:
        tail = [
            tail[0],
            {
                "role": "assistant",
                "content": (
                    "[Earlier tool rounds were compacted by the Hermes harness. "
                    "Use the frontier state packet and current workspace state; do not replay them.]"
                ),
            },
            *tail[starts[-keep_rounds]:],
        ]
    return [*controls, *recent, *map(_compact_message, tail)]


def _apply_history_compaction(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    updated = dict(request)
    message_key, messages = _request_messages(updated)
    before_messages = len(messages)
    before_tools = len(updated.get("tools") or []) if isinstance(updated.get("tools"), list) else 0
    if message_key:
        updated[message_key] = _compact_harness_context(messages)
    return updated, {
        "messages_before": before_messages,
        "messages_after": len(updated.get(message_key) or []) if message_key else before_messages,
        "tools_before": before_tools,
        "tools_after": before_tools,
    }


def _apply_fast_lane(
    request: dict[str, Any], *, no_tools: bool = False, full_tools: bool = False,
    force_mutation: bool = False, finalize: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    updated = dict(request)
    message_key, messages = _request_messages(updated)
    before_messages = len(messages)
    before_tools = len(updated.get("tools") or []) if isinstance(updated.get("tools"), list) else 0
    used = _used_tool_names(messages)
    if message_key:
        updated[message_key] = _compact_current_turn(messages)
        addendum = _FINALIZE_PROMPT if finalize else _MUTATION_GATE_PROMPT if force_mutation else ""
        if addendum and updated[message_key]:
            system = updated[message_key][0]
            if isinstance(system, dict) and system.get("role") == "system":
                system = dict(system)
                system["content"] = str(system.get("content") or "") + "\n" + addendum
                updated[message_key][0] = system
    if no_tools:
        # OpenAI-compatible servers differ here: vLLM rejects an explicit
        # empty tool list.  Omitting the field is the portable representation
        # of a prose-only request.
        updated.pop("tools", None)
        updated.pop("tool_choice", None)
    elif isinstance(updated.get("tools"), list) and not full_tools:
        updated["tools"] = _bounded_tools(
            updated["tools"],
            forced_tool=_forced_tool_name(updated),
            required=updated.get("tool_choice") == "required",
            used=used,
        )
        if force_mutation:
            mutation_tools = [
                tool for tool in updated["tools"] if _tool_name(tool) in _DIRECT_MUTATION_TOOLS
            ]
            if mutation_tools:
                updated["tools"] = mutation_tools
    # The cap only bounds runaway output; it does not speed up generation. It must
    # fit a whole write_file/patch payload, or the tool-call JSON is truncated and
    # the turn fails with a malformed call.
    try:
        fast_max = max(2048, int(_setting("fast_max_tokens", 16384)))
    except (TypeError, ValueError):
        fast_max = 16384
    if isinstance(updated.get("max_tokens"), int):
        updated["max_tokens"] = min(updated["max_tokens"], fast_max)
    elif "max_completion_tokens" in updated and isinstance(updated.get("max_completion_tokens"), int):
        updated["max_completion_tokens"] = min(updated["max_completion_tokens"], fast_max)
    return updated, {
        "messages_before": before_messages,
        "messages_after": len(updated.get(message_key) or []) if message_key else before_messages,
        "tools_before": before_tools,
        "tools_after": len(updated.get("tools") or []) if isinstance(updated.get("tools"), list) else 0,
    }


def _configured_effort(request: dict[str, Any]) -> str:
    extra = request.get("extra_body")
    if not isinstance(extra, dict):
        return ""
    ctk = extra.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        return ""
    return str(ctk.get("reasoning_effort") or "").lower()


def _remember_lane(turn_id: str, candidate: str, *, bounded_clamp: bool = False) -> str:
    if not turn_id:
        return candidate
    rank = {"fast": 0, "standard_compact": 1, "standard": 2, "full": 3, "finalize": 4}
    now = time.monotonic()
    with _TURN_LOCK:
        for key, (_, touched) in list(_TURN_LANES.items()):
            if now - touched > 3600:
                _TURN_LANES.pop(key, None)
        previous = _TURN_LANES.get(turn_id, ("fast", now))[0]
        # A bounded policy decision may repair an earlier accidental broad/low
        # classification. It must never demote full/high or reopen finalization.
        if bounded_clamp and previous in {"fast", "standard_compact", "standard"}:
            lane = candidate
        else:
            lane = candidate if rank[candidate] >= rank[previous] else previous
        _TURN_LANES[turn_id] = (lane, now)
        return lane


def _remembered_lane(turn_id: str) -> str:
    if not turn_id:
        return ""
    with _TURN_LOCK:
        remembered = _TURN_LANES.get(turn_id)
        return remembered[0] if remembered else ""


def _effective_fast_calls(messages: list[Any], api_call_count: int) -> int:
    """Do not burn the action budget on schema discovery or recovery scaffolding."""
    discovery_rounds = 0
    recovery_rounds = 0
    for message in _current_turn_messages(messages):
        if not isinstance(message, dict):
            continue
        if _is_synthetic_user(message):
            recovery_rounds += 1
            continue
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            continue
        names = {
            str((call.get("function") or {}).get("name") or call.get("name") or "")
            for call in calls
            if isinstance(call, dict)
        }
        if names and names <= _DISCOVERY_TOOLS:
            discovery_rounds += 1
    return max(1, api_call_count - discovery_rounds - recovery_rounds)


def _tool_activity(messages: list[Any]) -> tuple[int, int]:
    """Return current-turn read-only and direct-mutation call counts."""
    reads = mutations = 0
    for message in _current_turn_messages(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            name = str(
                (function.get("name") if isinstance(function, dict) else None)
                or call.get("name") or ""
            )
            if name in _READ_ONLY_TOOLS:
                reads += 1
            if name in _DIRECT_MUTATION_TOOLS:
                mutations += 1
    return reads, mutations


def _mutation_gate_needed(messages: list[Any], user: str) -> bool:
    if not _MUTATION_REQUEST.search(user or ""):
        return False
    reads, mutations = _tool_activity(messages)
    try:
        threshold = max(1, int(_setting("max_reads_before_edit", 3)))
    except (TypeError, ValueError):
        threshold = 3
    return mutations == 0 and reads >= threshold


def _int_setting(key: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(_setting(key, default)))
    except (TypeError, ValueError):
        return default


def _decide_lane(
    request: dict[str, Any], messages: list[Any], evidence: dict[str, Any] | None,
    *, api_call_count: int, turn_id: str,
) -> tuple[str, str]:
    state = _frontier_state(messages)
    failures = int(state.get("failure_count") or 0)
    repeated = int(state.get("max_repeated_signature") or 0)
    user, inherited = _routing_user_context(messages)
    configured = _configured_effort(request)
    remembered = _remembered_lane(turn_id)

    if configured == "high":
        return _remember_lane(turn_id, "full"), "picker requested high"
    if remembered and _latest_is_system_continuation(messages):
        return _remember_lane(turn_id, remembered), "synthetic continuation; preserve active lane"
    if len(user) > int(_setting("full_request_chars", 1800)) or _HARD_COMPLEX_GUARD.search(user):
        return _remember_lane(turn_id, "full"), "complex guard"
    if failures >= 2 or repeated >= 2:
        return _remember_lane(turn_id, "full"), f"failure={failures} repeated={repeated}"
    if isinstance(evidence, dict) and str(evidence.get("decision") or "") == "full":
        category = str(evidence.get("category") or "unknown")
        confidence = float(evidence.get("confidence") or 0.0)
        full_confidence = float(_setting("full_confidence", 0.60))
        if confidence >= full_confidence:
            return _remember_lane(turn_id, "full"), f"GLiNER {category} ({confidence:.2f})"
    bounded_chars = _int_setting("bounded_action_max_chars", 500, 80)
    fast_evidence = (
        isinstance(evidence, dict)
        and evidence.get("decision") == "fast"
        and float(evidence.get("confidence") or 0.0) >= float(_setting("fast_confidence", 0.30))
    )
    bounded_task = bool(
        user and len(user) <= bounded_chars and _BOUNDED_ACTION.search(user)
    ) or fast_evidence

    # Action budgets exist to keep System-1 work short. They never apply to normal
    # harness work, which Hermes' own max_turns and loop guardrails already bound.
    if bounded_task:
        max_calls = _int_setting("fast_max_calls", 6, 1)
        hard_max_calls = max(max_calls + 1, _int_setting("fast_hard_max_calls", 10, 2))
        finalize_after = max(hard_max_calls + 1, _int_setting("fast_finalize_after_calls", 24, 3))
        effective_calls = _effective_fast_calls(messages, api_call_count)
        if effective_calls > finalize_after:
            return _remember_lane(turn_id, "finalize"), (
                f"bounded task exhausted ({effective_calls}>{finalize_after}); force final"
            )
        if effective_calls > hard_max_calls:
            # The task is bigger than it looked. Give it the full harness rather
            # than cutting it off mid-change.
            return _remember_lane(turn_id, "standard"), (
                f"bounded budget exceeded ({effective_calls}>{hard_max_calls}); escalate to harness"
            )
        if effective_calls > max_calls:
            return _remember_lane(turn_id, "standard_compact", bounded_clamp=True), (
                f"fast action budget exceeded ({effective_calls}>{max_calls}); compact low"
            )
    if failures == 1 or _tool_result_failed(messages):
        if not bounded_task:
            # Normal work that hits a tool error needs the full harness, not less.
            return _remember_lane(turn_id, "standard"), "first tool failure; normal harness"
        # A single schema/argument miss on a bounded action is routine correction,
        # not permission to expand back to the entire 39-tool harness. The clamp
        # can narrow standard/low, but cannot demote full/high.
        # If this turn already needed standard/low reasoning, however, preserve
        # that reasoning after a tool miss. Dropping it to thinking-off caused
        # referential continuation turns to emit repeated empty responses.
        if _remembered_lane(turn_id) in {"standard", "standard_compact"}:
            return _remember_lane(turn_id, "standard_compact", bounded_clamp=True), (
                "first tool failure; preserve compact reasoning"
            )
        if user and len(user) <= bounded_chars:
            return _remember_lane(turn_id, "fast", bounded_clamp=True), (
                "first tool failure; bounded recovery"
            )
        return _remember_lane(turn_id, "standard_compact", bounded_clamp=True), (
            "first tool failure; compact recovery"
        )
    if user and len(user) <= bounded_chars and _BOUNDED_ACTION.search(user):
        weak = ""
        if isinstance(evidence, dict) and evidence.get("decision") == "full":
            weak = f"; weak GLiNER {evidence.get('category')} ({float(evidence.get('confidence') or 0.0):.2f})"
        inherited_note = "; inherited prior task" if inherited else ""
        return _remember_lane(turn_id, "fast", bounded_clamp=True), (
            f"short bounded action{inherited_note}{weak}"
        )
    if fast_evidence:
        category = str(evidence.get("category") or "unknown")
        confidence = float(evidence.get("confidence") or 0.0)
        inherited_note = "; inherited prior task" if inherited else ""
        return _remember_lane(turn_id, "fast", bounded_clamp=True), (
            f"GLiNER {category} ({confidence:.2f}){inherited_note}"
        )
    return _remember_lane(turn_id, "standard"), "uncertain; normal harness"


def _turn_key(kwargs: dict[str, Any], messages: list[Any]) -> str:
    """Hermes' turn id, else a stable id for the active real user message.

    Without a key, sticky high reasoning and continuation lane preservation
    silently stop working, so derive one from the session and user row.
    """
    turn_id = str(kwargs.get("turn_id") or "")
    if turn_id:
        return turn_id
    for idx in range(len(messages or []) - 1, -1, -1):
        message = messages[idx]
        if isinstance(message, dict) and message.get("role") == "user" and not _is_synthetic_user(message):
            digest = hashlib.sha1(
                json.dumps(message.get("content"), ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()[:16]
            return f"derived:{kwargs.get('session_id') or kwargs.get('task_id') or ''}:{idx}:{digest}"
    return ""


def _finalize_retry(messages: list[Any]) -> bool:
    """True when finalize already produced an empty/interrupted answer this turn."""
    for message in reversed(_current_turn_messages(messages)[1:]):
        if isinstance(message, dict) and message.get("role") == "user":
            return _is_synthetic_user(message)
        if isinstance(message, dict) and message.get("role") in {"assistant", "tool"}:
            return False
    return False


def on_llm_request(**kwargs: Any):
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None
    model = str(kwargs.get("model") or request.get("model") or "")
    if not _is_glm(model):
        return None
    _, messages = _request_messages(request)
    try:
        api_call_count = int(kwargs.get("api_call_count") or 1)
    except (TypeError, ValueError):
        api_call_count = 1
    turn_id = _turn_key(kwargs, messages)
    evidence = _gliner_evidence(_routing_user_text(messages))
    lane, reason = _decide_lane(
        request, messages, evidence, api_call_count=api_call_count, turn_id=turn_id
    )

    stats: dict[str, int] = {}
    user = _routing_user_text(messages)
    force_mutation = _mutation_gate_needed(messages, user)
    if lane == "fast":
        updated = _apply_effort(request, "low", thinking=False)
        # A prose answer drops tools, but only before the turn has used any:
        # replaying tool calls against a request with no tools makes GLM emit
        # tool markup as text or an empty reply.
        updated, stats = _apply_fast_lane(
            updated,
            no_tools=(
                isinstance(evidence, dict)
                and evidence.get("category") == "quick_response"
                and not _used_tool_names(messages)
            ),
            force_mutation=force_mutation,
        )
        effort = "off"
    elif lane == "standard_compact":
        updated = _apply_effort(request, "low", thinking=True)
        # Recovery gets a little reasoning, not the whole tool universe/history.
        updated, stats = _apply_fast_lane(updated, force_mutation=force_mutation)
        effort = "low"
    elif lane == "finalize":
        # If a thinking-off final already came back empty, give it low reasoning
        # instead of repeating the same empty request.
        retry = _finalize_retry(messages)
        updated = _apply_effort(request, "low", thinking=retry)
        updated, stats = _apply_fast_lane(updated, no_tools=True, finalize=True)
        effort = "low" if retry else "off"
    elif lane == "full":
        updated = _apply_effort(request, "high", thinking=True)
        updated, stats = _apply_history_compaction(updated)
        effort = "high"
    else:
        updated = _apply_effort(request, "low", thinking=True)
        updated, stats = _apply_history_compaction(updated)
        effort = "low"

    if not logger.isEnabledFor(logging.INFO):
        # Serializing a long transcript just to measure it costs real time on
        # every tool hop; skip it when nobody will read the log line.
        return {"request": updated, "source": "glm-codex-effort", "reason": f"{lane}:{reason}"}
    logger.info(
        "glm-system1-router: lane=%s effort=%s reason=%s model=%s call=%s "
        "messages=%s/%s tools=%s/%s payload_chars=%s+%s",
        lane,
        effort,
        reason,
        model,
        api_call_count,
        stats.get("messages_after", len(messages)),
        stats.get("messages_before", len(messages)),
        stats.get("tools_after", len(request.get("tools") or [])),
        stats.get("tools_before", len(request.get("tools") or [])),
        len(json.dumps(_request_messages(updated)[1], ensure_ascii=False, default=str)),
        len(json.dumps(updated.get("tools") or [], ensure_ascii=False, default=str)),
    )
    return {
        "request": updated,
        "source": "glm-codex-effort",
        "reason": f"{lane}:{reason}",
    }


def glm_prompt_section(session_info) -> str:
    model = ""
    if isinstance(session_info, dict):
        model = str(session_info.get("model") or "")
    return _FULL_STEERING if _is_glm(model) else ""


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    ctx.register_middleware("llm_request", on_llm_request)
    register_section = getattr(ctx, "register_system_prompt_section", None)
    if callable(register_section):
        register_section(
            "glm-codex-effort.steer",
            glm_prompt_section,
            position="after_memory",
            max_chars=700,
        )
