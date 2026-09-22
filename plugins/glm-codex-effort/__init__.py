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
"""

from __future__ import annotations

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
    r"\b(?:add|adjust|apply|changes?|check|create|delete|disable|enable|fix|implement|"
    r"inspect|make|move|open|remove|rename|replace|restart|run|set|start|stop|test|"
    r"update|verify)\b",
    re.I,
)
_MUTATION_REQUEST = re.compile(
    r"\b(?:add|adjust|apply|changes?|create|delete|fix|implement|make|move|remove|"
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
_RESULT_COMPLAINT = re.compile(
    r"\b(?:not\s+seeing|can't\s+see|cannot\s+see|didn'?t\s+(?:change|work|apply)|"
    r"doesn'?t\s+(?:change|work|apply)|not\s+reflected|still\s+(?:unchanged|wrong))\b",
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


def _semantic_mutation_intent(evidence: dict[str, Any] | None) -> bool:
    """Use the semantic extractor's intent signal; never infer mutation from a verb list."""
    return isinstance(evidence, dict) and evidence.get("mutation_intent") is True

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

_VERIFY_AFTER_MUTATION_PROMPT = """\
A concrete mutation already landed. Perform at most one proportional verification action now,
then return the concise result. Do not resume discovery, search outside the workspace, create a
second implementation, or revisit project selection.
"""

_FULL_STEERING = """\
GLM adaptive execution:
- Routine work: act with tools promptly and keep visible narration short.
- Complex, ambiguous, failing, multi-file, or architectural work: reason thoroughly.
- After a clear tool result, continue from that evidence; do not re-derive the whole strategy.
- Verification must be proportional to the requested change.
- Never expose private chain-of-thought; provide concise outcomes and evidence.
"""

_WORKSPACE_BOUNDARY_SUFFIX = (
    "\nResolve relative paths from this directory. Search and mutate only this directory "
    "or its descendants unless the user explicitly names another location. Never search "
    "the home directory or choose a similarly named sibling project. If the expected `src` "
    "directory is absent, inspect the workspace's immediate children for the nested app "
    "before selecting any other project."
)


def _setting(key: str, default: Any) -> Any:
    if _CTX is None:
        return default
    try:
        return _CTX.get_config(key, default)
    except Exception:
        return default


def _consume_gliner_evidence() -> dict[str, Any] | None:
    """Read the sibling plugin's request-local signal without a hard dependency."""
    for module in tuple(sys.modules.values()):
        if module is None or not str(getattr(module, "__name__", "")).endswith(".gliner_extract"):
            continue
        consume = getattr(module, "consume_evidence", None)
        if callable(consume):
            try:
                return consume()
            except Exception:
                logger.debug("GLiNER evidence consume failed", exc_info=True)
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


def _workspace_hint(messages: list[Any]) -> str:
    """Extract the trusted cwd line before System-1 compaction drops the host prompt."""
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = re.search(r"(?m)^Current working directory:\s*(.+?)\s*$", content)
        if match:
            return match.group(1).strip()
    return ""


def _strip_attached_context(text: str) -> str:
    for marker in ("\n\n--- Attached Context ---", "\n--- Attached Context ---"):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return text.strip()


def _is_contextual_followup(text: str) -> bool:
    """Recognize a short discourse-only continuation without task details."""
    if not text or len(text) > 160:
        return False
    if (
        re.search(r"\b(?:continue|resume|retry|try|again)\b", text, re.I)
        and re.search(r"\b(?:mentioned|requested|asked|changes?|edits?|work|task)\b", text, re.I)
    ):
        return True
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
    # Prefer the most recent concrete action contract over an intervening
    # complaint such as "I'm not seeing the changes".
    for previous in reversed(texts[:-1]):
        if (
            not _is_contextual_followup(previous)
            and not _RESULT_COMPLAINT.search(previous)
            and _BOUNDED_ACTION.search(previous)
            and _MUTATION_REQUEST.search(previous)
        ):
            return previous, True
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


def _latest_is_empty_recovery(messages: list[Any]) -> bool:
    """Whether the newest user row is Hermes' empty-after-tool recovery nudge.

    A fast-lane response that comes back empty is evidence that thinking-off did
    not finish the tool-followup. Retrying the same lane is byte-for-byte churn:
    keep the bounded context/tools, but enable low reasoning immediately.
    """
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        return isinstance(content, str) and content.strip() == _EMPTY_RECOVERY_TEXT
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

        return _analyze(_current_turn_messages(messages), _load_policy())
    except Exception:
        return {}


def _tool_result_failed(messages: list[Any]) -> bool:
    current = _current_turn_messages(messages)
    for message in reversed(current[-6:]):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            break
        if role != "tool":
            continue
        content = message.get("content")
        if _result_failed(content):
            return True
    return False


def _result_failed(content: Any) -> bool:
    """Classify one tool result without treating a call attempt as success."""
    try:
        from agent.frontier_harness import _failed_result

        return bool(_failed_result(content))
    except Exception:
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
            return bool(parsed.get("error") and not success)
        text = content if isinstance(content, str) else str(content or "")
        return bool(_EXPLICIT_FAILURE_TEXT.match(text))


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


def _bounded_tools(tools: Any, *, forced_tool: str = "", required: bool = False) -> list[Any]:
    if not isinstance(tools, list):
        return []
    configured = _setting("fast_tools", None)
    allowed = set(configured) if isinstance(configured, list) and configured else _FAST_TOOLS
    if forced_tool:
        allowed.add(forced_tool)
    selected = [tool for tool in tools if _tool_name(tool) in allowed]
    # An explicit required-tools request with no matching fast tool must retain
    # its original surface instead of becoming an invalid provider request.
    return tools if required and not selected else selected


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


def _recent_exchange(messages: list[Any], user_idx: int) -> list[Any]:
    """Keep only the endpoints of one completed exchange.

    A completed agent turn may contain dozens of assistant tool calls and tool
    results between its user request and final response.  Those intermediate
    messages are execution trace, not conversational continuity.  Replaying
    them in the next fast request defeats System-1 compaction and makes remote
    prefill dominate latency, so retain only the prior real user message and
    the final text response.
    """
    if user_idx <= 0:
        return []
    assistant_idx = -1
    for idx in range(user_idx - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, dict):
            continue
        if (
            message.get("role") == "assistant"
            and message.get("content")
            and str(message.get("content") or "").strip() != "(empty)"
            and not message.get("_empty_terminal_sentinel")
            and not message.get("tool_calls")
        ):
            assistant_idx = idx
            break
    if assistant_idx < 0:
        return []
    previous_user_idx = -1
    for idx in range(assistant_idx - 1, -1, -1):
        message = messages[idx]
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and not _is_synthetic_user(message)
        ):
            previous_user_idx = idx
            break
    selected = (
        [messages[previous_user_idx], messages[assistant_idx]]
        if previous_user_idx >= 0
        else [messages[assistant_idx]]
    )
    limit = max(160, int(_setting("fast_recent_context_chars", 1200)))
    compacted: list[Any] = []
    for message in selected:
        item = _compact_message(message)
        if isinstance(item, dict) and isinstance(item.get("content"), str) and len(item["content"]) > limit:
            item = dict(item)
            item["content"] = item["content"][:limit] + "\n[recent context truncated]"
        compacted.append(item)
    return compacted


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
    effective_user, inherited = _routing_user_context(messages)
    system_prompt = _FAST_SYSTEM_PROMPT
    workspace = _workspace_hint(messages)
    if workspace:
        system_prompt += (
            "\nWorkspace boundary: " + workspace +
            _WORKSPACE_BOUNDARY_SUFFIX
        )
    if inherited and effective_user:
        system_prompt += (
            "\nThe current message is a referential retry. Its original task is:\n"
            + effective_user[:1200]
        )
    return [
        {"role": "system", "content": system_prompt},
        *recent,
        *map(_compact_message, tail),
    ]


def _append_workspace_boundary(request: dict[str, Any]) -> dict[str, Any]:
    """Keep the trusted workspace rule when a high/full lane rebuilds context."""
    updated = dict(request)
    message_key, messages = _request_messages(updated)
    workspace = _workspace_hint(messages)
    if not message_key or not workspace or not messages:
        return updated
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return updated
    system = dict(first)
    content = str(system.get("content") or "")
    marker = "Workspace boundary: " + workspace
    if marker not in content:
        system["content"] = content + "\n" + marker + _WORKSPACE_BOUNDARY_SUFFIX
        messages = [system, *messages[1:]]
        updated[message_key] = messages
    return updated


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
    recent = _recent_exchange(messages, user_idx)
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
        updated = _append_workspace_boundary(updated)
    return updated, {
        "messages_before": before_messages,
        "messages_after": len(updated.get(message_key) or []) if message_key else before_messages,
        "tools_before": before_tools,
        "tools_after": before_tools,
    }


def _apply_fast_lane(
    request: dict[str, Any], *, no_tools: bool = False, full_tools: bool = False,
    force_mutation: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    updated = dict(request)
    message_key, messages = _request_messages(updated)
    before_messages = len(messages)
    before_tools = len(updated.get("tools") or []) if isinstance(updated.get("tools"), list) else 0
    if message_key:
        updated[message_key] = _compact_current_turn(messages)
        if force_mutation and updated[message_key]:
            system = updated[message_key][0]
            if isinstance(system, dict) and system.get("role") == "system":
                system = dict(system)
                system["content"] = str(system.get("content") or "") + "\n" + _MUTATION_GATE_PROMPT
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
        )
        if force_mutation:
            mutation_tools = [
                tool for tool in updated["tools"] if _tool_name(tool) in _DIRECT_MUTATION_TOOLS
            ]
            if mutation_tools:
                updated["tools"] = mutation_tools
    try:
        fast_max = max(512, int(_setting("fast_max_tokens", 4096)))
    except (TypeError, ValueError):
        fast_max = 4096
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


def _enforce_edit_action(
    request: dict[str, Any], *, require_action: bool, force_mutation: bool,
) -> dict[str, Any]:
    """Preserve the explicit-edit contract across every reasoning lane."""
    updated = dict(request)
    tools = updated.get("tools")
    if not isinstance(tools, list) or not tools:
        return updated
    if force_mutation:
        tools = [
            tool for tool in tools
            if _tool_name(tool) in _DIRECT_MUTATION_TOOLS
        ]
        if tools:
            updated["tools"] = tools
    if require_action and updated.get("tools"):
        updated["tool_choice"] = "required"
    return updated


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
    rank = {
        "fast": 0,
        "standard_compact": 1,
        "verify_compact": 2,
        "standard": 3,
        "full": 4,
        "finalize": 5,
        # A finalize request that itself returned empty must be allowed to
        # recover with reasoning. Keep tools closed, but never repeat the
        # exact thinking-off request that just failed.
        "finalize_reasoning": 6,
    }
    now = time.monotonic()
    with _TURN_LOCK:
        for key, (_, touched) in list(_TURN_LANES.items()):
            if now - touched > 3600:
                _TURN_LANES.pop(key, None)
        previous = _TURN_LANES.get(turn_id, ("fast", now))[0]
        # A bounded policy decision may repair an earlier accidental broad/low
        # classification. It must never demote full/high or reopen finalization.
        if bounded_clamp and previous in {"fast", "standard_compact", "verify_compact", "standard"}:
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


def _tool_activity_records(messages: list[Any]) -> list[dict[str, Any]]:
    """Return current-turn tool calls paired with their actual results."""
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for message in _current_turn_messages(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id and call_id in by_id:
                by_id[call_id]["result"] = message.get("content")
            continue
        if message.get("role") != "assistant":
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
            record = {"name": name, "result": None}
            records.append(record)
            call_id = str(call.get("id") or call.get("call_id") or "")
            if call_id:
                by_id[call_id] = record
    return records


def _mutation_landed(name: str, result: Any) -> bool:
    if result is None or name not in _DIRECT_MUTATION_TOOLS:
        return False
    if name in {"patch", "write_file"}:
        try:
            from agent.tool_result_classification import file_mutation_result_landed

            return bool(file_mutation_result_landed(name, result))
        except Exception:
            return False
    return not _result_failed(result)


def _mutation_failure_recovery(messages: list[Any]) -> bool:
    """Whether the latest failure can invalidate an edit/follow-up verification."""
    records = _tool_activity_records(messages)
    landed_before = False
    for record in records:
        if _mutation_landed(record["name"], record["result"]):
            landed_before = True
        if record["result"] is not None and _result_failed(record["result"]):
            if record["name"] in _DIRECT_MUTATION_TOOLS or landed_before:
                return True
    return False


def _tool_activity(messages: list[Any]) -> tuple[int, int]:
    """Return read calls and proven-successful direct mutations this turn."""
    records = _tool_activity_records(messages)
    reads = sum(record["name"] in _READ_ONLY_TOOLS for record in records)
    mutations = sum(
        _mutation_landed(record["name"], record["result"]) for record in records
    )
    return reads, mutations


def _reads_after_last_mutation(messages: list[Any]) -> tuple[int, int]:
    """Return landed mutations and successful subsequent verification reads."""
    records = _tool_activity_records(messages)
    mutation_indexes = [
        idx for idx, record in enumerate(records)
        if _mutation_landed(record["name"], record["result"])
    ]
    if not mutation_indexes:
        return 0, 0
    last = mutation_indexes[-1]
    verified_reads = sum(
        record["name"] in _READ_ONLY_TOOLS
        and record["result"] is not None
        and not _result_failed(record["result"])
        for record in records[last + 1:]
    )
    return len(mutation_indexes), verified_reads


def _mutation_gate_needed(
    messages: list[Any], user: str, evidence: dict[str, Any] | None = None,
) -> bool:
    if not _semantic_mutation_intent(evidence):
        return False
    reads, mutations = _tool_activity(messages)
    try:
        threshold = max(1, int(_setting("max_reads_before_edit", 3)))
    except (TypeError, ValueError):
        threshold = 3
    return mutations == 0 and reads >= threshold


def _mutation_action_required(
    messages: list[Any], user: str, evidence: dict[str, Any] | None = None,
) -> bool:
    """Require at least one real action before an explicit edit may finalize.

    Tool availability alone is not enough: a model can still answer in prose and
    claim an edit landed. Requiring a tool until the first direct mutation makes
    that false-success path structurally unavailable.
    """
    if not (
        _semantic_mutation_intent(evidence)
        and _BOUNDED_ACTION.search(user or "")
    ):
        return False
    _, mutations = _tool_activity(messages)
    return mutations == 0


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
    if _latest_is_empty_recovery(messages):
        if remembered in {"finalize", "finalize_reasoning"}:
            return _remember_lane(turn_id, "finalize_reasoning"), (
                "empty finalization recovery; enable reasoning"
            )
        return _remember_lane(turn_id, "standard_compact", bounded_clamp=True), (
            "empty-after-tool recovery; enable compact reasoning"
        )
    if remembered and _latest_is_system_continuation(messages):
        return _remember_lane(turn_id, remembered), "synthetic continuation; preserve active lane"
    if _mutation_failure_recovery(messages):
        return _remember_lane(turn_id, "standard_compact", bounded_clamp=True), (
            "latest edit/verification failed; keep recovery tools available"
        )
    mutations, post_mutation_reads = _reads_after_last_mutation(messages)
    if mutations and post_mutation_reads >= 1:
        return _remember_lane(turn_id, "finalize_reasoning"), (
            "mutation landed and verification attempted; finalize"
        )
    if mutations:
        return _remember_lane(turn_id, "verify_compact", bounded_clamp=True), (
            "mutation landed; allow one proportional verification"
        )
    if len(user) > int(_setting("full_request_chars", 1800)) or _HARD_COMPLEX_GUARD.search(user):
        return _remember_lane(turn_id, "full"), "complex guard"
    bounded_failure_recovery = (
        failures == 2
        and repeated < 2
        and user
        and len(user) <= int(_setting("bounded_action_max_chars", 500))
        and bool(_BOUNDED_ACTION.search(user))
    )
    if bounded_failure_recovery:
        return _remember_lane(turn_id, "standard_compact", bounded_clamp=True), (
            "two distinct bounded tool failures; compact reasoning recovery"
        )
    if failures >= 2 or repeated >= 2:
        return _remember_lane(turn_id, "full"), f"failure={failures} repeated={repeated}"
    if isinstance(evidence, dict) and str(evidence.get("decision") or "") == "full":
        category = str(evidence.get("category") or "unknown")
        confidence = float(evidence.get("confidence") or 0.0)
        full_confidence = float(_setting("full_confidence", 0.60))
        if confidence >= full_confidence:
            return _remember_lane(turn_id, "full"), f"GLiNER {category} ({confidence:.2f})"
    try:
        bounded_chars = max(80, int(_setting("bounded_action_max_chars", 500)))
    except (TypeError, ValueError):
        bounded_chars = 500
    try:
        max_calls = max(1, int(_setting("fast_max_calls", 6)))
    except (TypeError, ValueError):
        max_calls = 6
    effective_calls = _effective_fast_calls(messages, api_call_count)
    try:
        hard_max_calls = max(max_calls + 1, int(_setting("fast_hard_max_calls", 10)))
    except (TypeError, ValueError):
        hard_max_calls = 10
    if effective_calls > hard_max_calls:
        return _remember_lane(turn_id, "finalize"), (
            f"bounded action exhausted ({effective_calls}>{hard_max_calls}); force final"
        )
    if effective_calls > max_calls:
        return _remember_lane(turn_id, "standard_compact", bounded_clamp=True), (
            f"fast action budget exceeded ({effective_calls}>{max_calls}); compact low"
        )
    if failures == 1 or _tool_result_failed(messages):
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
    if isinstance(evidence, dict):
        decision = str(evidence.get("decision") or "")
        category = str(evidence.get("category") or "unknown")
        confidence = float(evidence.get("confidence") or 0.0)
        if decision == "fast" and confidence >= float(_setting("fast_confidence", 0.30)):
            inherited_note = "; inherited prior task" if inherited else ""
            return _remember_lane(turn_id, "fast", bounded_clamp=True), (
                f"GLiNER {category} ({confidence:.2f}){inherited_note}"
            )
    return _remember_lane(turn_id, "standard"), "uncertain; normal harness"


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
    turn_id = str(kwargs.get("turn_id") or "")
    evidence = _consume_gliner_evidence()
    lane, reason = _decide_lane(
        request, messages, evidence, api_call_count=api_call_count, turn_id=turn_id
    )

    stats: dict[str, int] = {}
    user = _routing_user_text(messages)
    force_mutation = _mutation_gate_needed(messages, user, evidence) or _mutation_failure_recovery(messages)
    require_mutation_action = _mutation_action_required(messages, user, evidence)
    if lane == "fast":
        updated = _apply_effort(request, "low", thinking=False)
        if require_mutation_action and isinstance(updated.get("tools"), list) and updated["tools"]:
            updated["tool_choice"] = "required"
        # GLiNER evidence is advisory. A stale/incorrect quick-response label
        # must never remove tools from an explicit action or edit request.
        prose_only = (
            isinstance(evidence, dict)
            and evidence.get("category") == "quick_response"
            and not _BOUNDED_ACTION.search(user or "")
            and not _semantic_mutation_intent(evidence)
        )
        updated, stats = _apply_fast_lane(
            updated,
            no_tools=prose_only,
            force_mutation=force_mutation,
        )
        effort = "off"
    elif lane == "standard_compact":
        updated = _apply_effort(request, "low", thinking=True)
        if require_mutation_action and isinstance(updated.get("tools"), list) and updated["tools"]:
            updated["tool_choice"] = "required"
        # Recovery gets a little reasoning, not the whole tool universe/history.
        updated, stats = _apply_fast_lane(updated, force_mutation=force_mutation)
        effort = "low"
    elif lane == "verify_compact":
        updated = _apply_effort(request, "low", thinking=True)
        updated, stats = _apply_fast_lane(updated)
        message_key, compact_messages = _request_messages(updated)
        if message_key and compact_messages and isinstance(compact_messages[0], dict):
            system = dict(compact_messages[0])
            system["content"] = str(system.get("content") or "") + "\n" + _VERIFY_AFTER_MUTATION_PROMPT
            compact_messages[0] = system
            updated[message_key] = compact_messages
        effort = "low"
    elif lane == "finalize":
        updated = _apply_effort(request, "low", thinking=False)
        updated, stats = _apply_fast_lane(updated, no_tools=True)
        effort = "off"
    elif lane == "finalize_reasoning":
        updated = _apply_effort(request, "low", thinking=True)
        updated, stats = _apply_fast_lane(updated, no_tools=True)
        effort = "low"
    elif lane == "full":
        updated = _apply_effort(request, "high", thinking=True)
        updated, stats = _apply_history_compaction(updated)
        effort = "high"
    else:
        updated = _apply_effort(request, "low", thinking=True)
        updated, stats = _apply_history_compaction(updated)
        effort = "low"

    updated = _enforce_edit_action(
        updated, require_action=require_mutation_action, force_mutation=force_mutation,
    )

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
