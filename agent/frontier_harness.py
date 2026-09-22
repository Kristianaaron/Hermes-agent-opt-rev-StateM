"""Frontier-style execution policy for tool-using local models.

This module is deliberately a control plane, not another agent loop.  It
coordinates Hermes' existing StateM prompts, todo store, checkpoint manager,
delegates, compaction, verification evidence, and tool-loop guardrails.

The hot path performs no model calls.  It infers the current execution phase
from the request history, applies a phase-sized thinking budget, and writes a
small resumable trajectory ledger.  Every operation is fail-open: a policy or
telemetry failure must never prevent the underlying provider request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.task_intent import is_internal_control_text, user_wants_action


_PROTOCOL_MARKER = "[HERMES_FRONTIER_HARNESS_V1]"
_STATE_MARKER = "[HERMES_FRONTIER_STATE_V1]"
_PROTOCOL = f"""

{_PROTOCOL_MARKER}
Use a bounded, evidence-driven execution loop:
1. Classify the task from the user's latest prose first. Host boilerplate such as "Examine it with the vision_analyze tool" is not a user instruction. For README, banner, descriptor, docs, or copy-file work, skip localization and write those files. For code changes, localize the smallest relevant file/symbol set before editing. Prefer concise search results and read windows of at most 120 lines; treat empty output as explicit evidence, not a reason to repeat the same call.
2. Maintain one active task contract and one active phase: localize, plan, execute, verify, repair, or handoff. Preserve durable commitments in the todo/StateM artifacts and keep only the latest five useful tool-result pairs active; summarize older evidence.
3. Before every tool call, validate the tool name and required argument shape. Batch independent read-only calls. Never repeat an unchanged failed or no-progress call.
4. Mutations are checkpoint-backed. After a failed verification, diagnose once, then change the strategy or roll back the affected attempt; do not stack speculative edits.
5. Completion requires evidence that matches the change. Inspect generated artifacts directly. README, banner, descriptor, docs, and copy-file tasks complete when those files are written — do not search the home directory, clone, or run pytest/ruff/mypy unless the user asked for tests or executable code changed. If verification is unavailable for a code change, report the concrete blocker and do not claim success.
6. Candidate search is exceptional and bounded: after ambiguity or two failed repair boundaries, use at most two isolated candidates when delegation is available, choose with explicit tests/evidence, and merge only the winner.
7. Stop immediately on user pause/cancel/redirect. On iteration pressure or interruption, persist a concise handoff containing objective, completed work, current phase, evidence, blockers, and exact next action. Answer questions about the previous turn from its record. Resume work only when the user asks to continue or finish it. Do not create a file the user did not ask for.
8. Missing operator information is a hard interaction boundary. Ask one focused question as the final response and stop. Never select a default, read a protected approval source, or continue because the operator is unavailable or the session is one-shot.
9. Treat mutations as non-atomic effects: use the supplied effect identity, verify before retry after timeout/interruption, and never replay an effect whose outcome is unknown.
10. If the live phase packet reports path drift or a repair boundary, stop issuing tools, inspect the last failure, change strategy once, and require fresh verification before continuing.
11. The user's latest prose is the task contract. Host boilerplate such as "Examine it with the vision_analyze tool" is not a user instruction. SVG and other source/asset attachments are copied with write_file; do not describe them.
12. Verification must match the change. README, banner, descriptor, docs, and copy-file tasks complete when those files are written. Do not search the home directory, clone, or run pytest/ruff/mypy unless the user asked for tests or the edited files are executable code.
13. If the user's latest prose is a ping or status ask ("are you there", "where are we", "share an update"), answer in text immediately with zero tool calls. Do not poll, SSH, or execute_code instead of replying. A question about the previous turn asks for an explanation, not another execution.
14. Codex turn rule: a turn is incomplete until you emit a user-visible assistant message. Hidden reasoning is not a reply. After tools, write one short status sentence before the next tool batch. Do not retry a memory replace/remove or patch whose anchor (old_text/old_string) was not found; read current state and change the anchor, or skip that write and continue the user's task. When tools are omitted mid-turn (commentary gate), the text you write is status commentary — the turn continues and tools return on the next call; do not treat that sentence as the final answer. Once evidence is sufficient for the requested action, perform that action. Do not keep inspecting, and do not write a note about the stop.
Do not expose hidden chain-of-thought. Surface concise phase, action, evidence, and blocker summaries instead.
""".rstrip()

_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    # The execution protocol is provider-neutral. Restrict only when a
    # deployment explicitly opts into a model allowlist; DeepSeek-specific
    # wire fields stay isolated in the transport layer.
    "model_patterns": [],
    "exclude_patterns": ["embedding", "rerank"],
    "inject_protocol": True,
    "persist_trajectory": True,
    "send_thinking_token_budget": False,
    "recent_tool_pairs": 5,
    "max_candidates": 2,
    "max_repairs_per_failure": 2,
    "escalate_after_failures": 2,
    "path_deviation_failure_streak": 2,
    "max_unproductive_actions": 4,
    "reasoning_budgets": {
        "localize": 768,
        "plan": 1280,
        "execute": 512,
        "verify": 768,
        "repair": 1280,
        "final": 768,
        "escalate": 2048,
    },
}

_MUTATING_TOOLS = {
    "apply_patch", "patch", "write_file", "replace_file", "edit_file",
    "notebook_edit", "str_replace_editor",
}
_VERIFICATION_TOOLS = {"test", "lint", "typecheck", "verify", "validate"}
_FAILURE_WORDS = re.compile(
    r"(?:traceback|timed?\s*out|timeout|http\s*50[0234]|permission denied|"
    r"invalid (?:tool|argument|command)|exit(?:ed)?\s+(?:code\s*)?[1-9]|"
    r"\bfailed\b|\berror\b)",
    re.IGNORECASE,
)
_EXPLICIT_FAILURE_TEXT = re.compile(
    r"^\s*(?:traceback\s*\(|(?:tool\s+)?errors?\s*:|failed\s*:|failure\s*:|"
    r"command\s+(?:failed|timed?\s*out)|\[command\s+timed?\s*out|blocked\s*:|"
    r"permission\s+denied\b|unauthorized\b|http\s+(?:401|403|429|5\d\d)\b|"
    r"exit\s+code\s*[1-9]\d*\b)",
    re.IGNORECASE,
)
_CONTEXTUAL_FOLLOWUP_ACTIONS = {"continue", "proceed", "resume", "retry", "rety", "try"}
_CONTEXTUAL_FOLLOWUP_FILLER = {
    "a", "again", "ahead", "and", "approach", "can", "change", "could", "different",
    "fix", "from", "go", "implementation", "it", "just", "last", "now", "off", "on",
    "please", "request", "same", "task", "that", "the", "then", "this", "to", "where",
    "with", "work", "working", "you",
}
def looks_like_edit_request(text: str) -> bool:
    """Backward-compatible name: any requested action, not one UI example."""
    return user_wants_action(text)
_VERIFY_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:pytest|python\s+-m\s+pytest|npm\s+(?:test|run\s+(?:test|lint|typecheck|check))|"
    r"pnpm\s+(?:test|lint|typecheck|check)|yarn\s+(?:test|lint|typecheck|check)|"
    r"cargo\s+(?:test|check|clippy)|go\s+test|ruff\s+check|mypy|pyright|"
    r"tsc(?:\s|$)|make\s+(?:test|check)|gradle\s+test|mvn\s+test)",
    re.IGNORECASE,
)
_MUTATE_COMMAND = re.compile(
    r"(?:sed\s+-i|perl\s+-[pi]|\b(?:rm|mv|cp|chmod|chown|mkdir|touch)\b|"
    r"(?:^|\s)(?:>|>>)(?:\s|$)|git\s+(?:apply|commit|merge|rebase))",
    re.IGNORECASE,
)

_config_lock = threading.Lock()
_config_cache: tuple[int, dict[str, Any]] | None = None


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _merge(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _load_policy() -> dict[str, Any]:
    """Load the small policy block, cached by config mtime."""
    global _config_cache
    path = _hermes_home() / "config.yaml"
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return dict(_DEFAULTS)
    with _config_lock:
        if _config_cache and _config_cache[0] == stamp:
            return _config_cache[1]
        try:
            import yaml

            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            agent_cfg = raw.get("agent") if isinstance(raw, dict) else {}
            frontier_cfg = (
                agent_cfg.get("frontier_harness")
                if isinstance(agent_cfg, dict)
                else {}
            )
            policy = _merge(_DEFAULTS, frontier_cfg if isinstance(frontier_cfg, dict) else {})
        except Exception:
            policy = dict(_DEFAULTS)
        _config_cache = (stamp, policy)
        return policy


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(content)
    return str(content or "")


def _tool_call(call: Any) -> tuple[str, Any, str]:
    if not isinstance(call, Mapping):
        return "", {}, ""
    function = call.get("function")
    function = function if isinstance(function, Mapping) else {}
    name = str(function.get("name") or call.get("name") or "")
    arguments = function.get("arguments", call.get("arguments", {}))
    call_id = str(call.get("id") or call.get("tool_call_id") or "")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, ValueError):
        parsed = arguments
    return name, parsed, call_id


def _failed_result(content: Any) -> bool:
    parsed = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = None
    if isinstance(parsed, Mapping):
        status = str(parsed.get("status") or "").lower()
        success = (
            parsed.get("success") is True
            or parsed.get("ok") is True
            or parsed.get("isError") is False
            or status in {"ok", "success", "completed", "complete"}
        )
        if (
            parsed.get("success") is False
            or parsed.get("ok") is False
            or parsed.get("isError") is True
            or status in {"error", "failed", "failure"}
        ):
            return True
        exit_code = parsed.get("exit_code", parsed.get("returncode"))
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        if isinstance(exit_code, int) and exit_code == 0:
            # Successful terminal output can quote the word "error" (source
            # dumps, compiler headers). That is not a failed tool call.
            return False
        if parsed.get("error") and not success:
            return True
        # Structured success (including skill/read payloads) is authoritative;
        # never scan arbitrary content fields for failure vocabulary.
        return False
    text = _content_text(content)
    if re.search(r"\b(?:0 errors|0 failures|errors?:\s*0|passed)\b", text, re.I):
        return False
    # Unstructured results are successful by default. Only explicit failure
    # envelopes count; source and documentation may mention error handling.
    return bool(_EXPLICIT_FAILURE_TEXT.match(text[:8000]))


def _is_mutation(name: str, arguments: Any) -> bool:
    lowered = name.lower()
    if lowered in _MUTATING_TOOLS:
        return True
    if lowered == "terminal":
        command = arguments.get("command", "") if isinstance(arguments, Mapping) else ""
        return bool(_MUTATE_COMMAND.search(str(command)))
    return False


def _is_verification(name: str, arguments: Any) -> bool:
    lowered = name.lower()
    if lowered in _VERIFICATION_TOOLS:
        return True
    if lowered == "terminal":
        command = arguments.get("command", "") if isinstance(arguments, Mapping) else ""
        return bool(_VERIFY_COMMAND.search(str(command)))
    return False


def _path_health(events: Sequence[Mapping[str, Any]], repeated: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Classify trajectory drift without making another model call."""
    failure_limit = max(2, int(policy.get("path_deviation_failure_streak", 2)))
    trailing_failures = 0
    for event in reversed(events):
        if event.get("status") != "failed":
            break
        trailing_failures += 1
    last_progress = max(
        (index for index, event in enumerate(events)
         if event.get("status") == "ok" and (event.get("mutation") or event.get("verification"))),
        default=-1,
    )
    unproductive = max(0, len(events) - last_progress - 1) if last_progress >= 0 else len(events)
    if trailing_failures >= failure_limit or repeated >= failure_limit:
        health = "path_drift"
        next_action = "stop tools; inspect the last failure, change strategy, then verify"
    elif last_progress >= 0 and unproductive >= max(2, int(policy.get("max_unproductive_actions", 4))):
        health = "repair_boundary"
        next_action = "stop speculative exploration; use the existing evidence to repair or verify"
    elif events and events[-1].get("mutation") and not events[-1].get("verification"):
        health = "verification_required"
        next_action = "perform fresh deterministic verification before another mutation"
    else:
        health = "on_path"
        next_action = "continue the current phase with bounded, evidence-producing actions"
    return {
        "health": health,
        "trailing_failure_streak": trailing_failures,
        "unproductive_actions": unproductive,
        "next_action": next_action,
    }


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


def _is_synthetic_user(message: Mapping[str, Any]) -> bool:
    if str(message.get("role") or "") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return True
    if str(message.get("display_kind") or "") in _SYNTHETIC_DISPLAY_KINDS:
        return True
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else ""
    return (
        text.startswith(_AUTO_CONTINUE_PREFIX)
        or text == _EMPTY_RECOVERY_TEXT
        or is_internal_control_text(text)
    )


def _is_contextual_followup(text: str) -> bool:
    if not text or len(text) > 120:
        return False
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words or not (_CONTEXTUAL_FOLLOWUP_ACTIONS & set(words)):
        return False
    return all(
        word in _CONTEXTUAL_FOLLOWUP_ACTIONS or word in _CONTEXTUAL_FOLLOWUP_FILLER
        for word in words
    )


def _effective_user_goal(messages: Sequence[Mapping[str, Any]]) -> str:
    indexed = [
        (index, _content_text(message.get("content"))[:1200])
        for index, message in enumerate(messages)
        if str(message.get("role") or "") == "user" and not _is_synthetic_user(message)
    ]
    if not indexed:
        return ""
    latest = indexed[-1][1]
    if not _is_contextual_followup(latest):
        return latest
    for _, previous in reversed(indexed[:-1]):
        if not _is_contextual_followup(previous):
            return previous
    return latest


def _active_turn(messages: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Scope control-plane evidence to the latest real user turn.

    A failed task must still escalate within its own tool loop, but its failure
    count and repeated-call signatures must not poison the next user request in
    the same durable session.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role") or "") != "user":
            continue
        if _is_synthetic_user(message):
            continue
        return messages[index:]
    return messages


def _analyze(messages: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    calls: dict[str, tuple[str, Any, str]] = {}
    signatures: list[str] = []
    events: list[dict[str, Any]] = []
    latest_goal = _effective_user_goal(messages)

    active_messages = _active_turn(messages)
    for message in active_messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            for raw_call in message.get("tool_calls") or []:
                name, arguments, call_id = _tool_call(raw_call)
                if not name:
                    continue
                canonical = json.dumps(
                    {"name": name, "arguments": arguments},
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
                signature = hashlib.sha256(canonical.encode()).hexdigest()[:16]
                signatures.append(signature)
                if call_id:
                    calls[call_id] = (name, arguments, signature)
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name, arguments, signature = calls.get(
                call_id,
                (str(message.get("name") or "unknown"), {}, ""),
            )
            failed = _failed_result(message.get("content"))
            events.append({
                "tool": name,
                "signature": signature,
                "status": "failed" if failed else "ok",
                "mutation": _is_mutation(name, arguments),
                "verification": _is_verification(name, arguments),
                "summary": _content_text(message.get("content")).strip()[:240],
                "result_hash": hashlib.sha256(_content_text(message.get("content")).encode()).hexdigest()[:16],
            })

    counts = Counter(signatures)
    repeated = max(counts.values(), default=0)
    failures = sum(event["status"] == "failed" for event in events)
    last_mutation = max((i for i, event in enumerate(events) if event["mutation"]), default=-1)
    last_passing_verify = max(
        (i for i, event in enumerate(events) if event["verification"] and event["status"] == "ok"),
        default=-1,
    )

    latest_lower = latest_goal.lower()
    if any(word in latest_lower for word in ("final response", "final report", "summarize findings")):
        phase = "final"
    elif events and events[-1]["status"] == "failed":
        phase = "repair"
    elif repeated >= 2:
        phase = "repair"
    elif last_mutation > last_passing_verify:
        phase = "verify"
    elif events:
        phase = "execute"
    elif len(active_messages) <= 3:
        phase = "localize"
    else:
        phase = "plan"

    escalate_at = max(1, int(policy.get("escalate_after_failures", 2)))
    escalated = failures >= escalate_at or repeated >= escalate_at
    budget_phase = "escalate" if escalated else phase
    budgets = policy.get("reasoning_budgets") or {}
    budget = int(budgets.get(budget_phase, budgets.get("plan", 1024)))
    recent_count = max(1, min(int(policy.get("recent_tool_pairs", 5)), 20))
    recent_events = events[-recent_count:]
    path = _path_health(recent_events, repeated, policy)

    return {
        "phase": phase,
        "budget_phase": budget_phase,
        "thinking_token_budget": max(128, min(budget, 8192)),
        "tool_call_count": len(signatures),
        "failure_count": failures,
        "max_repeated_signature": repeated,
        "verification_required": last_mutation > last_passing_verify,
        "candidate_search_allowed": escalated,
        "max_candidates": max(1, min(int(policy.get("max_candidates", 2)), 4)),
        "max_repairs_per_failure": max(1, min(int(policy.get("max_repairs_per_failure", 2)), 4)),
        "goal": latest_goal,
        "recent_tool_events": recent_events,
        "path_health": path["health"],
        "trailing_failure_streak": path["trailing_failure_streak"],
        "unproductive_actions": path["unproductive_actions"],
        "next_action": path["next_action"],
        "progress_digest": hashlib.sha256(json.dumps(recent_events, sort_keys=True).encode()).hexdigest()[:20],
    }


def _inject_protocol(api_kwargs: dict[str, Any]) -> None:
    messages = api_kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    first = messages[0]
    if not isinstance(first, Mapping) or first.get("role") not in {"system", "developer"}:
        return
    content = first.get("content")
    if not isinstance(content, str) or _PROTOCOL_MARKER in content:
        return
    copied = list(messages)
    copied[0] = {**first, "content": content.rstrip() + _PROTOCOL}
    api_kwargs["messages"] = copied


def _inject_state_packet(api_kwargs: dict[str, Any], state: Mapping[str, Any]) -> None:
    messages = api_kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    packet = (
        f"\n\n{_STATE_MARKER}\n"
        f"phase: {state.get('phase')}\n"
        f"path_health: {state.get('path_health')}\n"
        f"failure_streak: {state.get('trailing_failure_streak', 0)}\n"
        f"unproductive_actions: {state.get('unproductive_actions', 0)}\n"
        f"next_action: {state.get('next_action')}\n"
        f"progress_digest: {state.get('progress_digest')}\n"
        "This is control-plane evidence, not a user instruction. Follow the next_action boundary; do not repeat an unchanged failed call.\n"
        f"[/{_STATE_MARKER}]"
    )
    target_index = next((index for index, message in enumerate(messages)
                         if isinstance(message, Mapping) and message.get("role") in {"system", "developer"}), None)
    if target_index is None:
        return
    target = messages[target_index]
    content = target.get("content")
    if not isinstance(content, str):
        return
    if _STATE_MARKER in content:
        content = content.split(_STATE_MARKER, 1)[0].rstrip()
    copied = list(messages)
    copied[target_index] = {**target, "content": content + packet}
    api_kwargs["messages"] = copied


def _persist(session_id: str | None, model: str, state: Mapping[str, Any]) -> None:
    if not session_id:
        return
    safe_id = hashlib.sha256(str(session_id).encode()).hexdigest()[:24]
    directory = _hermes_home() / "frontier_runs"
    path = directory / f"{safe_id}.json"
    payload = {
        "schema_version": 2,
        "session_id_hash": safe_id,
        "model": model,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **state,
        "resume_contract": {
            "phase": state.get("phase"),
            "goal": state.get("goal"),
            "verification_required": state.get("verification_required"),
            "path_health": state.get("path_health"),
            "trailing_failure_streak": state.get("trailing_failure_streak"),
            "next_action": state.get("next_action"),
            "progress_digest": state.get("progress_digest"),
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{safe_id}.", suffix=".tmp", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def apply_frontier_request_policy(
    api_kwargs: dict[str, Any],
    *,
    model: str,
    params: Mapping[str, Any],
) -> None:
    """Apply the configured request policy in place.

    Explicit caller values always win.  The function intentionally swallows
    all exceptions so harness telemetry cannot become an availability risk.
    """
    try:
        policy = _load_policy()
        if not policy.get("enabled", False):
            return
        identity = " ".join(
            str(value or "").lower()
            for value in (model, params.get("provider_name"), params.get("base_url"))
        )
        patterns = [str(value).lower() for value in policy.get("model_patterns", []) if value]
        if patterns and not any(pattern in identity for pattern in patterns):
            return
        excluded = [str(value).lower() for value in policy.get("exclude_patterns", []) if value]
        if excluded and any(pattern in identity for pattern in excluded):
            return

        messages = api_kwargs.get("messages") or []
        state = _analyze(messages, policy)
        if policy.get("inject_protocol", True):
            _inject_protocol(api_kwargs)
            _inject_state_packet(api_kwargs, state)

        if policy.get("send_thinking_token_budget", False):
            extra_body = api_kwargs.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
                api_kwargs["extra_body"] = extra_body
            if "thinking_token_budget" not in extra_body and "thinking_token_budget" not in api_kwargs:
                extra_body["thinking_token_budget"] = state["thinking_token_budget"]

        if policy.get("persist_trajectory", True):
            _persist(params.get("session_id"), model, state)
    except Exception:
        return


__all__ = ["apply_frontier_request_policy"]
