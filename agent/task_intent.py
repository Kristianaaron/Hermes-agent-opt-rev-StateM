"""Small, conservative signals for routing user requests.

These are hints for tool availability, not a replacement for the model's task
judgment. In particular, a prohibition in one clause must not cancel a
separate affirmative request in another clause.
"""

from __future__ import annotations

import re


_FILE_ACTION = re.compile(
    r"\b(?:add|adjust|apply|bump|change|click|complete|create|decrease|delete|"
    r"edit|finish|fix|implement|increase|lower|make|modify|move|patch|raise|"
    r"reduce|remove|rename|replace|restyle|save|set|submit|tweak|update|write)\b",
    re.I,
)
_PROCESS_ACTION = re.compile(r"\b(?:build|deploy|install|launch|restart|run|start)\b", re.I)
_STOP_PROCESS = re.compile(
    r"\bstop\s+(?:(?:the|my|a|this|local|development)\s+){0,2}"
    r"(?:server|preview|service|process|app|container|daemon|worker|job)\b", re.I
)
_READ_ONLY_OPENING = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you|just)\s+)?"
    r"(?:read|show|list|inspect|review|look|find|search|explain|describe|"
    r"tell|what|why|how|when|where|who|which)\b",
    re.I,
)
_ACTION_LINK = re.compile(
    r"(?:\b(?:and|then|also|but)\b|[;.!?])\s*"
    r"(?:(?:please|can you|could you)\s+)?$",
    re.I,
)
_NEGATION = re.compile(r"\b(?:do not|don't|dont|never|without)\b(?:\s+\w+){0,3}\s*$", re.I)
_TURN_META = re.compile(
    r"\b(?:why did you|what happened|what did you do|you said|"
    r"you (?:didn'?t|did not|stopped|quit|halted|gave up)|"
    r"your (?:last|previous) (?:reply|message|response)|"
    r"that (?:message|reply)|you were supposed to)\b",
    re.I,
)
_INTERNAL_CONTROL_PREFIXES = (
    "[system: continue now.",
    "[system: your previous response contained only internal reasoning",
    "[system: your previous message ended the turn with a fragment",
    "[system: you edited code in this turn, but the workspace",
    "[system: your previous response was truncated by the output length limit",
    "[system: the previous response was cut off by a network error",
    "[system: your previous tool call ",
)


def requested_action_kinds(text: str) -> set[str]:
    """Return affirmative file/UI and process actions, ignoring local negations."""
    value = (text or "").strip()
    if not value:
        return set()
    candidates = sorted(
        [(match.start(), match.end(), "file") for match in _FILE_ACTION.finditer(value)]
        + [(match.start(), match.end(), "process") for match in _PROCESS_ACTION.finditer(value)]
        + [(match.start(), match.end(), "process") for match in _STOP_PROCESS.finditer(value)]
    )
    kinds: set[str] = set()
    for start, _end, kind in candidates:
        prefix = value[max(0, start - 60):start]
        clause = re.split(r"[;.!?\n]|\b(?:and|but|then)\b", prefix, flags=re.I)[-1]
        if _NEGATION.search(clause):
            continue
        if _READ_ONLY_OPENING.match(value):
            # "Explain how to fix" is a question; "review and fix" requests work.
            before = value[:start]
            if not _ACTION_LINK.search(before):
                continue
        kinds.add(kind)
    return kinds


def user_wants_action(text: str) -> bool:
    return bool(requested_action_kinds(text))


def is_turn_meta_question(text: str) -> bool:
    value = (text or "").strip()
    return bool(value and len(value) <= 400 and not user_wants_action(value) and _TURN_META.search(value))


def is_internal_control_text(text: str) -> bool:
    """Recognize known legacy control rows without swallowing arbitrary user text."""
    return (text or "").strip().lower().startswith(_INTERNAL_CONTROL_PREFIXES)
