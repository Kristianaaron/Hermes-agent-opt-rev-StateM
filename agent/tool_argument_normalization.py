"""Deterministic normalization for model-emitted tool arguments."""

from __future__ import annotations

import json
import re
from typing import Any


_SAFE_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "terminal": {
        "command": ("cmd", "shell_command"),
        "workdir": ("cwd", "working_directory"),
    },
    "read_file": {"path": ("file_path", "filename")},
    "write_file": {
        "path": ("file_path", "filename"),
        "content": ("text", "data"),
    },
    "patch": {"path": ("file_path", "filename")},
}


_SHELL_BACKGROUND_SUFFIX = re.compile(r"(?<!&)\s*(?:;\s*)?&\s*$")
_SHELL_DISOWN_SUFFIX = re.compile(r"\s*(?:;|&&)\s*disown\s*$", re.I)
_NOHUP_PREFIX = re.compile(r"^\s*nohup\s+", re.I)
_LONG_LIVED_SERVER_COMMAND = re.compile(
    r"(?:^|(?:&&|;|\|\|)\s*)"
    r"(?:"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|preview)\b"
    r"|npx\s+(?:vite|next\s+dev|astro\s+dev|serve)\b"
    r"|(?:vite|next\s+dev|astro\s+dev|ng\s+serve)\b"
    r"|python(?:3)?\s+-m\s+http\.server\b"
    r"|(?:uvicorn|gunicorn)\b"
    r"|flask\s+run\b"
    r"|(?:rails|bin/rails)\s+(?:server|s)\b"
    r"|(?:python(?:3)?\s+manage\.py\s+)?runserver\b"
    r")",
    re.I,
)
_OBVIOUS_FILE_GLOB = re.compile(r"^(?:\*|\?)[^\n]*$")


def is_long_lived_terminal_command(command: Any) -> bool:
    """Return whether *command* is a known foreground-unbounded server."""
    return isinstance(command, str) and bool(_LONG_LIVED_SERVER_COMMAND.search(command))


def _normalize_terminal_execution(arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Route detached/long-lived shell work through Hermes' process registry.

    Local models often emit ``npm run dev &`` or retry ``npm run dev`` in the
    foreground after it times out. Hermes already has a first-class
    ``background`` argument; normalize those equivalent spellings before the
    guardrail and approval layers see them. This keeps the launch observable
    and gives the model a process id it can poll instead of spawning orphans.
    """
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return arguments, False

    normalized_command = command
    shell_background = bool(
        _SHELL_BACKGROUND_SUFFIX.search(normalized_command)
        or _SHELL_DISOWN_SUFFIX.search(normalized_command)
        or _NOHUP_PREFIX.search(normalized_command)
    )
    long_lived = is_long_lived_terminal_command(normalized_command)
    if not shell_background and not long_lived:
        return arguments, False

    normalized_command = _SHELL_DISOWN_SUFFIX.sub("", normalized_command)
    normalized_command = _SHELL_BACKGROUND_SUFFIX.sub("", normalized_command)
    normalized_command = _NOHUP_PREFIX.sub("", normalized_command).strip()
    normalized = dict(arguments)
    normalized["command"] = normalized_command
    normalized["background"] = True
    return normalized, normalized != arguments


def _normalize_search_files(arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Repair the common ``pattern='*.ext'`` filename-search ambiguity.

    ``search_files`` defaults to content regex mode, where a leading ``*`` is
    invalid. Models regularly omit ``target='files'`` despite clearly emitting
    a glob. Treat only an unmistakable leading-wildcard pattern as filename
    intent. Reset exact modified ordering on this repaired call: it was not an
    explicit filename-mode request and is unsupported by BSD ``find``.
    """
    pattern = arguments.get("pattern")
    target = arguments.get("target")
    if (
        isinstance(pattern, str)
        and _OBVIOUS_FILE_GLOB.match(pattern.strip())
        and target in {None, "content"}
        and "file_glob" not in arguments
    ):
        normalized = dict(arguments)
        normalized["target"] = "files"
        if normalized.get("order") == "modified":
            normalized["order"] = "discovery"
        return normalized, normalized != arguments
    # The inverse confusion is also common: regex alternation or the universal
    # regex gets sent to filename-glob mode, returns zero, and prompts an
    # expensive search outside the workspace. These shapes are unambiguously
    # content regexes rather than useful filename globs.
    if (
        isinstance(pattern, str)
        and target == "files"
        and ("|" in pattern or pattern.strip() in {".*", "^.*$"})
    ):
        normalized = dict(arguments)
        normalized["target"] = "content"
        normalized.pop("order", None)
        return normalized, True
    return arguments, False


def normalize_tool_arguments(
    tool_name: str | None,
    value: Any,
    *,
    max_depth: int = 2,
) -> tuple[Any, bool]:
    """Normalize conservative, unambiguous model argument variations.

    Some OpenAI-compatible models occasionally emit the function-call transport
    field as part of the function payload itself.  Only a single-key wrapper
    whose nested value decodes to a JSON object is repaired, so legitimate tool
    arguments and non-object payloads are left unchanged.  A small per-tool
    alias table then canonicalizes common names only when the canonical field is
    absent and exactly one alias is present.
    """

    current = value
    changed = False
    for _ in range(max_depth):
        if not isinstance(current, dict) or set(current) != {"arguments"}:
            break
        nested = current["arguments"]
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except (json.JSONDecodeError, TypeError):
                break
        if not isinstance(nested, dict):
            break
        current = nested
        changed = True

    if not isinstance(current, dict):
        return current, changed

    normalized_name = str(tool_name or "")
    if normalized_name == "search_files":
        current, search_changed = _normalize_search_files(current)
        changed = changed or search_changed
    aliases = _SAFE_ALIASES.get(normalized_name, {})
    if not aliases:
        if normalized_name == "terminal":
            current, terminal_changed = _normalize_terminal_execution(current)
            changed = changed or terminal_changed
        return current, changed

    normalized = dict(current)
    for canonical, candidates in aliases.items():
        if canonical in normalized:
            continue
        present = [candidate for candidate in candidates if candidate in normalized]
        if len(present) != 1:
            continue
        normalized[canonical] = normalized.pop(present[0])
        changed = True
    if normalized_name == "terminal":
        normalized, terminal_changed = _normalize_terminal_execution(normalized)
        changed = changed or terminal_changed
    return normalized, changed


def unwrap_nested_tool_arguments(value: Any, *, max_depth: int = 2) -> tuple[Any, bool]:
    """Backward-compatible wrapper for callers that do not know the tool name."""

    return normalize_tool_arguments(None, value, max_depth=max_depth)
