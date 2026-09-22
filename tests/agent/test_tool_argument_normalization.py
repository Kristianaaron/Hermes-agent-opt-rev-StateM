"""Deterministic repair for common model-emitted tool arguments."""

from agent.tool_argument_normalization import normalize_tool_arguments


def test_local_dev_server_is_promoted_to_tracked_background_process():
    normalized, changed = normalize_tool_arguments(
        "terminal", {"command": "npm run dev", "workdir": "/tmp/app"},
    )
    assert changed is True
    assert normalized == {
        "command": "npm run dev",
        "workdir": "/tmp/app",
        "background": True,
    }


def test_shell_background_syntax_is_rewritten_to_native_background():
    normalized, changed = normalize_tool_arguments(
        "terminal", {"command": "nohup npm run dev > /tmp/vite.log 2>&1 &"},
    )
    assert changed is True
    assert normalized["background"] is True
    assert normalized["command"] == "npm run dev > /tmp/vite.log 2>&1"


def test_bounded_terminal_command_is_not_forced_to_background():
    original = {"command": "npm run build"}
    normalized, changed = normalize_tool_arguments("terminal", original)
    assert changed is False
    assert normalized == original


def test_alias_repair_and_server_normalization_compose():
    normalized, changed = normalize_tool_arguments(
        "terminal", {"cmd": "python -m http.server 5173"},
    )
    assert changed is True
    assert normalized == {"command": "python -m http.server 5173", "background": True}


def test_obvious_file_glob_is_not_sent_as_content_regex():
    normalized, changed = normalize_tool_arguments(
        "search_files", {"pattern": "*.html", "order": "modified", "limit": 20},
    )
    assert changed is True
    assert normalized == {
        "pattern": "*.html",
        "target": "files",
        "order": "discovery",
        "limit": 20,
    }


def test_content_regex_and_explicit_filename_search_are_not_rewritten():
    regex_args = {"pattern": "heading|<h1|<h2", "target": "content"}
    explicit_args = {"pattern": "*.tsx", "target": "files", "order": "modified"}
    assert normalize_tool_arguments("search_files", regex_args) == (regex_args, False)
    assert normalize_tool_arguments("search_files", explicit_args) == (explicit_args, False)


def test_regex_mislabelled_as_filename_search_is_repaired():
    for pattern in ("step|Step|heading|subhead", ".*", "^.*$"):
        normalized, changed = normalize_tool_arguments(
            "search_files", {"pattern": pattern, "target": "files", "order": "modified"},
        )
        assert changed is True
        assert normalized == {"pattern": pattern, "target": "content"}
