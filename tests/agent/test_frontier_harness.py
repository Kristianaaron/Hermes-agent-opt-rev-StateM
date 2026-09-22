"""Frontier harness loop-signal tests."""

import json

from agent.frontier_harness import _analyze, _failed_result, _persist


def test_successful_terminal_dump_is_not_a_failure():
    payload = json.dumps(
        {
            "output": "void had_error_handler()\n{\n    // error path\n}\n",
            "exit_code": 0,
            "error": None,
        }
    )
    assert _failed_result(payload) is False


def test_nonzero_exit_is_a_failure():
    payload = json.dumps({"output": "boom", "exit_code": 1, "error": None})
    assert _failed_result(payload) is True


def test_plain_success_output_can_describe_failures_without_poisoning_state():
    assert _failed_result(
        "Skill instructions: detect errors, failures, exceptions, and timeout behavior."
    ) is False


def test_structured_success_beats_nonfatal_error_text():
    assert _failed_result(json.dumps({"ok": True, "error": "non-fatal warning"})) is False


def test_explicit_plain_failure_envelope_is_still_a_failure():
    assert _failed_result("ERROR: permission denied") is True


def test_analyze_does_not_mark_path_drift_on_successful_source_dumps():
    messages = [
        {"role": "user", "content": "Okay go ahead and continue"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "grep error kernel.cuh"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "terminal",
            "content": json.dumps(
                {
                    "output": "exl3_moe_kernel.cuh: error handler for warp",
                    "exit_code": 0,
                    "error": None,
                }
            ),
        },
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["failure_count"] == 0
    assert state["path_health"] != "path_drift"


def test_new_real_user_turn_resets_old_failure_and_repetition_evidence():
    failed_call = {
        "role": "assistant",
        "tool_calls": [{
            "id": "old-call",
            "function": {"name": "terminal", "arguments": json.dumps({"command": "false"})},
        }],
    }
    failed_result = {
        "role": "tool",
        "tool_call_id": "old-call",
        "name": "terminal",
        "content": json.dumps({"output": "failed", "exit_code": 1, "error": None}),
    }
    messages = [
        {"role": "user", "content": "Old task"},
        failed_call,
        failed_result,
        failed_call,
        failed_result,
        {"role": "assistant", "content": "The old task failed."},
        {"role": "user", "content": "Hello"},
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["failure_count"] == 0
    assert state["max_repeated_signature"] == 0
    assert state["phase"] == "localize"


def test_retry_followup_inherits_goal_without_inheriting_old_failures():
    messages = [
        {"role": "user", "content": "Add the two data options to the card"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "old-call",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "false"})},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "old-call",
            "name": "terminal",
            "content": json.dumps({"output": "failed", "exit_code": 1, "error": None}),
        },
        {"role": "assistant", "content": "The attempt stopped."},
        {"role": "user", "content": "please rety and continue"},
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["goal"] == "Add the two data options to the card"
    assert state["failure_count"] == 0


def test_synthetic_user_nudge_does_not_reset_current_turn_failure_evidence():
    messages = [
        {"role": "user", "content": "Start the server"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "npm run dev"})},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "terminal",
            "content": json.dumps({"output": "failed", "exit_code": 1, "error": None}),
        },
        {"role": "user", "content": "continue", "_pre_verify_synthetic": True},
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["failure_count"] == 1


def test_empty_recovery_nudge_does_not_reset_current_turn_failure_evidence():
    messages = [
        {"role": "user", "content": "Start the server"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "false"})},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "terminal",
            "content": json.dumps({"output": "failed", "exit_code": 1, "error": None}),
        },
        {"role": "user", "content": "Continue processing", "_empty_recovery_synthetic": True},
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["failure_count"] == 1


def test_persisted_auto_continue_row_does_not_reset_current_turn_evidence():
    messages = [
        {"role": "user", "content": "Add the data options"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "false"})},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "terminal",
            "content": json.dumps({"output": "failed", "exit_code": 1, "error": None}),
        },
        {
            "role": "user",
            "content": "[System note: Your previous turn was interrupted mid-run — resume]",
            "display_kind": "auto_continue",
        },
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["failure_count"] == 1
    assert state["goal"] == "Add the data options"


def test_persist_keeps_stop_next_action(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = {
        "phase": "repair",
        "path_health": "path_drift",
        "next_action": "stop tools; inspect the last failure, change strategy, then verify",
        "trailing_failure_streak": 2,
        "goal": "continue",
        "verification_required": True,
        "progress_digest": "abc",
    }
    _persist("session-x", "glm-test", state)
    saved = json.loads(next((tmp_path / "frontier_runs").glob("*.json")).read_text())
    assert saved["resume_contract"]["next_action"].startswith("stop tools")


def test_frontier_protocol_includes_codex_turn_and_stale_anchor_rules():
    from agent.frontier_harness import _PROTOCOL

    assert "Codex turn rule" in _PROTOCOL
    assert "old_text/old_string" in _PROTOCOL
    assert "user-visible assistant message" in _PROTOCOL
    assert "asks for an explanation" in _PROTOCOL
    assert "file the user did not ask for" in _PROTOCOL


def test_harness_control_row_does_not_replace_the_goal():
    messages = [
        {"role": "user", "content": "Decrease the step header weights from 700 to 500"},
        {"role": "assistant", "content": "patched one heading"},
        {
            "role": "user",
            "content": (
                "[System: You edited code in this turn, but the workspace does not "
                "have fresh passing verification evidence yet. Create a tempfile "
                "named hermes-verify- and run it."
            ),
        },
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["goal"] == "Decrease the step header weights from 700 to 500"


def test_quoting_previous_reply_is_a_question_not_a_repeated_goal():
    reply = "I opened the workspace, listed the components, and stopped before editing anything."
    messages = [
        {"role": "user", "content": "Rename the export button to Save"},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "what did you do?\n\n" + reply},
    ]
    state = _analyze(messages, {"path_deviation_failure_streak": 2, "max_unproductive_actions": 4})
    assert state["goal"].startswith("what did you do?")
