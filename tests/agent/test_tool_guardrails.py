"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    LoopCapConfig,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
    classify_user_goal_kind,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)


def test_default_config_is_soft_warning_only_with_hard_stop_disabled():
    cfg = ToolCallGuardrailConfig()

    assert cfg.warnings_enabled is True
    assert cfg.hard_stop_enabled is False
    assert cfg.non_interactive_hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 2
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.no_progress_warn_after == 2
    assert cfg.exact_failure_block_after == 5
    assert cfg.same_tool_failure_halt_after == 8
    assert cfg.no_progress_block_after == 5


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_gateway_platform_defaults_to_hard_stop_without_changing_interactive_defaults():
    interactive_configs = [
        ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        for platform in ("cli", "tui", "desktop", "acp")
    ]
    telegram_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="telegram")
    cron_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="cron")

    assert all(cfg.hard_stop_enabled is False for cfg in interactive_configs)
    assert telegram_cfg.hard_stop_enabled is True
    assert cron_cfg.hard_stop_enabled is True


def test_non_interactive_hard_stop_can_be_disabled_explicitly():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {"non_interactive_hard_stop_enabled": False},
        platform="telegram",
    )

    assert cfg.hard_stop_enabled is False
    assert cfg.non_interactive_hard_stop_enabled is False


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2














def test_skill_read_tools_are_idempotent_and_block_repeated_identical_success_output():
    cases = [
        (
            "skill_view",
            {"name": "gui-agent-ml-operations"},
            '{"success":true,"name":"gui-agent-ml-operations","content":"same"}',
        ),
        (
            "skills_list",
            {"category": "mlops"},
            '{"success":true,"skills":[{"name":"gui-agent-ml-operations"}]}',
        ),
    ]

    for tool_name, args, result in cases:
        controller = ToolCallGuardrailController(
            ToolCallGuardrailConfig(
                hard_stop_enabled=True,
                no_progress_warn_after=2,
                no_progress_block_after=2,
            )
        )

        assert controller.before_call(tool_name, args).action == "allow"
        assert controller.after_call(tool_name, args, result, failed=False).action == "allow"
        assert controller.before_call(tool_name, args).action == "allow"
        warn = controller.after_call(tool_name, args, result, failed=False)
        assert warn.action == "warn"
        assert warn.code == "idempotent_no_progress_warning"

        blocked = controller.before_call(tool_name, args)
        assert blocked.action == "block"
        assert blocked.code == "idempotent_no_progress_block"


def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"


def test_identical_call_streak_halts_any_tool_when_hard_stop_enabled():
    # #89069 / #100849 bundle: a model replaying the same SUCCESSFUL
    # terminal/skill_view call with a byte-identical result is not covered by
    # the idempotent_tools no-progress block. The consecutive-identical
    # streak (observe_call) is tool-agnostic; under hard_stop it must halt.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=5)
    )
    args = {"command": "hermes config get memory.provider"}
    for i in range(1, 5):
        controller.after_call("terminal", args, "local\n", failed=False)
        controller.observe_call("terminal", args, "local\n", failed=False)
        assert controller.halt_decision is None, f"halted early at {i}"

    controller.after_call("terminal", args, "local\n", failed=False)
    controller.observe_call("terminal", args, "local\n", failed=False)
    halt = controller.halt_decision
    assert halt is not None and halt.should_halt
    assert halt.code == "identical_call_streak_halt"
    assert halt.tool_name == "terminal" and halt.count == 5


def test_identical_call_streak_never_halts_when_hard_stop_disabled_or_for_pollers():
    soft = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False, no_progress_block_after=2)
    )
    for _ in range(6):
        soft.observe_call("terminal", {"command": "ls"}, "a\nb\n", failed=False)
    assert soft.halt_decision is None  # notice-only in interactive sessions

    hard = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, no_progress_block_after=2)
    )
    for _ in range(6):
        hard.observe_call("process_manage", {"action": "poll", "session_id": "p1"}, "running", failed=False)
    assert hard.halt_decision is None  # an unchanged poll is legitimate progress

    # A changed result resets the streak.
    for i in range(6):
        hard.observe_call("terminal", {"command": "date"}, f"t{i}", failed=False)
    assert hard.halt_decision is None






# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────


def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == 50
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == 50


def test_web_search_cap_blocks_after_limit_regardless_of_hard_stop():
    # Loop caps fire even with hard_stop_enabled=False (the per-turn loop
    # detector's flag). Each distinct query avoids the loop detector so we know
    # the block came from the loop cap, not exact-failure repetition.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt is True


def test_classify_user_goal_kind_distinguishes_ping_status_and_continue():
    assert classify_user_goal_kind("Sorry are you there?") == "ping"
    assert classify_user_goal_kind("are you there") == "ping"
    assert classify_user_goal_kind("Are you stuck?") == "ping"
    assert classify_user_goal_kind("Where are we at so far?") == "status"
    assert classify_user_goal_kind("Please share an update") == "status"
    assert classify_user_goal_kind("sooory continue please") == "work"
    assert classify_user_goal_kind("please keep going on the calib") == "work"
    assert classify_user_goal_kind("Okay go ahead and continue") == "work"


def test_ping_blocks_the_first_tool_call():
    controller = ToolCallGuardrailController()
    controller.observe_user_goal("Sorry are you there?")
    blocked = controller.before_call("execute_code", {"code": "print(1)"})
    assert blocked.action == "block"
    assert blocked.code == "tool_only_reply_required_block"
    assert "I am here" in blocked.message


def test_status_ask_allows_one_tool_only_round_then_blocks():
    controller = ToolCallGuardrailController()
    controller.observe_user_goal("Where are we at so far?")
    assert controller.before_call("execute_code", {"code": "print(1)"}).action == "allow"
    controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    notice = controller.consume_tool_only_brake_notice()
    assert notice and "TOOL LOOP BRAKE" in notice
    blocked = controller.before_call("execute_code", {"code": "print(2)"})
    assert blocked.action == "block"
    assert blocked.code == "tool_only_reply_required_block"


def test_work_turn_nudges_after_streak_but_does_not_halt():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(max_tool_only_iterations=3, commentary_gate_after=0)
    )
    controller.observe_user_goal("continue the HY4 calib")
    for i in range(3):
        assert controller.before_call("execute_code", {"code": str(i)}).action == "allow"
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    notice = controller.consume_tool_only_brake_notice()
    assert notice and "TOOL LOOP NUDGE" in notice
    assert controller.before_call("execute_code", {"code": "4"}).action == "allow"


def test_work_reasoning_still_counts_as_mute_tool_loop():
    # Codex: reasoning items are not a user-visible assistant message.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            max_tool_only_iterations=2, work_tool_only_halt_after=0, commentary_gate_after=0
        )
    )
    controller.observe_user_goal("keep going on A1")
    for i in range(2):
        assert controller.before_call("execute_code", {"code": str(i)}).action == "allow"
        controller.observe_assistant_round(
            has_tool_calls=True, has_visible_text=False, has_reasoning=True
        )
    assert controller.tool_only_streak == 2
    notice = controller.consume_tool_only_brake_notice()
    assert notice and "TOOL LOOP NUDGE" in notice


def test_work_turn_halts_after_ignored_nudges():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            max_tool_only_iterations=2, work_tool_only_halt_after=4, commentary_gate_after=0
        )
    )
    controller.observe_user_goal("continue the HY4 calib")
    for i in range(4):
        assert controller.before_call("execute_code", {"code": str(i)}).action == "allow"
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    notice = controller.consume_tool_only_brake_notice()
    assert notice and "TOOL LOOP BRAKE" in notice
    blocked = controller.before_call("execute_code", {"code": "halt"})
    assert blocked.action == "block"
    assert blocked.code == "tool_only_reply_required_block"


def test_visible_text_with_tools_resets_work_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(max_tool_only_iterations=2)
    )
    controller.observe_user_goal("keep going")
    controller.before_call("execute_code", {"code": "1"})
    controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    controller.observe_assistant_round(has_tool_calls=True, has_visible_text=True)
    assert controller.tool_only_streak == 0
    assert controller.before_call("execute_code", {"code": "2"}).action == "allow"


def test_config_parses_tool_only_brake_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "max_tool_only_iterations": 6,
            "work_tool_only_halt_after": 12,
            "status_ask_max_tool_only": 2,
            "ping_max_tool_only": 0,
        }
    )
    assert cfg.max_tool_only_iterations == 6
    assert cfg.work_tool_only_halt_after == 12
    assert cfg.status_ask_max_tool_only == 2
    assert cfg.ping_max_tool_only == 0


def test_config_parses_codex_stale_anchor_and_housekeeping_caps():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "housekeeping_tool_only_halt_after": 3,
            "stale_anchor_block_after": 1,
            "work_tool_only_halt_after": 24,
            "commentary_gate_after": 2,
        }
    )
    assert cfg.housekeeping_tool_only_halt_after == 3
    assert cfg.stale_anchor_block_after == 1
    assert cfg.work_tool_only_halt_after == 24
    assert cfg.commentary_gate_after == 2


def test_config_parses_commentary_gate_empty_cap():
    cfg = ToolCallGuardrailConfig.from_mapping({"commentary_gate_empty_cap": 3})
    assert cfg.commentary_gate_empty_cap == 3


def test_commentary_gate_arms_after_mute_rounds_and_continues_turn():
    """Codex mid-turn rule: mute tools → tools-off commentary → continue, not halt."""
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            max_tool_only_iterations=8,
            work_tool_only_halt_after=24,
            commentary_gate_after=2,
        )
    )
    controller.observe_user_goal("keep going on A1")
    for i in range(2):
        assert controller.before_call("execute_code", {"code": str(i)}).action == "allow"
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    notice = controller.consume_tool_only_brake_notice()
    assert notice and "COMMENTARY GATE" in notice
    assert controller.begin_commentary_request() is True
    assert controller.commentary_gate_active() is True
    # Tools stay allowed in before_call — the request omits them; halt path is separate.
    assert controller.before_call("execute_code", {"code": "should-not-run"}).action == "allow"
    assert controller.finish_commentary_request(has_visible_text=True) is True
    assert controller.tool_only_streak == 0
    assert controller.commentary_gate_active() is False
    assert controller.begin_commentary_request() is False


def test_commentary_gate_empty_response_rearms_tools_off():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(commentary_gate_after=2, work_tool_only_halt_after=24)
    )
    controller.observe_user_goal("continue")
    for i in range(2):
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    assert controller.begin_commentary_request() is True
    assert controller.finish_commentary_request(has_visible_text=False) is True
    assert controller.tool_only_streak == 2
    assert controller.begin_commentary_request() is True


def test_commentary_gate_empty_response_cap_restores_tools():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            commentary_gate_after=2,
            commentary_gate_empty_cap=2,
            work_tool_only_halt_after=24,
        )
    )
    controller.observe_user_goal("continue")
    for i in range(2):
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    assert controller.begin_commentary_request() is True
    assert controller.finish_commentary_request(has_visible_text=False) is True
    assert controller.begin_commentary_request() is True
    assert controller.finish_commentary_request(has_visible_text=False) is False
    assert controller.commentary_gate_active() is False
    assert controller.begin_commentary_request() is False


def _armed_commentary_controller():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(commentary_gate_after=2, work_tool_only_halt_after=24)
    )
    controller.observe_user_goal("continue")
    for _ in range(2):
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    assert controller.begin_commentary_request() is True
    return controller


def test_commentary_gate_forces_empty_tools_even_when_caller_passes_none():
    from types import SimpleNamespace

    from agent.chat_completion_helpers import (
        _apply_commentary_gate_to_tools,
        _strip_tools_from_kwargs_for_commentary_gate,
    )

    controller = _armed_commentary_controller()
    tools = [{"type": "function", "function": {"name": "execute_code"}}]
    agent = SimpleNamespace(_tool_guardrails=controller, tools=tools)
    assert _apply_commentary_gate_to_tools(agent, None) == []
    assert _apply_commentary_gate_to_tools(agent, tools) == []
    kwargs = {"model": "x", "tools": tools, "functions": tools, "extra_body": {"tools": tools}}
    _strip_tools_from_kwargs_for_commentary_gate(agent, kwargs)
    assert "tools" not in kwargs
    assert "functions" not in kwargs
    assert "tools" not in kwargs["extra_body"]


def test_commentary_gate_inactive_keeps_default_tools():
    from types import SimpleNamespace

    from agent.chat_completion_helpers import _apply_commentary_gate_to_tools

    tools = [{"type": "function", "function": {"name": "execute_code"}}]
    agent = SimpleNamespace(_tool_guardrails=ToolCallGuardrailController(), tools=tools)
    assert _apply_commentary_gate_to_tools(agent, None) is tools


def test_wire_boundary_omits_empty_tool_collections():
    from agent.chat_completion_helpers import _omit_empty_tools_at_wire_boundary

    kwargs = {
        "model": "GLM-5.3-Flash-EXL3",
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [],
        "functions": [],
        "extra_body": {"tools": [], "functions": [], "keep": True},
    }

    result = _omit_empty_tools_at_wire_boundary(kwargs)

    assert "tools" not in result
    assert "functions" not in result
    assert result["extra_body"] == {"keep": True}


def test_wire_boundary_preserves_nonempty_tools():
    from agent.chat_completion_helpers import _omit_empty_tools_at_wire_boundary

    tools = [{"type": "function", "function": {"name": "terminal"}}]
    kwargs = {"tools": tools, "extra_body": {"tools": tools}}

    result = _omit_empty_tools_at_wire_boundary(kwargs)

    assert result["tools"] is tools
    assert result["extra_body"]["tools"] is tools


def test_redecorate_does_not_restore_tools_when_commentary_gate_active():
    from types import SimpleNamespace

    from agent.conversation_loop import _redecorate_prompt_cache_for_provider

    controller = _armed_commentary_controller()
    tools = [{"type": "function", "function": {"name": "execute_code"}}]
    agent = SimpleNamespace(
        _tool_guardrails=controller,
        tools=tools,
        _use_prompt_caching=False,
        provider="openai",
        client=None,
    )
    messages = [{"role": "user", "content": "hi"}]
    _out, _prepared, planned = _redecorate_prompt_cache_for_provider(
        agent, messages, tools_for_api=[]
    )
    assert planned == []
    _out2, _prepared2, planned_from_none = _redecorate_prompt_cache_for_provider(
        agent, messages, tools_for_api=None
    )
    assert planned_from_none == []


def test_phase_after_commentary_gate_drops_tool_calls_without_text():
    from types import SimpleNamespace

    from agent.conversation_loop import _phase_after_commentary_gate, finish_text_response

    controller = _armed_commentary_controller()
    agent = SimpleNamespace(
        _tool_guardrails=controller,
        session_id="s",
        _strip_think_blocks=lambda t: t,
    )
    msg = SimpleNamespace(content="", tool_calls=[{"id": "1"}])
    assert _phase_after_commentary_gate(agent, msg) is finish_text_response
    assert msg.tool_calls is None
    assert controller.commentary_gate_active() is True


def test_phase_after_commentary_gate_preamble_then_tools():
    from types import SimpleNamespace

    from agent.conversation_loop import _phase_after_commentary_gate, run_tool_round

    controller = _armed_commentary_controller()
    agent = SimpleNamespace(
        _tool_guardrails=controller,
        session_id="s",
        _strip_think_blocks=lambda t: t,
    )
    msg = SimpleNamespace(content="Still working on A1.", tool_calls=[{"id": "1"}])
    assert _phase_after_commentary_gate(agent, msg) is run_tool_round
    assert msg.tool_calls == [{"id": "1"}]
    assert controller.commentary_gate_active() is False
    assert controller.tool_only_streak == 0


def test_commentary_gate_disabled_when_zero():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(commentary_gate_after=0, max_tool_only_iterations=2)
    )
    controller.observe_user_goal("continue")
    for i in range(2):
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    assert controller.begin_commentary_request() is False
    notice = controller.consume_tool_only_brake_notice()
    assert notice and "TOOL LOOP NUDGE" in notice


def test_mute_halt_still_outranks_commentary_gate():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            commentary_gate_after=2,
            max_tool_only_iterations=2,
            work_tool_only_halt_after=4,
        )
    )
    controller.observe_user_goal("continue")
    for i in range(4):
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    assert controller.begin_commentary_request() is False
    notice = controller.consume_tool_only_brake_notice()
    assert notice and "TOOL LOOP BRAKE" in notice
    blocked = controller.before_call("execute_code", {"code": "halt"})
    assert blocked.code == "tool_only_reply_required_block"


def test_tool_only_halt_copy_is_not_a_retry_accusation():
    from agent.tool_guardrails import controlled_halt_response

    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(max_tool_only_iterations=1, work_tool_only_halt_after=2)
    )
    controller.observe_user_goal("please continue")
    controller.before_call("execute_code", {"code": "1"})
    controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    controller.before_call("execute_code", {"code": "2"})
    controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    blocked = controller.before_call("memory", {"action": "replace", "old_text": "stale"})
    assert blocked.code == "tool_only_reply_required_block"
    text = controlled_halt_response(blocked)
    assert "retrying memory" not in text.lower()
    assert "Codex" in text
    assert "assistant message" in text.lower()


def test_housekeeping_only_mute_halts_before_productive_work_cap():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            max_tool_only_iterations=8,
            work_tool_only_halt_after=24,
            housekeeping_tool_only_halt_after=2,
        )
    )
    controller.observe_user_goal("keep going on A1")
    for i in range(6):
        assert controller.before_call("terminal", {"command": f"ls {i}"}).action == "allow"
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    assert controller.tool_only_streak == 6
    assert controller.before_call("memory", {"action": "add", "content": "x"}).action == "allow"
    controller.observe_assistant_round(
        has_tool_calls=True, has_visible_text=False, housekeeping_only=True
    )
    assert controller.before_call("todo_list", {"todos": []}).action == "allow"
    controller.observe_assistant_round(
        has_tool_calls=True, has_visible_text=False, housekeeping_only=True
    )
    blocked = controller.before_call("memory", {"action": "add", "content": "y"})
    assert blocked.action == "block"
    assert blocked.code == "tool_only_reply_required_block"
    assert "housekeeping" in blocked.message.lower()
    assert "retrying memory" not in blocked.message.lower()


def test_productive_mute_tools_are_not_cut_by_housekeeping_cap():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            max_tool_only_iterations=8,
            work_tool_only_halt_after=24,
            housekeeping_tool_only_halt_after=2,
        )
    )
    controller.observe_user_goal("continue")
    for i in range(8):
        assert controller.before_call("write_file", {"path": f"a{i}", "content": "x"}).action == "allow"
        controller.observe_assistant_round(has_tool_calls=True, has_visible_text=False)
    assert controller.tool_only_streak == 8
    assert controller.housekeeping_only_streak == 0
    assert controller.before_call("terminal", {"command": "true"}).action == "allow"


def test_stale_memory_replace_blocks_identical_retry_without_hard_stop():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False, stale_anchor_block_after=1)
    )
    args = {"action": "replace", "target": "memory", "old_text": "gone", "content": "new"}
    result = json.dumps({
        "success": False,
        "error": "No entry matched 'gone'. Check current_entries below and retry with the exact text.",
        "current_entries": ["live entry"],
    })
    warned = controller.after_call("memory", args, result, failed=True)
    assert warned.action == "warn"
    assert warned.code == "stale_anchor_retry_warning"
    assert "Do not retry the same old_text" in warned.message
    blocked = controller.before_call("memory", args)
    assert blocked.action == "block"
    assert blocked.code == "stale_anchor_retry_block"
    from agent.tool_guardrails import controlled_halt_response
    text = controlled_halt_response(blocked)
    assert "retrying memory" not in text.lower()
    assert "anchor" in text.lower()


def test_stale_memory_allows_a_different_old_text():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False, stale_anchor_block_after=1)
    )
    stale = {"action": "replace", "old_text": "gone", "content": "new"}
    fresh = {"action": "replace", "old_text": "live entry", "content": "updated"}
    controller.after_call(
        "memory",
        stale,
        json.dumps({"success": False, "error": "No entry matched 'gone'."}),
        failed=True,
    )
    assert controller.before_call("memory", stale).action == "block"
    assert controller.before_call("memory", fresh).action == "allow"


def test_stale_patch_old_string_blocks_identical_retry():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=False, stale_anchor_block_after=1)
    )
    args = {"path": "x.py", "old_string": "missing", "new_string": "here"}
    controller.after_call(
        "patch", args, json.dumps({"error": "old_string not found. Use read_file."}), failed=True,
    )
    blocked = controller.before_call("patch", args)
    assert blocked.action == "block"
    assert blocked.code == "stale_anchor_retry_block"


def test_classify_memory_no_match_is_a_failure():
    failed, tag = classify_tool_failure(
        "memory",
        json.dumps({"success": False, "error": "No entry matched 'stale'."}),
    )
    assert failed is True
    assert "stale-anchor" in tag


def test_live_config_keeps_codex_mute_and_stale_anchor_policy():
    from pathlib import Path

    text = Path("/Users/kristian/.hermes/config.yaml").read_text(encoding="utf-8")
    assert "work_tool_only_halt_after: 24" in text
    assert "housekeeping_tool_only_halt_after: 2" in text
    assert "stale_anchor_block_after: 1" in text













# ── Legitimate flows must survive hard stops (Teknium, Sep 2026) ────────────
# Hard stops default ON for unattended platforms. These pin the flows that
# must NEVER be cut off there: edit -> re-run loops, diagnostic sweeps of
# distinct red commands, and browser retry-after-action — while the pure
# replay (same call, nothing changed between attempts) is still stopped.

_HARD = lambda: ToolCallGuardrailController(  # noqa: E731
    ToolCallGuardrailConfig(hard_stop_enabled=True)
)
_PYTEST = {"command": "pytest tests/test_x.py -q"}
_RED = '{"output": "1 failed", "exit_code": 1}'


def _run_red(c, args=_PYTEST):
    assert c.before_call("terminal", args).allows_execution
    return c.after_call("terminal", args, _RED, failed=True)


def test_fix_retest_loop_is_never_hard_stopped():
    c = _HARD()
    for i in range(12):
        d = _run_red(c)
        assert not d.should_halt, f"halted on red run {i + 1}"
        # the model edits between runs — a landed mutation is progress
        c.after_call("patch", {"path": "x.py", "old_string": "a", "new_string": f"b{i}"},
                     '{"success": true, "diff": "..."}', failed=False)
    assert c.halt_decision is None
    assert c.before_call("terminal", _PYTEST).allows_execution


def test_pure_replay_with_no_intervening_change_is_still_blocked():
    c = _HARD()
    for _ in range(5):
        _run_red(c)
    d = c.before_call("terminal", _PYTEST)
    assert d.action == "block" and d.code == "repeated_exact_failure_block"


def test_intervening_mutation_resets_the_replay_streak_only_once():
    # 4 reds, one edit, then 4 reds with NO edit: the second run of 4 is a
    # fresh streak, and the 5th unchanged retry after it is blocked.
    c = _HARD()
    for _ in range(4):
        _run_red(c)
    c.after_call("write_file", {"path": "x.py", "content": "y"}, '{"bytes_written": 1}', failed=False)
    for _ in range(5):
        assert c.before_call("terminal", _PYTEST).allows_execution
        c.after_call("terminal", _PYTEST, _RED, failed=True)
    assert c.before_call("terminal", _PYTEST).action == "block"


def test_distinct_failing_terminal_commands_warn_but_never_halt():
    # A diagnostic sweep: grep with no matches, missing binaries, red builds.
    c = _HARD()
    for i in range(12):
        args = {"command": f"grep -q needle{i} haystack.txt"}
        d = c.after_call("terminal", args, _RED, failed=True)
        assert not d.should_halt, f"same_tool halt on distinct command #{i + 1}"
    assert c.halt_decision is None
    # ...while a non-tolerant tool failing 8 distinct ways still halts.
    c2 = _HARD()
    last = None
    for i in range(8):
        last = c2.after_call("send_message", {"to": f"u{i}"}, '{"error": "no route"}', failed=True)
    assert last.should_halt and last.code == "same_tool_failure_halt"


def test_browser_retry_after_action_is_not_a_replay():
    c = _HARD()
    nav = {"url": "https://example.test/app"}
    for _ in range(8):
        assert c.before_call("browser_navigate", nav).allows_execution
        c.after_call("browser_navigate", nav, '{"error": "timeout"}', failed=True)
        c.after_call("browser_click", {"selector": "#retry"}, '{"ok": true}', failed=False)
    assert c.halt_decision is None


def test_supervised_task_platforms_keep_warning_only_default():
    for platform in ("subagent", "api_server", "cli"):
        cfg = ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        assert cfg.hard_stop_enabled is False, platform
    for platform in ("telegram", "discord", "cron", "kanban"):
        cfg = ToolCallGuardrailConfig.from_mapping({}, platform=platform)
        assert cfg.hard_stop_enabled is True, platform


def test_harness_refusals_are_not_tool_failures_on_either_classifier(tmp_path):
    """Every loop refusal the file tools emit (read dedup block, consecutive-read block,
    repeated-search block) carries `"error"` for the model's benefit -- exactly what the
    substring tests key on. Neither `classify_tool_failure` nor the executor's live seam
    `_detect_tool_failure` may count them, or the cheap refusal feeds the streak that
    fires `repeated_exact_failure_block` over calls that never failed."""
    from agent.display import _detect_tool_failure
    from tools.file_tools import _dedup_stub_or_block, read_file_tool, search_tool

    target = tmp_path / "responses.ts"
    target.write_text("export const x = 1;\n" * 30, encoding="utf-8")

    task = {"dedup_hits": {}}
    for _ in range(3):
        dedup_block = _dedup_stub_or_block(task, (str(target), 1, 999), str(target))
    consecutive_block = [read_file_tool(str(target), offset=1, limit=5, task_id="t-read") for _ in range(4)][-1]
    search_block = [search_tool("const x", path=str(tmp_path), task_id="t-search") for _ in range(4)][-1]

    for refusal in (dedup_block, consecutive_block, search_block):
        assert json.loads(refusal)["error"].startswith("BLOCKED"), refusal
        assert classify_tool_failure("read_file", refusal) == (False, ""), refusal
        assert _detect_tool_failure("read_file", refusal) == (False, ""), refusal


def test_a_real_tool_error_is_still_a_failure():
    """The exemption is keyed on the marker, not on the word: a body that
    genuinely failed still counts, or the streak that stops a real loop is gone."""
    from agent.display import _detect_tool_failure

    real = '{"error": "ENOENT: no such file"}'
    assert classify_tool_failure("read_file", real)[0] is True
    assert _detect_tool_failure("read_file", real)[0] is True
    # The marker is only honoured as the literal boolean, never as truthy prose.
    assert classify_tool_failure("read_file", '{"error": "x", "guardrail_refusal": "yes"}')[0] is True


def test_read_only_halt_can_be_released_once_for_an_edit():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True,
        max_consecutive_read_only=2,
    ))
    for index in range(2):
        args = {"path": f"/tmp/{index}"}
        assert controller.before_call("read_file", args).allows_execution
        controller.after_call("read_file", args, "contents", failed=False)
    halted = controller.before_call("read_file", {"path": "/tmp/again"})
    assert halted.should_halt
    assert controller.release_read_only_halt() is True
    assert controller.halt_decision is None
    assert controller.release_read_only_halt() is False
    halted_again = controller.before_call("read_file", {"path": "/tmp/third"})
    assert halted_again.should_halt


def test_read_only_streak_stops_varied_browser_inspection_loop():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True,
        max_consecutive_read_only=3,
        max_total_tool_calls=20,
    ))
    for index in range(3):
        args = {"selector": f"#panel-{index}"}
        assert controller.before_call("browser_snapshot", args).allows_execution
        controller.after_call("browser_snapshot", args, f'{{"view": {index}}}', failed=False)

    halted = controller.before_call("browser_snapshot", {"selector": "#another"})
    assert halted.should_halt
    assert halted.code == "read_only_streak_halt"


def test_successful_concrete_action_resets_read_only_streak():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True,
        max_consecutive_read_only=2,
    ))
    for index in range(2):
        args = {"path": f"/tmp/{index}"}
        assert controller.before_call("read_file", args).allows_execution
        controller.after_call("read_file", args, "contents", failed=False)
    click = {"selector": "#continue"}
    assert controller.before_call("browser_click", click).allows_execution
    controller.after_call("browser_click", click, '{"ok": true}', failed=False)
    assert controller.before_call("browser_snapshot", {"selector": "body"}).allows_execution


def test_semantically_equivalent_process_failures_halt_even_when_ids_change():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True,
        semantic_failure_halt_after=3,
    ))
    decision = None
    for process_id in ("missing-a", "missing-b", "missing-c"):
        args = {"action": "poll", "session_id": process_id}
        assert controller.before_call("process_manage", args).allows_execution
        decision = controller.after_call(
            "process_manage", args, '{"error": "process not found"}', failed=True,
        )
    assert decision is not None and decision.should_halt
    assert decision.code == "semantic_failure_halt"


def test_total_tool_call_cap_bounds_varied_action_loops():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=True,
        max_total_tool_calls=3,
        max_consecutive_read_only=20,
        max_unproductive_tool_calls=20,
    ))
    for index in range(3):
        args = {"command": f"printf {index}"}
        assert controller.before_call("terminal", args).allows_execution
        controller.after_call("terminal", args, str(index), failed=False)
    halted = controller.before_call("terminal", {"command": "printf done"})
    assert halted.should_halt
    assert halted.code == "total_tool_call_cap"
