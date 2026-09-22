#!/usr/bin/env python3
"""Offline checks for semantic System-1 routing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parent / "__init__.py"
_SPEC = importlib.util.spec_from_file_location("test_glm_codex_effort_plugin", _PLUGIN)
router = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(router)


def msgs(text: str, tools: bool = False, failed: bool = False):
    out = [{"role": "system", "content": "very large system prompt"}, {"role": "user", "content": text}]
    if tools:
        out.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
        })
        out.append({
            "role": "tool",
            "tool_call_id": "1",
            "content": '{"ok": false, "error": "command failed"}' if failed else '{"ok": true}',
        })
    return out


def tool(name):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {}}}


def request(text: str, *, high: bool = False, tools: bool = False, failed: bool = False):
    effort = "high" if high else "max"
    return {
        "model": "GLM-5.3-Flash-EXL3",
        "messages": msgs(text, tools=tools, failed=failed),
        "tools": [
            tool("terminal"), tool("process"), tool("read_file"), tool("skill_view"),
            tool("web_search"), tool("tool_search"), tool("tool_call"), tool("patch"),
            tool("write_file"),
        ],
        "max_tokens": 32768,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort}},
    }


def evidence(decision="fast", category="bounded_operation", confidence=.9, mutation_intent=None):
    if mutation_intent is None:
        mutation_intent = category == "bounded_operation"
    return {
        "decision": decision, "category": category, "confidence": confidence,
        "mutation_intent": mutation_intent,
    }


def test_fast_lane_compacts_context_tools_and_disables_thinking(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Please initiate the development environment")
    out = router.on_llm_request(request=req, model=req["model"], api_call_count=1, turn_id="t-fast")
    routed = out["request"]
    ctk = routed["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is False
    assert "reasoning_effort" not in ctk
    assert len(routed["messages"]) == 2
    assert {_tool["function"]["name"] for _tool in routed["tools"]} == {
        "terminal", "process", "read_file", "web_search", "patch", "write_file"
    }
    assert routed["max_tokens"] == 4096


def test_fast_tool_hop_keeps_tool_protocol(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Initiate the environment", tools=True)
    out = router.on_llm_request(request=req, model=req["model"], api_call_count=2, turn_id="t-hop")
    roles = [message["role"] for message in out["request"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]


def test_first_failed_bounded_tool_stays_fast_and_bounded(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Initiate the environment", tools=True, failed=True)
    first = router.on_llm_request(request=req, model=req["model"], api_call_count=2, turn_id="t-promote")
    first_ctk = first["request"]["extra_body"]["chat_template_kwargs"]
    assert first_ctk["enable_thinking"] is False
    assert "reasoning_effort" not in first_ctk
    assert len(first["request"]["messages"]) == 4
    assert len(first["request"]["tools"]) < len(req["tools"])
    clean = request("Initiate the environment")
    second = router.on_llm_request(request=clean, model=clean["model"], api_call_count=2, turn_id="t-promote")
    assert second["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_second_failure_promotes_to_full_high(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Initiate the environment", tools=True, failed=True)
    req["messages"].extend([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "2",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "2", "content": '{"ok":false,"error":"failed again"}'},
    ])
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=3, turn_id="t-two-failures"
    )
    assert out["request"]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"


def test_manual_high_is_absolute_override(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Initiate the environment", high=True)
    out = router.on_llm_request(request=req, model=req["model"], api_call_count=1, turn_id="t-high")
    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "high"
    assert len(out["request"]["messages"]) == len(req["messages"])


def test_uncertain_gliner_result_keeps_normal_harness(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence("unknown", "ambiguous", .1))
    req = request("Handle the earlier thing")
    out = router.on_llm_request(request=req, model=req["model"], api_call_count=1, turn_id="t-unknown")
    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert len(out["request"]["messages"]) == len(req["messages"])


def test_old_turn_failures_do_not_poison_new_user_turn(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    old_failure = {
        "role": "tool",
        "content": '{"ok": false, "error": "old task failed"}',
        "tool_call_id": "old",
    }
    req = request("Hello")
    req["messages"] = [
        {"role": "user", "content": "Previous task"},
        old_failure,
        {"role": "assistant", "content": "Previous task ended."},
        {"role": "user", "content": "Hello"},
    ]
    out = router.on_llm_request(request=req, model=req["model"], api_call_count=1, turn_id="t-new-turn")
    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is False


def test_successful_tool_output_that_mentions_error_stays_fast(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Inspect the source", tools=True)
    req["messages"][-1]["content"] = (
        '{"output":"error handler source; 0 errors",'
        '"exit_code":0,"error":null}'
    )
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=2, turn_id="t-success-error-word"
    )
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_successful_plain_skill_output_with_failure_words_is_not_failure(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Inspect the source", tools=True)
    req["messages"][-1]["content"] = (
        "Skill guide: handle error and timeout cases. This is documentation, not a failed call."
    )
    assert router._tool_result_failed(req["messages"]) is False
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=2, turn_id="t-plain-success"
    )
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_structured_success_with_error_field_is_not_failure():
    req = request("Inspect the source", tools=True)
    req["messages"][-1]["content"] = '{"ok":true,"error":"non-fatal warning"}'
    assert router._tool_result_failed(req["messages"]) is False


def test_explicit_plain_tool_error_is_failure():
    req = request("Inspect the source", tools=True)
    req["messages"][-1]["content"] = "ERROR: permission denied"
    assert router._tool_result_failed(req["messages"]) is True


def test_named_forced_tool_is_retained(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Perform the bounded operation")
    req["tools"].append(tool("custom_bounded_tool"))
    req["tool_choice"] = {"type": "function", "function": {"name": "custom_bounded_tool"}}
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-forced-tool"
    )
    names = {_tool["function"]["name"] for _tool in out["request"]["tools"]}
    assert "custom_bounded_tool" in names


def test_required_tool_request_never_becomes_empty(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Perform the required operation")
    req["tools"] = [tool("custom_only_tool")]
    req["tool_choice"] = "required"
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-required-tool"
    )
    assert [_tool["function"]["name"] for _tool in out["request"]["tools"]] == [
        "custom_only_tool"
    ]


def test_fast_lane_call_budget_promotes_to_compact_low(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Perform the bounded operation")
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=7, turn_id="t-call-budget"
    )
    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert len(out["request"]["messages"]) == 2
    assert len(out["request"]["tools"]) < len(req["tools"])


def test_fast_lane_hard_budget_forces_text_completion(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Perform the bounded operation")
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=11, turn_id="t-hard-budget"
    )
    assert "tools" not in out["request"]
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_retry_followup_inherits_prior_bounded_intent_and_clamps_broad_lane(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: None)
    req = request("please rety and continue")
    req["messages"] = [
        {"role": "system", "content": "large system"},
        {"role": "user", "content": "Add the two data options to the card"},
        {"role": "assistant", "content": "I stopped before completing the edit."},
        {"role": "user", "content": "please rety and continue"},
    ]
    router._TURN_LANES["t-inherit"] = ("standard", router.time.monotonic())
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-inherit"
    )
    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is False
    assert len(out["request"]["tools"]) < len(req["tools"])
    assert router._routing_user_text(req["messages"]) == "Add the two data options to the card"
    assert "inherited prior task" in out["reason"]


def test_natural_language_continue_question_inherits_prior_intent():
    messages = [
        {"role": "user", "content": "Replace the carousel images"},
        {"role": "assistant", "content": "The provider failed before completion."},
        {"role": "user", "content": "So are we able to continue?"},
    ]

    assert router._routing_user_context(messages) == ("Replace the carousel images", True)


def test_first_failure_does_not_turn_standard_reasoning_off(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: None)
    req = request("So are we able to continue?", tools=True, failed=True)
    req["messages"] = [
        {"role": "user", "content": "Replace the carousel images from the reference page"},
        {"role": "assistant", "content": "Paused."},
        {"role": "user", "content": "So are we able to continue?"},
        *req["messages"][-2:],
    ]
    router._TURN_LANES["t-preserve-low"] = ("standard", router.time.monotonic())

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=4, turn_id="t-preserve-low"
    )

    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert "preserve compact reasoning" in out["reason"]


def test_retry_followup_reaches_compact_and_finalize_budgets(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: None)
    req = request("retry and continue")
    req["messages"] = [
        {"role": "user", "content": "Update the card title"},
        {"role": "assistant", "content": "Paused."},
        {"role": "user", "content": "retry and continue"},
    ]
    router._TURN_LANES["t-inherit-budget"] = ("standard", router.time.monotonic())
    compact = router.on_llm_request(
        request=req, model=req["model"], api_call_count=7, turn_id="t-inherit-budget"
    )
    assert compact["request"]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert len(compact["request"]["tools"]) < len(req["tools"])
    final = router.on_llm_request(
        request=req, model=req["model"], api_call_count=11, turn_id="t-inherit-budget"
    )
    assert "tools" not in final["request"]


def test_bounded_clamp_never_demotes_full_lane(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Retry and continue")
    req["messages"] = [
        {"role": "user", "content": "Fix the card"},
        {"role": "assistant", "content": "Paused."},
        {"role": "user", "content": "Retry and continue"},
    ]
    router._TURN_LANES["t-full-sticky"] = ("full", router.time.monotonic())
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-full-sticky"
    )
    assert out["request"]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "high"


def test_attached_context_size_does_not_force_full_reasoning(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request(
        "Create content in that modal and add it to the table\n\n--- Attached Context ---\n"
        + ("reference material " * 500)
    )
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-attachment"
    )
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_bounded_edit_forces_mutation_tools_after_three_reads(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Add the text to the card")
    for index, name in enumerate(("read_file", "search_files", "read_file"), start=1):
        req["messages"].extend([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": str(index), "type": "function",
                    "function": {"name": name, "arguments": '{"query":"item-%d"}' % index},
                }],
            },
            {"role": "tool", "tool_call_id": str(index), "content": '{"ok":true}'},
        ])
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=4, turn_id="t-mutation-gate"
    )
    names = {_tool["function"]["name"] for _tool in out["request"]["tools"]}
    assert names == {"patch", "write_file"}
    assert "Do not read or search again" in out["request"]["messages"][0]["content"]


def test_discovery_rounds_do_not_consume_fast_action_budget(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Continue the bounded operation")
    req["messages"].extend([
        {"role": "assistant", "tool_calls": [{"id": "s", "function": {"name": "tool_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "s", "content": '{"ok":true}'},
        {"role": "assistant", "tool_calls": [{"id": "d", "function": {"name": "tool_describe", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "d", "content": '{"ok":true}'},
    ])
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=7, turn_id="t-discovery-budget"
    )
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_fast_lane_keeps_one_recent_conversation_exchange(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Continue working on it")
    req["messages"] = [
        {"role": "system", "content": "large system"},
        {"role": "user", "content": "Is the server running?"},
        {"role": "assistant", "content": "Yes, it is live on port 5173."},
        {"role": "user", "content": "Continue working on it"},
    ]
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-recent-context"
    )
    compact = out["request"]["messages"]
    assert [message["role"] for message in compact] == ["system", "user", "assistant", "user"]
    assert "port 5173" in compact[2]["content"]


def test_fast_lane_drops_prior_exchange_tool_trace(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Continue with the phone images")
    req["messages"] = [
        {"role": "system", "content": "large system"},
        {"role": "user", "content": "Update the product cards"},
    ]
    for index in range(16):
        call_id = f"prior-{index}"
        req["messages"].extend([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "x" * 8000,
            },
        ])
    req["messages"].extend([
        {"role": "assistant", "content": "The product cards are updated."},
        {"role": "user", "content": "Continue with the phone images"},
    ])

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-drop-prior-trace"
    )
    compact = out["request"]["messages"]

    assert [message["role"] for message in compact] == ["system", "user", "assistant", "user"]
    assert compact[1]["content"] == "Update the product cards"
    assert compact[2]["content"] == "The product cards are updated."
    assert all(message["role"] != "tool" for message in compact)
    assert len(compact) == 4


def test_referential_retry_skips_empty_turn_and_recovers_original_edit(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Okay let's try this again with the heading changes I mentioned")
    req["messages"] = [
        {"role": "system", "content": "large system"},
        {"role": "user", "content": "Make the step headings thinner"},
        {"role": "assistant", "content": "The headings are updated."},
        {"role": "user", "content": "I'm not seeing the changes reflected"},
        {"role": "assistant", "content": "(empty)", "_empty_terminal_sentinel": True},
        {"role": "user", "content": "Okay let's try this again with the heading changes I mentioned"},
    ]

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-referential-retry"
    )

    compact = out["request"]["messages"]
    assert [message["content"] for message in compact[1:]] == [
        "Make the step headings thinner",
        "The headings are updated.",
        "Okay let's try this again with the heading changes I mentioned",
    ]
    assert "Its original task is:\nMake the step headings thinner" in compact[0]["content"]
    assert out["request"]["tool_choice"] == "required"


def test_repeated_referential_retries_resolve_back_to_original_task():
    messages = [
        {"role": "user", "content": "Make the headings thinner but not Step 1, 2, 3"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "I'm not seeing the changes reflected"},
        {"role": "assistant", "content": "(empty)", "_empty_terminal_sentinel": True},
        {"role": "user", "content": "Let's try the heading changes I mentioned again"},
        {"role": "assistant", "content": "I reached the action limit."},
        {"role": "user", "content": "Try that task again"},
    ]

    effective, inherited = router._routing_user_context(messages)

    assert inherited is True
    assert effective == "Make the headings thinner but not Step 1, 2, 3"


def test_two_distinct_bounded_tool_failures_stay_compact(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    monkeypatch.setattr(
        router,
        "_frontier_state",
        lambda messages: {"failure_count": 2, "max_repeated_signature": 1},
    )
    req = request("Make the step headings thinner")

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=3, turn_id="t-two-distinct-failures"
    )

    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert len(out["request"]["tools"]) < len(req["tools"])
    assert "two distinct bounded tool failures" in out["reason"]


def test_fast_lane_self_contained_request_drops_prior_exchange(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Grab all phone images from the supplied URL")
    req["messages"] = [
        {"role": "system", "content": "large system"},
        {"role": "user", "content": "Update the product cards"},
        {"role": "assistant", "content": "The product cards are updated."},
        {"role": "user", "content": "Grab all phone images from the supplied URL"},
    ]

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-new-self-contained"
    )
    compact = out["request"]["messages"]

    assert [message["role"] for message in compact] == ["system", "user"]
    assert compact[1]["content"] == "Grab all phone images from the supplied URL"


def test_fast_lane_preserves_workspace_boundary(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Make the heading thinner")
    req["messages"][0]["content"] = (
        "Host details\nCurrent working directory: /Users/me/ui-next\nMore host details"
    )

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-workspace-boundary"
    )

    prompt = out["request"]["messages"][0]["content"]
    assert "Workspace boundary: /Users/me/ui-next" in prompt
    assert "Never search the home directory" in prompt


def test_quick_prose_response_has_no_tools_and_no_thinking(monkeypatch):
    monkeypatch.setattr(
        router, "_consume_gliner_evidence", lambda: evidence("fast", "quick_response", .9)
    )
    req = request("What does Vite host binding mean?")
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-prose"
    )
    assert "tools" not in out["request"]
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_quick_response_evidence_cannot_remove_tools_from_explicit_edit(monkeypatch):
    """Intent guards outrank a stale GLiNER quick-response classification."""
    monkeypatch.setattr(
        router, "_consume_gliner_evidence", lambda: evidence("fast", "quick_response", .9)
    )
    req = request("Make the step headings a bit thinner")

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-quick-edit"
    )

    names = {tool["function"]["name"] for tool in out["request"]["tools"]}
    assert {"patch", "write_file"} <= names
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_weight_decrease_request_is_a_required_edit(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request(
        "Decrease the main step header weights by one or two points; do not change the red number subheadings."
    )
    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=1, turn_id="t-weight-decrease"
    )

    assert out["request"]["tool_choice"] == "required"


def test_full_lane_preserves_workspace_boundary(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Decrease the main step header weights")
    req["messages"][0]["content"] = (
        "Current working directory: /Users/kristian/vf-supermobile-ui-next"
    )
    req["messages"].extend([
        {"role": "assistant", "tool_calls": [{
            "id": "same", "type": "function",
            "function": {"name": "search_files", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "same", "content": "same result"},
        {"role": "assistant", "tool_calls": [{
            "id": "same2", "type": "function",
            "function": {"name": "search_files", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "same2", "content": "same result"},
    ])
    router._TURN_LANES["t-full-workspace"] = ("full", router.time.monotonic())

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=6, turn_id="t-full-workspace"
    )
    system = out["request"]["messages"][0]["content"]
    assert "Workspace boundary: /Users/kristian/vf-supermobile-ui-next" in system
    assert "similarly named sibling project" in system


@pytest.mark.parametrize(
    "text",
    [
        "Do another sweep for edge cases. If you spot anything, do not implement it.",
        "Please analyze the code first. Do not change anything.",
        "Review the current behavior without modifying the prototype.",
    ],
)
def test_negated_review_request_does_not_force_mutation(monkeypatch, text):
    """Read-only review instructions must not acquire a required edit tool."""
    monkeypatch.setattr(
        router, "_consume_gliner_evidence", lambda: evidence("fast", "quick_response", .9)
    )
    out = router.on_llm_request(
        request=request(text), model="GLM-5.3-Flash-EXL3", api_call_count=1,
        turn_id="t-negated-review-" + str(abs(hash(text))),
    )

    assert out["request"].get("tool_choice") != "required"


def test_explicit_edit_can_finalize_after_a_mutation_tool_lands(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Make the step headings a bit thinner")
    req["messages"].extend([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "edit-1",
                "type": "function",
                "function": {"name": "patch", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "edit-1", "content": '{"success":true}'},
    ])

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=2, turn_id="t-edit-landed"
    )

    assert out["request"].get("tool_choice") != "required"
    assert out["reason"].startswith("verify_compact:")
    assert "at most one proportional verification" in out["request"]["messages"][0]["content"]


def test_edit_finalizes_after_one_post_mutation_read(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Make the step headings a bit thinner")
    req["messages"].extend([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "edit", "type": "function",
                "function": {"name": "patch", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "edit", "content": '{"success":true}'},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "verify", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "verify", "content": "font-weight: 400"},
    ])

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=3, turn_id="t-edit-verified"
    )

    assert "tools" not in out["request"]
    assert out["request"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert "verification attempted; finalize" in out["reason"]


def test_failed_write_is_not_treated_as_a_landed_edit(monkeypatch):
    """A stale-write refusal must keep edit tools open instead of finalizing."""
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Make the step headings a bit thinner")
    req["messages"].extend([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "edit", "type": "function",
                "function": {"name": "write_file", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "edit",
            "content": '{"error":"Refusing to overwrite: file was NOT modified",'
                       '"stale_write_blocked":true}',
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "read", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "read", "content": "current source"},
    ])

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=8, turn_id="t-failed-write"
    )

    names = {tool["function"]["name"] for tool in out["request"]["tools"]}
    assert names <= {"patch", "write_file"}
    assert names
    assert out["request"]["tool_choice"] == "required"
    assert "latest edit/verification failed; keep recovery tools available" in out["reason"]
    assert "finalize" not in out["reason"]


def test_failed_verification_does_not_close_tools(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Make the step headings a bit thinner")
    req["messages"].extend([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "edit", "type": "function",
                "function": {"name": "patch", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "edit", "content": '{"success":true}'},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "verify", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "verify", "content": '{"error":"read failed"}'},
    ])

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=3, turn_id="t-failed-verify"
    )

    assert out["request"].get("tools")
    assert "latest edit/verification failed; keep recovery tools available" in out["reason"]


def test_synthetic_recovery_user_does_not_replace_real_user_intent():
    messages = [
        {"role": "user", "content": "Start the server"},
        {"role": "user", "content": "Continue processing", "_empty_recovery_synthetic": True},
    ]
    assert router._latest_user_text(messages) == "Start the server"
    assert router._current_turn_messages(messages) == messages


def test_persisted_auto_continue_row_preserves_original_user_turn():
    messages = [
        {"role": "user", "content": "Add the two data options"},
        {"role": "assistant", "content": "working"},
        {
            "role": "user",
            "content": "[System note: Your previous turn was interrupted mid-run — resume]",
            "display_kind": "auto_continue",
        },
    ]
    assert router._latest_user_text(messages) == "Add the two data options"
    assert router._current_turn_messages(messages) == messages


def test_persisted_empty_recovery_text_is_synthetic_without_runtime_flag():
    messages = [
        {"role": "user", "content": "Start the server"},
        {"role": "user", "content": router._EMPTY_RECOVERY_TEXT},
    ]
    assert router._latest_user_text(messages) == "Start the server"


def test_empty_after_tool_recovery_immediately_enables_compact_reasoning(monkeypatch):
    """Do not repeat a thinking-off fast request after it already returned empty."""
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Make the heading thinner")
    req["messages"] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Make the heading thinner"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "source"},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "user",
            "content": router._EMPTY_RECOVERY_TEXT,
            "_empty_recovery_synthetic": True,
        },
    ]

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=4, turn_id="t-empty-recovery"
    )

    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert "empty-after-tool recovery" in out["reason"]


def test_empty_finalize_recovery_reopens_reasoning_but_not_tools(monkeypatch):
    """A sticky finalize lane must not repeat a known-empty thinking-off request."""
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("Make the heading thinner")
    req["messages"].extend([
        {"role": "assistant", "content": "(empty)", "_empty_recovery_synthetic": True},
        {
            "role": "user",
            "content": router._EMPTY_RECOVERY_TEXT,
            "_empty_recovery_synthetic": True,
        },
    ])
    router._TURN_LANES["t-empty-finalize"] = ("finalize", router.time.monotonic())

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=12, turn_id="t-empty-finalize"
    )

    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert "tools" not in out["request"]
    assert out["reason"].startswith("finalize_reasoning:empty finalization recovery")


def test_system_continue_does_not_demote_active_reasoning_lane(monkeypatch):
    monkeypatch.setattr(router, "_consume_gliner_evidence", lambda: evidence())
    req = request("[System: Continue now. Execute the required tool calls and only send your final answer after completing the task.]")
    req["messages"] = [
        {"role": "user", "content": "Replace the carousel images from the reference page"},
        {"role": "assistant", "content": "Paused."},
        {"role": "user", "content": req["messages"][-1]["content"]},
    ]
    router._TURN_LANES["t-system-continue"] = ("standard_compact", router.time.monotonic())

    out = router.on_llm_request(
        request=req, model=req["model"], api_call_count=6, turn_id="t-system-continue"
    )

    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert "synthetic continuation" in out["reason"]


def test_non_glm_is_untouched():
    assert router.on_llm_request(request={"model": "gpt-5.6-sol"}, model="gpt-5.6-sol") is None


def test_chat_template_off_is_really_off():
    assert router.chat_template_kwargs("low", thinking=False) == {
        "enable_thinking": False,
        "clear_thinking": True,
    }
