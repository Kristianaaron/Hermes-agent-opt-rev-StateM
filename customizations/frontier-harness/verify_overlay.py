#!/usr/bin/env python3
"""Fast offline compatibility verification for the Desktop overlay."""

from __future__ import annotations

import argparse
import ast
import json
import py_compile
import subprocess
import sys
from pathlib import Path


PYTHON_FILES = (
    "agent/agent_runtime_helpers.py", "agent/chat_completion_helpers.py",
    "agent/conversation_loop.py", "agent/empty_response_guard.py",
    "agent/frontier_harness.py", "agent/statem_runtime.py",
    "agent/tool_argument_normalization.py", "agent/tool_executor.py",
    "agent/tool_guardrails.py", "agent/transports/chat_completions.py",
    "agent/turn_finalizer.py", "agent/turn_final_response.py",
    "agent/turn_request_assembly.py", "agent/turn_tool_round.py",
    "agent/chat_completion_nonstream.py",
    "hermes_cli/oneshot.py",
    "hermes_cli/gateway.py", "hermes_cli/plugins.py", "hermes_cli/timeouts.py",
    "model_tools.py",
    "tools/code_execution_tool.py", "tools/frontier_harness_audit.py",
    "tools/memory_tool.py", "tools/registry.py", "tools/statem_mcp_server.py",
    "tests/agent/test_frontier_harness.py", "tests/agent/test_tool_guardrails.py",
    "tests/hermes_cli/test_timeouts.py",
    "tests/test_dispatch_session_id.py", "tests/tools/test_registry.py",
    "tests/run_agent/test_sequential_tool_timeout.py",
)

SOURCE_MARKERS = {
    "agent/chat_completion_helpers.py": (
        "_provider_wait_notice",
        "_surface_countdown_wait",
        "Reasoning will appear automatically if the backend streams it",
        "_apply_commentary_gate_to_tools",
        "_strip_tools_from_kwargs_for_commentary_gate",
    ),
    "agent/conversation_loop.py": (
        "begin_turn_protocol", "observe_user_turn", "observe_user_goal",
        "_phase_after_commentary_gate",
        "Commentary gate: dropping tool_calls",
    ),
    "agent/turn_tool_round.py": ("observe_assistant_round", "consume_tool_only_brake_notice", "controlled_halt_response", "HOUSEKEEPING_TOOL_NAMES"),
    "agent/statem_runtime.py": (
        "TurnProtocol",
        '"tool_execution": "Frontier: executing tools"',
        "Prefill/think gaps stay pulse+timer only",
    ),
    "agent/tool_executor.py": (
        "normalize_tool_arguments", "record_tool_result",
        "_resolve_registered_tool_timeout",
    ),
    "agent/tool_guardrails.py": (
        "cross_turn_identical_block_after",
        "file_mutation_result_landed",
        "classify_user_goal_kind",
        "tool_only_reply_required_block",
        "work_tool_only_halt_after",
        "housekeeping_tool_only_halt_after",
        "commentary_gate_after",
        "commentary_gate_empty_cap",
        "begin_commentary_request",
        "stale_anchor_retry_block",
        "controlled_halt_response",
        "del has_reasoning",
    ),
    "agent/turn_request_assembly.py": (
        "begin_commentary_request",
        "Commentary gate: omitting tools",
    ),
    "agent/turn_final_response.py": (
        "finish_commentary_request",
        "Commentary gate: mid-turn status delivered",
    ),
    "agent/frontier_harness.py": (
        "apply_frontier_request_policy",
        "Successful terminal output can quote the word",
        "stop tools",
        "Codex turn rule",
        "commentary gate",
    ),
    "tools/code_execution_tool.py": ("_fixed_anc_node_timeout_hint", "contains_gateway_lifecycle_command"),
    "tools/memory_tool.py": ("new_text", "normalized_operations", "STALE ANCHOR"),
    "apps/desktop/src/store/reasoning-disclosure.ts": ("storedBoolean",),
    "hermes_cli/gateway.py": ("compatibility_entrypoint", "gateway_start.py"),
    "hermes_cli/plugins.py": ("execution_timeout_seconds", "accepts_progress_callback"),
    "hermes_cli/timeouts.py": ("_resolve_provider_config", "fails closed"),
    "model_tools.py": ("tool_progress_callback",),
    "tools/registry.py": ("execution_timeout_seconds", "accepts_progress_callback"),
    "apps/desktop/vitest.setup.ts": ("installedStorage?.getItem",),
}


def verify(repo: Path) -> dict:
    errors = []
    for relative in PYTHON_FILES:
        path = repo / relative
        if not path.is_file():
            errors.append("missing {}".format(relative))
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append("{}: {}: {}".format(relative, type(exc).__name__, exc))
    for relative, markers in SOURCE_MARKERS.items():
        path = repo / relative
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append("{} unreadable: {}".format(relative, exc))
            continue
        for marker in markers:
            if marker not in text:
                errors.append("{} missing semantic marker {}".format(relative, marker))
    smoke = subprocess.run(
        [sys.executable, "-c", (
            "import agent.frontier_harness, agent.statem_runtime, "
            "agent.tool_argument_normalization, agent.tool_guardrails; "
            "from agent.tool_argument_normalization import normalize_tool_arguments; "
            "assert normalize_tool_arguments('terminal', {'arguments': {'command': 'true'}})"
        )], cwd=str(repo), capture_output=True, text=True, timeout=60,
    )
    if smoke.returncode != 0:
        errors.append("runtime import smoke test failed: " + (smoke.stderr or smoke.stdout).strip())
    return {"status": "failed" if errors else "ok", "errors": errors, "repo": str(repo)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo.resolve())
    print(json.dumps(result, sort_keys=True))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
