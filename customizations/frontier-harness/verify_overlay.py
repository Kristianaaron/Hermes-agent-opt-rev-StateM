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
    "agent/sdk_transform_bypass.py",
    "hermes_cli/oneshot.py",
    "hermes_cli/gateway.py", "hermes_cli/plugins.py", "hermes_cli/timeouts.py",
    "model_tools.py",
    "tools/code_execution_tool.py", "tools/frontier_harness_audit.py",
    "tools/memory_tool.py", "tools/registry.py", "tools/statem_mcp_server.py",
    "tests/agent/test_frontier_harness.py", "tests/agent/test_tool_guardrails.py",
    "tests/hermes_cli/test_timeouts.py",
    "tests/test_dispatch_session_id.py", "tests/tools/test_registry.py",
)

SOURCE_MARKERS = {
    "agent/chat_completion_helpers.py": (
        "_provider_wait_notice",
        "_surface_countdown_wait",
        "_kill_hard_timeout",
        "_stream_hard_timeout = get_provider_hard_timeout",
        "_apply_commentary_gate_to_tools",
        "_strip_tools_from_kwargs_for_commentary_gate",
        "_omit_empty_tools_at_wire_boundary",
    ),
    "agent/sdk_transform_bypass.py": (
        "_omit_genuinely_empty_tool_fields",
        "bypass_chat_sdk_request_transform",
    ),
    "agent/turn_tool_round.py": ("_HOUSEKEEPING_TOOLS", "_toolguard_controlled_halt_response"),
    "agent/statem_runtime.py": (
        "TurnProtocol",
        "preflight_tool", "record_tool_result", "observe_user_turn",
    ),
    "agent/tool_executor.py": (
        "_resolve_registered_tool_timeout",
    ),
    "agent/tool_guardrails.py": (
        "ToolCallGuardrailController", "file_mutation_result_landed", "is_guardrail_refusal",
        "classify_user_goal_kind", "begin_commentary_request", "controlled_halt_response",
        "stale_anchor_retry_block", "cross_turn_identical_block_after",
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
