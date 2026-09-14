#!/usr/bin/env python3
"""Offline checks for glm-codex-effort (no GPU, no training)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from __init__ import choose_effort, on_llm_request, chat_template_kwargs  # noqa: E402


def msgs(text: str, tools: bool = False, failed: bool = False):
    out = [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]
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


def main() -> int:
    cases = [
        ("rename foo to bar in config.yaml", 1, False, False, "low"),
        ("change enable_thinking to false", 1, False, False, "low"),
        ("Why is the gateway hung? Investigate the root cause.", 1, False, False, "high"),
        ("implement a new auth system from scratch", 1, False, False, "high"),
        ("please add a login page with oauth and migrations", 1, False, False, "low"),
        ("tweak the comment", 2, True, False, "low"),
        ("debug this crash", 2, True, True, "high"),
    ]
    failed = []
    for text, n, tools, tool_fail, expect in cases:
        effort, reason = choose_effort(msgs(text, tools=tools, failed=tool_fail), n)
        ok = effort == expect
        print(f"{'OK' if ok else 'FAIL'} expect={expect} got={effort} ({reason}) :: {text[:60]}")
        if not ok:
            failed.append(text)

    req = {
        "model": "glm-5.3-flash-exl3-dspark",
        "messages": msgs("rename foo to bar"),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}},
    }
    out = on_llm_request(request=req, model="glm-5.3-flash-exl3-dspark", api_call_count=1)
    ctk = out["request"]["extra_body"]["chat_template_kwargs"]
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_effort"] == "low"
    assert ctk["clear_thinking"] is True
    print("middleware kwargs", ctk)

    skipped = on_llm_request(
        request={"model": "gpt-5.6-sol", "messages": msgs("rename x")},
        model="gpt-5.6-sol",
        api_call_count=1,
    )
    assert skipped is None
    print("non-glm skipped")

    assert chat_template_kwargs("medium")["reasoning_effort"] == "low"
    print("invalid effort clamped to low")

    if failed:
        print("FAILED", failed)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
