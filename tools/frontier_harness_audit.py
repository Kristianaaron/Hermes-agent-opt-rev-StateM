#!/usr/bin/env python3
"""Offline regression audit for the Hermes frontier harness.

This utility performs no model calls and is never imported by the runtime. It
summarizes persisted StateM events so harness changes can be compared without
adding latency or prompt tokens to normal sessions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _events(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.glob("*/events.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if isinstance(value, dict):
                    value["_run"] = path.parent.name
                    rows.append(value)
        except (OSError, ValueError, TypeError):
            continue
    return rows


def summarize(root: Path) -> dict[str, Any]:
    rows = _events(root)
    results = [row for row in rows if row.get("type") == "tool_result"]
    failures = Counter(
        str(row.get("failure_class"))
        for row in results
        if row.get("failure_class")
    )
    durations = [
        max(0, int(row.get("duration_ms") or 0))
        for row in results
        if isinstance(row.get("duration_ms"), (int, float))
    ]
    runs = {str(row.get("_run")) for row in rows}
    total = len(results)
    return {
        "schema_version": 2,
        "runs": len(runs),
        "events": len(rows),
        "tool_results": total,
        "failed_tool_results": sum(failures.values()),
        "failure_rate": round(sum(failures.values()) / total, 6) if total else 0.0,
        "failure_classes": dict(sorted(failures.items())),
        "no_progress_blocks": sum(row.get("type") == "no_progress_block" for row in rows),
        "unknown_mutation_blocks": sum(row.get("type") == "unknown_mutation_block" for row in rows),
        "duplicate_effect_blocks": sum(row.get("type") == "duplicate_effect_block" for row in rows),
        "operation_reconciliations": sum(row.get("type") == "operation_reconciled" for row in rows),
        "interaction_waits": sum(row.get("type") == "interaction_wait" for row in rows),
        "interaction_resolutions": sum(row.get("type") == "interaction_resolved" for row in rows),
        "tool_duration_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": max(durations, default=0),
        },
    }


def _regressions(current: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> list[str]:
    issues: list[str] = []
    current_rate = float(current.get("failure_rate") or 0.0)
    baseline_rate = float(baseline.get("failure_rate") or 0.0)
    if current_rate > baseline_rate + tolerance:
        issues.append(f"failure_rate {current_rate:.4f} > {baseline_rate:.4f} + {tolerance:.4f}")
    current_p95 = int((current.get("tool_duration_ms") or {}).get("p95") or 0)
    baseline_p95 = int((baseline.get("tool_duration_ms") or {}).get("p95") or 0)
    if baseline_p95 and current_p95 > baseline_p95 * 1.25:
        issues.append(f"tool p95 {current_p95}ms > 125% of baseline {baseline_p95}ms")
    for key in ("provider_502", "malformed_tool_call", "tool_timeout"):
        now = int((current.get("failure_classes") or {}).get(key) or 0)
        before = int((baseline.get("failure_classes") or {}).get(key) or 0)
        if now > before:
            issues.append(f"{key} count increased from {before} to {now}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".hermes/statem/runs")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-rate-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    report = summarize(args.state_dir.expanduser())
    exit_code = 0
    if args.baseline:
        baseline = json.loads(args.baseline.expanduser().read_text(encoding="utf-8"))
        report["regressions"] = _regressions(report, baseline, max(0.0, args.failure_rate_tolerance))
        exit_code = 2 if report["regressions"] else 0
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.expanduser().write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
