#!/usr/bin/env python3
"""Typed MCP adapter around official StateM for Hermes."""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, tempfile, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(SOURCE))
from agent.statem_runtime import ROOT, load_registry, update_run
from mcp.server import MCPServer

mcp = MCPServer("hermes-statem")
RUNBOOKS, RUNS = ROOT / "runbooks", ROOT / "runs"
STATEM_BIN = os.environ.get("STATEM_BIN", str(SOURCE / "venv" / "bin" / "statem"))
def _now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def _json(path: Path, default: Any):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError): return default
def _write(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
def _event(record, value):
    with (Path(record["run_dir"]) / "events.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"at": _now(), **value}, ensure_ascii=False) + "\n")
def _record(run_id):
    value = load_registry()["runs"].get(run_id)
    if not value: raise ValueError(f"unknown StateM run: {run_id}")
    return value
def _cli(record, *args):
    proc = subprocess.run([STATEM_BIN, *args, "--run-id", record["run_id"], "--state-dir", record["state_dir"], "--json"], cwd=record["workdir"], text=True, capture_output=True, timeout=60)
    output = (proc.stdout or proc.stderr or "").strip()
    try: payload = json.loads(output)
    except ValueError: payload = {"output": output}
    if proc.returncode: raise RuntimeError(json.dumps({"returncode": proc.returncode, "details": payload}, ensure_ascii=False))
    return payload
def _route(task, risk):
    low = task.lower()
    if risk in {"high", "production"} or any(x in low for x in ("migrate", "deploy", "architecture", "multi-file", "investigate", "production", "security", "database")): return "full"
    if len(task.split()) <= 10 and not any(x in low for x in ("fix", "build", "change", "implement", "debug", "test")): return "direct"
    return "compact"
def _state(payload, fallback):
    if isinstance(payload, dict):
        for key in ("current", "node", "state", "current_node"):
            value = payload.get(key)
            if isinstance(value, str): return value
            if isinstance(value, dict):
                for nested in ("name", "id", "node"):
                    if isinstance(value.get(nested), str): return value[nested]
    return fallback
def _packet(run_id):
    record = _record(run_id); contract = _json(Path(record["run_dir"]) / "contract.json", {}); events = []
    try: events = [json.loads(x) for x in (Path(record["run_dir"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()[-12:]]
    except (OSError, ValueError): pass
    return {"run_id": run_id, "route": record.get("route"), "state": record.get("state"), "status": record.get("status"), "objective": str(contract.get("task", ""))[:2000], "pending_interaction": record.get("pending_interaction"), "unresolved_operation": record.get("inflight_operation"), "recent_events": events, "artifacts": record.get("run_dir"), "instruction": "Continue only the current phase; never replay an effect with unknown outcome, and wait for a real user turn when interaction is pending."}

@mcp.tool()
def statem_begin(task: str, workdir: str, route: str = "auto", risk: str = "normal", verification: str = "") -> dict:
    """Start an adaptive durable run: auto, direct, compact, or full."""
    chosen = _route(task, risk) if route == "auto" else route
    if chosen not in {"direct", "compact", "full"}: raise ValueError("invalid route")
    if chosen == "direct": return {"route": "direct", "instruction": "Answer directly without a tool loop."}
    resolved = Path(workdir).expanduser().resolve()
    if not resolved.is_dir(): raise ValueError(f"invalid workdir: {resolved}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir, state_dir = RUNS / run_id, RUNS / run_id / "runtime"; run_dir.mkdir(parents=True)
    _write(run_dir / "contract.json", {"task": task, "risk": risk, "verification": verification, "workdir": str(resolved), "created_at": _now()})
    (run_dir / "progress.md").write_text(f"# Progress\n\n## Objective\n{task}\n\n## Discoveries\n\n## Decisions\n\n## Verification\n", encoding="utf-8")
    initial = "start" if chosen == "full" else "execute"
    record = update_run(run_id, status="active", route=chosen, state=initial, entry=1, workdir=str(resolved), run_dir=str(run_dir), state_dir=str(state_dir), cancel_requested=False, pending_interaction=None, inflight_operation=None, effect_ledger=[])
    payload = _cli(record, "start", str(RUNBOOKS / f"{chosen}.yaml")); current = _state(payload, initial); update_run(run_id, state=current); _event(record, {"type": "run_started", "route": chosen, "state": current})
    return {"run_id": run_id, "route": chosen, "state": current, "phase_packet": _packet(run_id)}
@mcp.tool()
def statem_current(run_id: str) -> dict:
    """Return current durable state and official prompt."""
    record = _record(run_id); payload = _cli(record, "cur"); current = _state(payload, record.get("state", "start")); update_run(run_id, state=current); return {"run_id": run_id, "status": record.get("status"), "state": current, "statem": payload}
@mcp.tool()
def statem_next(run_id: str) -> dict:
    """List legal checked transitions."""
    return _cli(_record(run_id), "next")
def _checks(record):
    results = []
    for check in _json(Path(record["run_dir"]) / "checks.json", []):
        if check.get("state") != record.get("state") or check.get("entry") != record.get("entry") or not check.get("blocking", True): continue
        argv = check.get("command")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv): results.append({"name": check.get("name"), "passed": False, "error": "invalid argv"}); continue
        proc = subprocess.run(argv, cwd=record["workdir"], text=True, capture_output=True, timeout=int(check.get("timeout", 180)))
        results.append({"name": check.get("name"), "passed": proc.returncode == 0, "returncode": proc.returncode, "output": (proc.stdout or proc.stderr or "")[-4000:]})
    return results
@mcp.tool()
def statem_transition(run_id: str, target: str, evidence: list[str], decision: str = "") -> dict:
    """Attempt a fail-closed transition with concrete evidence and checks."""
    record = _record(run_id)
    if record.get("status") != "active": raise RuntimeError("run must be active")
    pending = record.get("pending_interaction")
    if isinstance(pending, dict) and not pending.get("resolved_at"): raise RuntimeError("a real user response is still pending")
    inflight = record.get("inflight_operation")
    if isinstance(inflight, dict) and inflight.get("status") in {"dispatched", "unknown"}: raise RuntimeError("an effect has unknown outcome; reconcile it before transition")
    source, evidence = str(record.get("state") or ""), [x.strip() for x in evidence if x.strip()]
    if source in {"plan", "execute", "verify", "repair"} and not evidence: raise ValueError(f"transition from {source} requires evidence")
    checks = _checks(record); failed = [x for x in checks if not x.get("passed")]
    receipt = {"source": source, "target": target, "evidence": evidence, "decision": decision, "checks": checks, "at": _now()}; rid = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()[:16]; _write(Path(record["run_dir"]) / "receipts" / f"{rid}.json", receipt)
    if failed: _event(record, {"type": "transition_blocked", "receipt": rid, "failed_checks": failed}); return {"transitioned": False, "state": source, "receipt": rid, "failed_checks": failed}
    payload = _cli(record, "goto", target, "--yes"); entry = int(record.get("entry", 0)) + 1; update_run(run_id, state=target, entry=entry, last_action={}, cancel_requested=False, status="complete" if target == "handoff" else "active"); _event(record, {"type": "transition", "source": source, "target": target, "receipt": rid}); return {"transitioned": True, "state": target, "entry": entry, "receipt": rid, "statem": payload}
@mcp.tool()
def statem_add_check(run_id: str, name: str, command: list[str], timeout: int = 180, blocking: bool = True) -> dict:
    """Register an argv verification check for the current state entry."""
    record = _record(run_id)
    if not command or not all(isinstance(x, str) and x for x in command): raise ValueError("command must be argv")
    path = Path(record["run_dir"]) / "checks.json"; checks = _json(path, []); item = {"id": uuid.uuid4().hex[:12], "state": record.get("state"), "entry": record.get("entry"), "name": name, "command": command, "timeout": max(1, min(timeout, 1800)), "blocking": blocking}; checks.append(item); _write(path, checks); return item
@mcp.tool()
def statem_record_evidence(run_id: str, summary: str, artifacts: list[str] | None = None, failure_class: str = "") -> dict:
    """Persist discovery, decision, verification, or classified failure."""
    record = _record(run_id); value = {"type": "evidence", "state": record.get("state"), "summary": summary[:4000], "artifacts": artifacts or [], "failure_class": failure_class or None}; _event(record, value); return {"recorded": True, **value}
@mcp.tool()
def statem_checkpoint(run_id: str, paths: list[str], label: str = "checkpoint") -> dict:
    """Checkpoint explicit files before consequential edits."""
    record = _record(run_id); cid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]; root = Path(record["run_dir"]) / "checkpoints" / cid; manifest = {"id": cid, "label": label, "files": [], "created_at": _now()}; workdir = Path(record["workdir"])
    for raw in paths:
        source = Path(raw).expanduser().resolve()
        try: relative = source.relative_to(workdir)
        except ValueError: raise ValueError(f"path outside workdir: {source}")
        if source.is_file():
            target = root / "files" / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target); manifest["files"].append({"path": str(relative), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    _write(root / "manifest.json", manifest); return manifest
@mcp.tool()
def statem_rollback(run_id: str, checkpoint_id: str) -> dict:
    """Restore only files captured by an explicit checkpoint."""
    record = _record(run_id); root = Path(record["run_dir"]) / "checkpoints" / checkpoint_id; manifest = _json(root / "manifest.json", None)
    if not manifest: raise ValueError(f"unknown checkpoint: {checkpoint_id}")
    restored, workdir = [], Path(record["workdir"])
    for item in manifest.get("files", []):
        relative = Path(item["path"]); target = workdir / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root / "files" / relative, target); restored.append(str(target))
    _event(record, {"type": "rollback", "checkpoint": checkpoint_id, "restored": restored}); return {"restored": restored}
@mcp.tool()
def statem_pause(run_id: str, reason: str = "user requested pause") -> dict:
    """Pause authoritatively; subsequent model and tool actions fail closed."""
    record = update_run(run_id, status="paused", cancel_requested=True, pause_reason=reason); _event(record, {"type": "paused", "reason": reason}); return {"run_id": run_id, "status": "paused"}
@mcp.tool()
def statem_wait(run_id: str, kind: str, prompt: str, scope: str = "") -> dict:
    """Wait fail-closed for a real subsequent user clarification or authorization."""
    if kind not in {"clarification", "authorization"}: raise ValueError("kind must be clarification or authorization")
    if not prompt.strip(): raise ValueError("prompt is required")
    record = _record(run_id)
    if record.get("status") != "active": raise RuntimeError("run must be active")
    interaction = {"kind": kind, "prompt": prompt.strip()[:2000], "scope": scope.strip()[:500], "requested_at": _now(), "requested_state": record.get("state"), "resolved_at": None}
    updated = update_run(run_id, status=f"waiting_{kind}", pending_interaction=interaction)
    _event(updated, {"type": "interaction_wait", "kind": kind, "scope": scope.strip()[:500]})
    return {"run_id": run_id, "status": f"waiting_{kind}", "prompt": interaction["prompt"], "instruction": "End the turn now. Do not infer or fabricate the user's answer."}
@mcp.tool()
def statem_reconcile_operation(run_id: str, operation_id: str, outcome: str, evidence: list[str]) -> dict:
    """Resolve a non-atomic effect using independently observed evidence."""
    if outcome not in {"applied", "not_applied", "rolled_back"}: raise ValueError("invalid reconciliation outcome")
    cleaned = [item.strip()[:2000] for item in evidence if item.strip()]
    if not cleaned: raise ValueError("reconciliation requires concrete evidence")
    record = _record(run_id); ledger = [item for item in (record.get("effect_ledger") or []) if isinstance(item, dict)]
    effect = next((item for item in reversed(ledger) if item.get("operation_id") == operation_id), None)
    if effect is None: raise ValueError(f"unknown operation: {operation_id}")
    status = "verified" if outcome == "applied" else outcome
    updated_effect = {**effect, "status": status, "reconciled_at": _now(), "reconciliation_evidence": cleaned}
    ledger = [item for item in ledger if item.get("operation_id") != operation_id] + [updated_effect]
    inflight = record.get("inflight_operation")
    clear = isinstance(inflight, dict) and inflight.get("operation_id") == operation_id
    updated = update_run(run_id, effect_ledger=ledger[-32:], inflight_operation=None if clear else inflight)
    _event(updated, {"type": "operation_reconciled", "operation_id": operation_id, "outcome": outcome, "evidence": cleaned})
    return {"run_id": run_id, "operation_id": operation_id, "outcome": outcome, "status": status}
@mcp.tool()
def statem_resume(run_id: str) -> dict:
    """Resume from durable current state."""
    current = _record(run_id); pending = current.get("pending_interaction")
    if isinstance(pending, dict) and not pending.get("resolved_at"): raise RuntimeError("cannot resume until a real subsequent user turn resolves the pending interaction")
    record = update_run(run_id, status="active", cancel_requested=False); _event(record, {"type": "resumed", "state": record.get("state")}); return {"run_id": run_id, "status": "active", "phase_packet": _packet(run_id)}
@mcp.tool()
def statem_phase_packet(run_id: str) -> dict:
    """Return bounded resume context while history stays in artifacts."""
    return _packet(run_id)
if __name__ == "__main__": mcp.run()
