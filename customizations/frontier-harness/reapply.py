#!/usr/bin/python3
"""Reapply the local Hermes overlay after a completed upstream update.

The controller is intentionally conservative:
- It never runs while Hermes' shared update marker is live.
- It first checks whether the overlay is already present.
- It applies only when ``git apply --check`` succeeds cleanly.
- It never resets, checks out, stashes, or overwrites config.
- Any ambiguity creates NEEDS_ATTENTION and leaves upstream source untouched.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
REPO = HERMES_HOME / "hermes-agent"
BUNDLE = HERMES_HOME / "customizations" / "frontier-harness"
PATCH = BUNDLE / "overlay.patch"
MANIFEST = BUNDLE / "manifest.json"
CONFIG_SNAPSHOT = BUNDLE / "config.snapshot.yaml"
STATE = BUNDLE / "state.json"
LOCK = BUNDLE / "reapply.lock"
ATTENTION = BUNDLE / "NEEDS_ATTENTION"
LOG = HERMES_HOME / "customizations" / "logs" / "frontier-harness.log"
UPDATE_MARKER = HERMES_HOME / ".hermes-update-in-progress"
CONFIG = HERMES_HOME / "config.yaml"
MAX_UPDATE_MARKER_AGE = 20 * 60

CONFIG_INVARIANTS = (
    "deepseek-v4-flash-0731",
    "frontier_harness:",
    "verify_on_stop: auto",
    "reasoning_effort: max",
    "execution_governor:",
    "tool_loop_guardrails:",
    "checkpoints:",
    "LiquidAI/LFM2.5-VL-3B-MLX-8bit",
    "STATEFUL EXECUTION POLICY:",
    "platform_toolsets:",
    "statem_mcp_server.py",
    "fail_closed_when_active: true",
    "dsv4-vision:",
)

EXTERNAL_ASSETS = (
    HERMES_HOME / "bin" / "hermes-statem",
    REPO / "venv" / "bin" / "python",
    REPO / "venv" / "bin" / "statem",
    HERMES_HOME / "vendor" / "statem-20260820",
    HERMES_HOME / "statem" / "runbooks",
    HERMES_HOME / "bin" / "lfm25_vl_server.py",
    HERMES_HOME / "bin" / "start-lfm25-vl3b-local.sh",
    Path.home() / "Library/LaunchAgents/com.hermes.lfm25-vision.plist",
    Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-VL-3B-MLX-8bit",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} {message}\n")


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _head() -> str:
    result = _run_git("rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _update_is_live() -> bool:
    try:
        lines = UPDATE_MARKER.read_text(encoding="utf-8").splitlines()
        pid = int(lines[0].strip())
        started = float(lines[1].strip())
    except (OSError, IndexError, TypeError, ValueError):
        return False
    if time.time() - started > MAX_UPDATE_MARKER_AGE:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_state(status: str, **details: object) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": _now(),
        "status": status,
        "head": _head(),
        "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest()
        if PATCH.exists()
        else None,
        **details,
    }
    BUNDLE.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=BUNDLE)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, STATE)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _config_audit() -> list[str]:
    try:
        text = CONFIG.read_text(encoding="utf-8")
    except OSError:
        return ["config.yaml unreadable or missing"]
    return [marker for marker in CONFIG_INVARIANTS if marker not in text]


def _external_audit() -> list[str]:
    return [str(path) for path in EXTERNAL_ASSETS if not path.exists()]


def _bundle_audit() -> list[str]:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ["manifest.json unreadable or invalid"]
    issues = []
    expected_patch = str(manifest.get("overlay_sha256") or "")
    actual_patch = hashlib.sha256(PATCH.read_bytes()).hexdigest() if PATCH.exists() else ""
    if not expected_patch or expected_patch != actual_patch:
        issues.append("overlay.patch digest does not match manifest")
    expected_config = str(manifest.get("config_snapshot_sha256") or "")
    actual_config = hashlib.sha256(CONFIG_SNAPSHOT.read_bytes()).hexdigest() if CONFIG_SNAPSHOT.exists() else ""
    if expected_config and expected_config != actual_config:
        issues.append("config snapshot digest does not match manifest")
    for relative in manifest.get("owned_source_files") or []:
        if not (REPO / str(relative)).is_file():
            issues.append(f"owned source missing after overlay: {relative}")
    return issues


def _attention(reason: str, *, detail: str = "") -> int:
    message = f"{_now()} {reason}"
    if detail:
        message += f"\n{detail.strip()[:4000]}"
    ATTENTION.write_text(message + "\n", encoding="utf-8")
    os.chmod(ATTENTION, 0o600)
    _write_state("needs_attention", reason=reason, detail=detail.strip()[:4000])
    _log(f"NEEDS_ATTENTION: {reason}")
    return 2


def _apply_overlay() -> tuple[str, str]:
    reverse = _run_git("apply", "--reverse", "--check", str(PATCH))
    if reverse.returncode == 0:
        return "already_applied", ""

    check = _run_git("apply", "--check", "--whitespace=nowarn", str(PATCH))
    if check.returncode != 0:
        detail = check.stderr or check.stdout
        return "conflict", detail

    applied = _run_git("apply", "--whitespace=nowarn", str(PATCH))
    if applied.returncode != 0:
        detail = applied.stderr or applied.stdout
        return "apply_failed", detail
    return "applied", ""


def main() -> int:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        if _update_is_live():
            _write_state("deferred_update_in_progress")
            return 0
        if not REPO.is_dir() or not (REPO / ".git").exists():
            return _attention("Hermes Git checkout is missing")
        if not PATCH.is_file() or PATCH.stat().st_size == 0:
            return _attention("Overlay patch is missing or empty")

        result, detail = _apply_overlay()
        if result in {"conflict", "apply_failed"}:
            return _attention(
                "Overlay no longer applies cleanly; upstream source was left untouched",
                detail=detail,
            )

        bundle_issues = _bundle_audit()
        if bundle_issues:
            return _attention(
                "Customization bundle integrity check failed; upstream source was not overwritten further",
                detail="; ".join(bundle_issues),
            )

        missing_config = _config_audit()
        if missing_config:
            return _attention(
                "Hermes config invariants changed; config was not overwritten",
                detail="Missing markers: " + ", ".join(missing_config),
            )

        missing_assets = _external_audit()
        if missing_assets:
            return _attention(
                "External StateM or LFM assets are missing; nothing was overwritten",
                detail="Missing paths: " + ", ".join(missing_assets),
            )

        try:
            ATTENTION.unlink()
        except FileNotFoundError:
            pass
        _write_state(result, config_audit="passed")
        if result == "applied":
            _log(f"Overlay applied cleanly at HEAD {_head()[:12]}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
