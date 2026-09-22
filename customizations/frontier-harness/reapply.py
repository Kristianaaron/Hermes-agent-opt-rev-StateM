#!/usr/bin/python3
"""Transactionally preserve the Hermes Desktop customization overlay."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import plistlib
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


HERMES_HOME = Path(
    os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
).expanduser()
REPO = HERMES_HOME / "hermes-agent"
BUNDLE = HERMES_HOME / "customizations" / "frontier-harness"
PATCH = BUNDLE / "overlay.patch"
MANIFEST = BUNDLE / "manifest.json"
CONFIG_SNAPSHOT = BUNDLE / "config.snapshot.yaml"
VERIFY = BUNDLE / "verify_overlay.py"
STATE = BUNDLE / "state.json"
LOCK = BUNDLE / "reapply.lock"
ATTENTION = BUNDLE / "NEEDS_ATTENTION"
LOG = HERMES_HOME / "customizations" / "logs" / "frontier-harness.log"
UPDATE_MARKER = HERMES_HOME / ".hermes-update-in-progress"
CONFIG = HERMES_HOME / "config.yaml"
PYTHON = REPO / "venv" / "bin" / "python"
GLINER_PLUGIN = HERMES_HOME / "plugins" / "gliner-extract" / "__init__.py"
SYSTEM1_PLUGIN = HERMES_HOME / "plugins" / "glm-codex-effort" / "__init__.py"
CUSTOMIZATION_PLIST = Path.home() / "Library" / "LaunchAgents" / "ai.hermes.customizations.plist"
MAX_UPDATE_MARKER_AGE = 20 * 60

# Model-agnostic manager/worker behavior. Switching providers must not trip
# the Desktop compatibility gate.
CONFIG_INVARIANTS = (
    "frontier_harness:",
    "verify_on_stop: auto",
    "reasoning_effort:",
    "execution_governor:",
    "tool_loop_guardrails:",
    "max_tool_only_iterations:",
    "work_tool_only_halt_after:",
    "housekeeping_tool_only_halt_after:",
    "commentary_gate_after:",
    "stale_anchor_block_after:",
    "ping_max_tool_only:",
    "checkpoints:",
    "STATEFUL EXECUTION POLICY:",
    "platform_toolsets:",
    "statem_mcp_server.py",
    "fail_closed_when_active: true",
    "stale_timeout_seconds: 21600",
    "context_length: 1048576",
    "context_timeout_seconds: 1800",
)

EXTERNAL_ASSETS = (
    HERMES_HOME / "bin" / "hermes-statem",
    PYTHON,
    REPO / "venv" / "bin" / "statem",
    HERMES_HOME / "vendor" / "statem-20260820",
    HERMES_HOME / "statem" / "runbooks",
    HERMES_HOME / "bin" / "lfm25_vl_server.py",
    HERMES_HOME / "bin" / "start-lfm25-vl3b-local.sh",
    GLINER_PLUGIN.parent,
    SYSTEM1_PLUGIN.parent,
    Path.home() / "Library" / "LaunchAgents" / "com.hermes.lfm25-vision.plist",
)


def _customization_launch_agent_payload() -> dict:
    return {
        "Label": "ai.hermes.customizations",
        "ProgramArguments": ["/usr/bin/python3", str(BUNDLE / "reapply.py")],
        "RunAtLoad": True,
        "StartInterval": 300,
        "ThrottleInterval": 15,
        "LowPriorityIO": True,
        "ProcessType": "Background",
        "WatchPaths": [
            str(UPDATE_MARKER),
            str(REPO / ".git" / "HEAD"),
            str(REPO / ".git" / "refs" / "heads" / "main"),
            str(Path.home() / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist"),
        ],
        "StandardOutPath": str(HERMES_HOME / "customizations" / "logs" / "launchagent.stdout.log"),
        "StandardErrorPath": str(HERMES_HOME / "customizations" / "logs" / "launchagent.stderr.log"),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write("{} {}\n".format(_now(), message))


def _run(args, cwd=None, timeout=300):
    return subprocess.run(
        [str(item) for item in args], cwd=str(cwd or REPO), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )


def _git(*args, **kwargs):
    return _run(["git", *args], cwd=kwargs.get("cwd", REPO))


def _head(repo=REPO) -> str:
    result = _git("rev-parse", "HEAD", cwd=repo)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _update_is_live() -> bool:
    try:
        lines = UPDATE_MARKER.read_text(encoding="utf-8").splitlines()
        pid, started = int(lines[0].strip()), float(lines[1].strip())
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


def _write_state(status: str, **details) -> None:
    payload = {
        "schema_version": 2, "updated_at": _now(), "status": status,
        "head": _head(), "patch_sha256": _sha256(PATCH), **details,
    }
    BUNDLE.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=str(BUNDLE))
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


def _attention(reason: str, detail: str = "") -> int:
    message = "{} {}".format(_now(), reason)
    if detail:
        message += "\n" + detail.strip()[:4000]
    ATTENTION.write_text(message + "\n", encoding="utf-8")
    os.chmod(ATTENTION, 0o600)
    _write_state("needs_attention", reason=reason, detail=detail.strip()[:4000])
    _log("NEEDS_ATTENTION: {}: {}".format(reason, detail.strip()[:500]))
    return 2


def _read_manifest():
    try:
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("manifest.json unreadable or invalid: {}".format(exc))
    if not isinstance(value, dict):
        raise RuntimeError("manifest.json must contain an object")
    return value


def _bundle_integrity(manifest) -> list:
    issues = []
    if manifest.get("overlay_sha256") != _sha256(PATCH):
        issues.append("overlay.patch digest does not match manifest")
    if manifest.get("config_snapshot_sha256") != _sha256(CONFIG_SNAPSHOT):
        issues.append("config snapshot digest does not match manifest")
    if not VERIFY.is_file():
        issues.append("verify_overlay.py is missing")
    protected = {
        "controller_sha256": BUNDLE / "reapply.py",
        "verifier_sha256": VERIFY,
        "gateway_entrypoint_sha256": BUNDLE / "gateway_start.py",
        "gliner_extract_sha256": GLINER_PLUGIN,
        "glm_system1_router_sha256": SYSTEM1_PLUGIN,
    }
    for key, path in protected.items():
        if manifest.get(key) != _sha256(path):
            issues.append("{} digest does not match manifest".format(path.name))
    patch_paths = set()
    listed = _git("apply", "--numstat", str(PATCH))
    if listed.returncode != 0:
        issues.append("overlay.patch paths could not be enumerated")
    else:
        for line in listed.stdout.splitlines():
            fields = line.split("\t", 2)
            if len(fields) == 3:
                patch_paths.add(fields[2])
    owned_paths = {str(path) for path in manifest.get("owned_source_files") or []}
    undeclared = sorted(patch_paths - owned_paths)
    stale = sorted(owned_paths - patch_paths)
    if undeclared:
        issues.append("overlay paths missing from manifest: " + ", ".join(undeclared))
    if stale:
        issues.append("manifest paths missing from overlay: " + ", ".join(stale))
    return issues


def _config_audit() -> list:
    try:
        text = CONFIG.read_text(encoding="utf-8")
    except OSError:
        return ["config.yaml unreadable or missing"]
    return [marker for marker in CONFIG_INVARIANTS if marker not in text]


def _external_audit() -> list:
    return [str(path) for path in EXTERNAL_ASSETS if not path.exists()]


def _bundle_has_revision(bundle: Path, revision: str, cwd: Path) -> bool:
    result = _run(["git", "bundle", "list-heads", bundle], cwd=cwd, timeout=30)
    if result.returncode != 0:
        return False
    return any(line.split(maxsplit=1)[0] == revision for line in result.stdout.splitlines())


def _restore_checkout_if_missing(path: Path, bundle: Path, revision: str) -> list:
    """Recover a deleted integration only; never overwrite an existing directory."""
    if path.exists():
        return []
    if not bundle.is_file() or not revision:
        return ["cannot restore {}: recovery bundle or revision is missing".format(path)]
    path.parent.mkdir(parents=True, exist_ok=True)
    restored = _run(["git", "clone", "--no-checkout", bundle, path], cwd=path.parent, timeout=60)
    if restored.returncode != 0:
        return ["could not restore {}: {}".format(path, (restored.stderr or restored.stdout).strip()[:500])]
    checked = _run(["git", "checkout", "--detach", revision], cwd=path, timeout=30)
    if checked.returncode != 0:
        shutil.rmtree(path, ignore_errors=True)
        return ["could not select approved revision for {}: {}".format(
            path, (checked.stderr or checked.stdout).strip()[:500]
        )]
    return []



def _ensure_customization_service_definition():
    """Keep the post-update reconciliation job present across installer rewrites."""
    expected = _customization_launch_agent_payload()
    current = None
    try:
        with CUSTOMIZATION_PLIST.open("rb") as handle:
            current = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException):
        pass
    changed = current != expected
    if changed:
        CUSTOMIZATION_PLIST.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".ai.hermes.customizations.", suffix=".plist", dir=str(CUSTOMIZATION_PLIST.parent)
        )
        try:
            os.fchmod(fd, 0o644)
            with os.fdopen(fd, "wb") as handle:
                plistlib.dump(expected, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, CUSTOMIZATION_PLIST)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    domain = "gui/{}/ai.hermes.customizations".format(os.getuid())
    loaded = _run(["launchctl", "print", domain], cwd=HERMES_HOME, timeout=15)
    if loaded.returncode != 0:
        bootstrapped = _run(
            ["launchctl", "bootstrap", "gui/{}".format(os.getuid()), CUSTOMIZATION_PLIST],
            cwd=HERMES_HOME,
            timeout=30,
        )
        if bootstrapped.returncode != 0:
            return False, (bootstrapped.stderr or bootstrapped.stdout).strip()
        return True, "restored_and_loaded"
    return True, "refreshed" if changed else "current"


def _owned_audit(manifest, repo=REPO) -> list:
    return [str(relative) for relative in manifest.get("owned_source_files") or []
            if not (repo / str(relative)).is_file()]


def _ensure_gateway_service_definition():
    """Restore the guarded launchd plist if an updater rewrote it."""
    if os.environ.get("HERMES_COMPAT_STARTUP") == "1":
        return True, "startup wrapper already active"
    probe = _run([
        PYTHON, "-c",
        "from hermes_cli.gateway import launchd_plist_is_current; "
        "raise SystemExit(0 if launchd_plist_is_current() else 1)",
    ])
    if probe.returncode == 0:
        return True, "current"
    refresh = _run([PYTHON, "-m", "hermes_cli.main", "gateway", "start"], timeout=180)
    if refresh.returncode != 0:
        return False, (refresh.stderr or refresh.stdout).strip()
    return True, "refreshed"


def _verify(repo: Path):
    result = _run([PYTHON, VERIFY, "--repo", repo], cwd=repo, timeout=180)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, result.stdout.strip()


def _apply_direct():
    check = _git("apply", "--check", "--whitespace=nowarn", str(PATCH))
    if check.returncode != 0:
        return False, (check.stderr or check.stdout).strip()
    applied = _git("apply", "--whitespace=nowarn", str(PATCH))
    if applied.returncode != 0:
        return False, (applied.stderr or applied.stdout).strip()
    good, detail = _verify(REPO)
    if not good:
        _git("apply", "--reverse", "--whitespace=nowarn", str(PATCH))
        return False, "verification failed after direct apply: " + detail
    return True, detail


def _transactional_rebase(manifest):
    temporary = Path(tempfile.mkdtemp(prefix="hermes-overlay-rebase."))
    checkout, rebased = temporary / "checkout", temporary / "overlay.patch"
    added = False
    try:
        create = _git("worktree", "add", "--detach", str(checkout), "HEAD")
        if create.returncode != 0:
            return False, "worktree creation failed: " + (create.stderr or create.stdout).strip()
        added = True
        merged = _git("apply", "--3way", "--index", "--whitespace=nowarn", str(PATCH), cwd=checkout)
        if merged.returncode != 0:
            conflicts = _git("diff", "--name-only", "--diff-filter=U", cwd=checkout)
            detail = (merged.stderr or merged.stdout).strip()
            if conflicts.stdout.strip():
                detail += "\nConflicts: " + ", ".join(conflicts.stdout.splitlines())
            return False, detail
        good, detail = _verify(checkout)
        if not good:
            return False, "rebased overlay verification failed: " + detail
        generated = _git("diff", "--cached", "--binary", "--output={}".format(rebased), cwd=checkout)
        if generated.returncode != 0 or not rebased.is_file() or rebased.stat().st_size == 0:
            return False, "could not generate rebased overlay"
        live_check = _git("apply", "--check", "--whitespace=nowarn", str(rebased))
        if live_check.returncode != 0:
            return False, "rebased overlay failed live preflight: " + (live_check.stderr or live_check.stdout).strip()
        live_apply = _git("apply", "--whitespace=nowarn", str(rebased))
        if live_apply.returncode != 0:
            return False, "rebased overlay failed live apply: " + (live_apply.stderr or live_apply.stdout).strip()
        good, live_detail = _verify(REPO)
        if not good:
            _git("apply", "--reverse", "--whitespace=nowarn", str(rebased))
            return False, "live verification failed; overlay rolled back: " + live_detail
        os.replace(str(rebased), str(PATCH))
        CONFIG_SNAPSHOT.write_bytes(CONFIG.read_bytes())
        os.chmod(CONFIG_SNAPSHOT, 0o600)
        manifest.update({
            "schema_version": max(7, int(manifest.get("schema_version", 0))),
            "name": "Hermes Desktop model-agnostic frontier harness",
            "hermes_upstream_commit": _head(),
            "hermes_upstream_version": "auto-rebased",
            "overlay_sha256": _sha256(PATCH),
            "config_snapshot_sha256": _sha256(CONFIG_SNAPSHOT),
        })
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return True, live_detail
    finally:
        if added:
            _git("worktree", "remove", "--force", str(checkout))
        shutil.rmtree(temporary, ignore_errors=True)


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
            return _attention("Hermes Desktop Git checkout is missing")
        if not PATCH.is_file() or PATCH.stat().st_size == 0:
            return _attention("Desktop overlay patch is missing or empty")
        try:
            manifest = _read_manifest()
        except RuntimeError as exc:
            return _attention(str(exc))
        issues = _bundle_integrity(manifest)
        if issues:
            return _attention("Customization bundle integrity check failed", "; ".join(issues))
        missing_config = _config_audit()
        if missing_config:
            return _attention("Model-agnostic Hermes config invariants changed",
                              "Missing markers: " + ", ".join(missing_config))
        missing_assets = _external_audit()
        if missing_assets:
            return _attention("Required Desktop assets are missing", "; ".join(missing_assets))
        reverse = _git("apply", "--reverse", "--check", str(PATCH))
        if reverse.returncode == 0:
            good, detail = _verify(REPO)
            if not good:
                return _attention("Installed Desktop overlay failed verification", detail)
            result = "already_applied"
        else:
            good, detail = _apply_direct()
            if good:
                result = "applied"
            else:
                good, detail = _transactional_rebase(manifest)
                if not good:
                    return _attention("Desktop overlay could not be safely rebased after a Hermes update", detail)
                result = "rebased_applied"

        missing_owned = _owned_audit(manifest)
        if missing_owned:
            return _attention("Owned Desktop customization files are missing", "; ".join(missing_owned))
        custom_service_ok, custom_service_detail = _ensure_customization_service_definition()
        if not custom_service_ok:
            return _attention(
                "Hermes customization reconciliation service could not be restored",
                custom_service_detail,
            )
        service_ok, service_detail = _ensure_gateway_service_definition()
        if not service_ok:
            return _attention("Guarded Hermes Desktop launchd service could not be restored", service_detail)
        try:
            ATTENTION.unlink()
        except FileNotFoundError:
            pass
        _write_state(
            result, config_audit="passed", verification="passed",
            gateway_service=service_detail,
            customization_service=custom_service_detail,
        )
        if result != "already_applied":
            _log("Desktop overlay {} at HEAD {}".format(result, _head()[:12]))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
