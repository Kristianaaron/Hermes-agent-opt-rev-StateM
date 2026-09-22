#!/usr/bin/env python3
"""Fail-closed launchd entrypoint for the Hermes Desktop gateway."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HOME = Path.home()
HERMES = Path(os.environ.get("HERMES_HOME", str(HOME / ".hermes"))).expanduser()
AGENT = HERMES / "hermes-agent"
REAPPLY = HERMES / "customizations" / "frontier-harness" / "reapply.py"
PYTHON = AGENT / "venv" / "bin" / "python"


def main() -> int:
    gate_env = dict(os.environ)
    gate_env["HERMES_COMPAT_STARTUP"] = "1"
    checked = subprocess.run(
        ["/usr/bin/python3", str(REAPPLY)], timeout=240, env=gate_env
    )
    if checked.returncode != 0:
        print(
            "Hermes Desktop startup blocked: customization compatibility gate failed. "
            "See ~/.hermes/customizations/frontier-harness/NEEDS_ATTENTION.",
            file=sys.stderr,
        )
        return 78
    argv = [str(PYTHON), "-m", "hermes_cli.stderr_timestamp", "--error-log",
            str(HERMES / "logs" / "gateway.error.log"), "--", str(PYTHON),
            "-m", "hermes_cli.main", "gateway", "run", "--external-supervisor"]
    os.chdir(str(HERMES))
    os.execv(str(PYTHON), argv)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
