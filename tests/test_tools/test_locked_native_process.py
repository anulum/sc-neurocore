# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual scanner subprocess output and lifetime contracts

"""Exercise native process receipts without replacing command execution."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tools.security_scan.locked_native_process import execute_command


def test_real_command_output_and_nonzero_status_are_retained(tmp_path: Path) -> None:
    """Keep actual JSON, diagnostics and exit status without converting failure to success.

    Parameters
    ----------
    tmp_path : Path
        Owned directories for actual interpreter command packets.
    """
    command = [
        sys.executable,
        "-c",
        "import sys; print('{}'); print('diagnostic', file=sys.stderr); sys.exit(7)",
    ]
    rc, payload = execute_command(command, tmp_path, tmp_path / "packet")
    assert rc == 7 and payload == {}
    assert (tmp_path / "packet/stderr.log").read_text() == "diagnostic\n"
    assert json.loads((tmp_path / "packet/receipt.json").read_text()) == {
        "returncode": 7,
        "timed_out": False,
        "process_reaped": True,
    }
    rc, payload = execute_command(
        [sys.executable, "-c", "print('installed')"],
        tmp_path,
        tmp_path / "install",
        json_output=False,
    )
    assert rc == 0 and payload is None


def test_real_invalid_json_keeps_the_completed_process_receipt(tmp_path: Path) -> None:
    """Decode failure must retain genuine output and completed-process evidence.

    Parameters
    ----------
    tmp_path : Path
        Fresh packet location for actual non-JSON interpreter output.
    """
    with pytest.raises(json.JSONDecodeError):
        execute_command([sys.executable, "-c", "print('not-json')"], tmp_path, tmp_path / "packet")
    assert json.loads((tmp_path / "packet/receipt.json").read_text())["process_reaped"] is True
    assert (tmp_path / "packet/stdout.json").read_text() == "not-json\n"


def test_real_timed_out_command_is_stopped_and_reaped(tmp_path: Path) -> None:
    """A real sleeping command cannot survive the public execution budget.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for process timeout and reaping receipts.
    """
    with pytest.raises(ValueError, match="timed out"):
        execute_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            tmp_path / "packet",
            timeout=0.2,
        )
    receipt = json.loads((tmp_path / "packet/receipt.json").read_text())
    assert receipt["timed_out"] is True and receipt["process_reaped"] is True
    assert receipt["returncode"] != 0
    with pytest.raises(ValueError, match="positive timeout"):
        execute_command(
            [sys.executable, "-c", "print('{}')"], tmp_path, tmp_path / "invalid", timeout=0
        )
    assert not (tmp_path / "invalid").exists()


def test_real_signal_resistant_command_is_forcibly_reaped(tmp_path: Path) -> None:
    """Escalate to SIGKILL when the actual command ignores the graceful stop.

    Parameters
    ----------
    tmp_path : Path
        Owned directory for the resistant process and its retained receipt.
    """
    script = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    with pytest.raises(ValueError, match="timed out"):
        execute_command([sys.executable, "-c", script], tmp_path, tmp_path / "packet", timeout=1)
    receipt = json.loads((tmp_path / "packet/receipt.json").read_text())
    assert receipt == {"returncode": -9, "timed_out": True, "process_reaped": True}
    assert (tmp_path / "packet/stdout.json").read_text() == "ready\n"
