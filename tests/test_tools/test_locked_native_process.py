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


@pytest.mark.parametrize(
    "payload", [b"not-json\r\n", b'{"value":"\xff"}\r\n', b"\xef\xbb\xbf{}", "{}".encode("utf-16")]
)
def test_real_invalid_json_keeps_the_completed_process_receipt(
    tmp_path: Path, payload: bytes
) -> None:
    """Decode failure must retain genuine output and completed-process evidence.

    Parameters
    ----------
    tmp_path : Path
        Fresh packet location for actual non-JSON interpreter output.
    payload : bytes
        Actual child output with invalid JSON, invalid UTF-8, a BOM or UTF-16.
    """
    command = [
        sys.executable,
        "-c",
        f"import os; os.write(1,{payload!r}); os.write(2,b'diagnostic\\r\\n')",
    ]
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        execute_command(command, tmp_path, tmp_path / "packet")
    assert json.loads((tmp_path / "packet/receipt.json").read_text())["process_reaped"] is True
    assert (tmp_path / "packet/stdout.json").read_bytes() == payload
    assert (tmp_path / "packet/stderr.log").read_bytes() == b"diagnostic\r\n"


@pytest.mark.parametrize("code", [0, 7])
def test_real_binary_streams_and_exit_status_are_preserved(tmp_path: Path, code: int) -> None:
    """Opaque command output retains invalid UTF-8, CRLF and NUL bytes on either exit.

    Parameters
    ----------
    tmp_path : Path
        Owned directory for the real process packet.
    code : int
        Successful or nonzero native exit status, returned without reinterpretation.
    """
    stdout, stderr = b"out\xff\r\n\0", b"err\xfe\r\n\0"
    command = [
        sys.executable,
        "-c",
        f"import os,sys; os.write(1,{stdout!r}); os.write(2,{stderr!r}); sys.exit({code})",
    ]
    packet = tmp_path / "packet"
    rc, payload = execute_command(command, tmp_path, packet, json_output=False)
    assert rc == code and payload is None
    assert (packet / "stdout.json").read_bytes() == stdout
    assert (packet / "stderr.log").read_bytes() == stderr
    assert json.loads((packet / "receipt.json").read_bytes()) == {
        "returncode": code,
        "timed_out": False,
        "process_reaped": True,
    }


def test_real_failed_spawn_keeps_the_command_and_actual_os_exception(tmp_path: Path) -> None:
    """An absent executable records its real OS error without inventing native output.

    Parameters
    ----------
    tmp_path : Path
        Owned directory with an absent executable and a fresh packet.
    """
    command = [str(tmp_path / "absent-executable")]
    packet = tmp_path / "packet"
    with pytest.raises(FileNotFoundError) as caught:
        execute_command(command, tmp_path, packet)
    assert json.loads((packet / "command.json").read_bytes()) == command
    assert (packet / "stdout.json").read_bytes() == b""
    assert (packet / "stderr.log").read_bytes() == b""
    assert json.loads((packet / "receipt.json").read_bytes()) == {
        "returncode": None,
        "timed_out": False,
        "process_reaped": False,
        "exception_type": "FileNotFoundError",
        "exception": str(caught.value),
    }


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
