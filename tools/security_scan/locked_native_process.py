# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native dependency scanner process custody

"""Retain scanner output and stop the entire process group on timeout."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess


def execute_command(
    command: list[str],
    cwd: Path,
    directory: Path,
    *,
    json_output: bool = True,
    timeout: float = 360,
) -> tuple[int, object]:
    """Run one native command with retained output and a bounded lifetime.

    Parameters
    ----------
    command : list of str
        Exact executable and argument vector.
    cwd : Path
        Working directory of the native process.
    directory : Path
        New directory for command, output and process receipts.
    json_output : bool, optional
        Decode stdout as JSON; installation commands return None instead.
    timeout : float, optional
        Positive execution budget, in seconds.

    Returns
    -------
    tuple of int and object
        Native exit status and decoded JSON, or None for non-JSON commands.

    Raises
    ------
    OSError, ValueError, subprocess.SubprocessError
        Process creation, output decoding or bounded completion failed. Original
        output bytes remain in the packet when decoding fails. A failed spawn
        records the OS exception and empty native streams before propagating it.
    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 360:
        raise ValueError("Native audit command requires a positive timeout.")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "command.json").write_text(json.dumps(command) + "\n", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        (directory / "stdout.json").write_bytes(b"")
        (directory / "stderr.log").write_bytes(b"")
        (directory / "receipt.json").write_text(
            json.dumps(
                {
                    "returncode": None,
                    "timed_out": False,
                    "process_reaped": False,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    (directory / "stdout.json").write_bytes(stdout)
    (directory / "stderr.log").write_bytes(stderr)
    (directory / "receipt.json").write_text(
        json.dumps(
            {
                "returncode": process.returncode,
                "timed_out": timed_out,
                "process_reaped": process.poll() is not None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if timed_out:
        raise ValueError("Dependency audit timed out; its process group was stopped.")
    return process.returncode, json.loads(stdout.decode("utf-8")) if json_output else None
