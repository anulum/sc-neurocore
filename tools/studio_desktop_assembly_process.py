# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded desktop assembly command custody

"""Retain real command output and reap each assembly process group before returning."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from sc_neurocore.studio.platform.jobs_reaper import reap_process_group


def run_assembly_command(
    argv: Sequence[str],
    timeout_seconds: float,
    receipts: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run one bounded command with original output and verified group cleanup.

    Parameters
    ----------
    argv : sequence of str
        Exact executable and arguments, without a shell.
    timeout_seconds : float
        Finite positive execution deadline. Cleanup uses the existing Studio
        process custodian's separately defined termination and kill grace periods.
    receipts : pathlib.Path
        New directory for original command, stdout, stderr and terminal receipt.
    environment : mapping of str to str or None
        Complete child environment, or None to inherit the caller's environment.
        Values are passed directly to the child and never copied into receipts.

    Returns
    -------
    subprocess.CompletedProcess
        Actual collected exit status and original byte streams after group cleanup.

    Raises
    ------
    ValueError
        Arguments or execution deadline are invalid.
    subprocess.SubprocessError
        The command fails or times out; original output and cleanup are retained.
    RuntimeError
        The scoped process custodian cannot verify complete group cleanup.
    OSError
        Creating receipts or starting the actual executable fails.
    """
    if not argv or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Assembly commands need arguments and a finite positive deadline.")
    receipts.mkdir(parents=True)
    (receipts / "command.json").write_text(json.dumps(list(argv), indent=2) + "\n")
    timed_out = False
    with (
        (receipts / "stdout").open("xb") as output,
        (receipts / "stderr").open("xb") as errors,
    ):
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=errors,
            start_new_session=True,
            env=environment,
        )
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            cleanup = reap_process_group(process, owned_group_id=process.pid)
    stdout = (receipts / "stdout").read_bytes()
    stderr = (receipts / "stderr").read_bytes()
    record = {"timed_out": timed_out, "cleanup": cleanup.to_public_dict()}
    (receipts / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    if not cleanup.reaped or cleanup.returncode is None:
        raise RuntimeError("The assembly command group could not be fully reaped.")
    if timed_out:
        raise subprocess.TimeoutExpired(list(argv), timeout_seconds, output=stdout, stderr=stderr)
    result = subprocess.CompletedProcess(list(argv), cleanup.returncode, stdout, stderr)
    result.check_returncode()
    return result
