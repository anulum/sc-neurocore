# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SHD native lifecycle acceptance

"""Observe actual compiled Mojo CLI parent death, group cancellation and native alarm."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.shd_process_lifecycle_support import check_reader_shutdown
from tests.test_accel_mojo_shd import mojo_shd_executable as mojo_shd_executable


@pytest.mark.parametrize("group", [False, True])
def test_mojo_blocked_reader_dies_with_parent_or_group(
    mojo_shd_executable: Path, tmp_path: Path, group: bool
) -> None:
    """The actual blocked reader is reaped after its producer dies or its group is cancelled."""
    recording = tmp_path / "parent-death.h5"
    os.mkfifo(recording)
    check_reader_shutdown(
        [str(mojo_shd_executable), str(recording), "0", "64", "libhdf5_serial.so"], group=group
    )


@pytest.mark.parametrize("parent", ["current", "stale"])
def test_mojo_cli_enforces_native_alarm_and_expected_parent(
    mojo_shd_executable: Path, tmp_path: Path, parent: str
) -> None:
    """A blocked read independently gets SIGALRM while a stale parent refuses before reading."""
    recording = tmp_path / "alarm.h5"
    os.mkfifo(recording)
    expected_parent = str(os.getpid()) if parent == "current" else "0"
    started = time.monotonic()
    result = subprocess.run(
        [str(mojo_shd_executable), str(recording), "0", "64", "libhdf5_serial.so", expected_parent],
        capture_output=True,
        timeout=40,
    )
    assert result.stdout == b""
    if parent == "current":
        assert result.returncode == -signal.SIGALRM, result.stderr.decode()
        assert 28 <= time.monotonic() - started <= 40
    else:
        assert result.returncode == 124, result.stderr.decode()
        assert time.monotonic() - started < 10
