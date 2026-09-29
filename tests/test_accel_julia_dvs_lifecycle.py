# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia DVS native process lifecycle acceptance

"""Exercise actual Julia parent/group death and independent blocked I/O deadlines."""

import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from tests.shd_process_lifecycle_support import check_reader_shutdown
from tests.test_accel_julia_dvs_backends import julia_dvs_executable as julia_dvs_executable


def _command(executable: Path, recording: Path, maximum: int = 64) -> list[str]:
    """Build the actual runtime and packaged CLI arguments before the expected-parent field."""
    return [
        str(executable),
        "--startup-file=no",
        "--threads=1",
        "--check-bounds=yes",
        "--depwarn=error",
        str(
            Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/dvs_cli.jl"
        ),
        str(recording),
        str(maximum),
    ]


@pytest.mark.parametrize("group", [False, True])
def test_julia_dvs_actual_parent_or_group_death_reaps_blocked_reader(
    julia_dvs_executable: Path, tmp_path: Path, group: bool
) -> None:
    """A contained subreaper observes blocked FIFO I/O before killing and reaping its producer."""
    recording = tmp_path / "parent-death.npy"
    os.mkfifo(recording)
    check_reader_shutdown(_command(julia_dvs_executable, recording), group=group)


@pytest.mark.parametrize("parent", ["0", "-1", "invalid", "999999999999999999999999", "1"])
def test_julia_dvs_invalid_parent_refuses_before_blocked_input(
    julia_dvs_executable: Path, tmp_path: Path, parent: str
) -> None:
    """An invalid or stale parent is refused before a real FIFO can block the input."""
    recording = tmp_path / "invalid-parent.npy"
    os.mkfifo(recording)
    result = subprocess.run(
        _command(julia_dvs_executable, recording) + [parent],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 1, result.stderr.decode()
    assert result.stdout == b"" and result.stderr


def test_julia_dvs_own_alarm_terminates_blocked_input(
    julia_dvs_executable: Path, tmp_path: Path
) -> None:
    """The native timer independently ends FIFO input while the creating parent remains alive."""
    recording = tmp_path / "alarm.npy"
    os.mkfifo(recording)
    started = time.monotonic()
    result = subprocess.run(
        _command(julia_dvs_executable, recording) + [str(os.getpid())],
        capture_output=True,
        timeout=40,
    )
    assert result.returncode == -signal.SIGALRM, result.stderr.decode()
    assert result.stdout == b""
    assert 29 <= time.monotonic() - started < 40


def test_julia_dvs_own_alarm_terminates_blocked_output(
    julia_dvs_executable: Path, tmp_path: Path
) -> None:
    """An undrained binary pipe cannot retain the native reader beyond its own alarm."""
    recording = tmp_path / "blocked-output.npy"
    count = 8192
    np.save(recording, np.zeros((count, 4)))
    started = time.monotonic()
    process = subprocess.Popen(
        _command(julia_dvs_executable, recording, count * 32) + [str(os.getpid())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert process.wait(timeout=40) == -signal.SIGALRM
        elapsed = time.monotonic() - started
        output, error = process.communicate(timeout=5)
        assert 29 <= elapsed < 40
        assert output[:4] == b"DVS1" and len(output) < 12 + count * 32
        assert error == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
