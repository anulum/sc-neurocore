# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia SHD native process lifetime acceptance

"""Exercise real Julia CLI parent-death and independent deadline behavior."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

import h5py
import numpy as np

from tests.shd_process_lifecycle_support import check_reader_shutdown
from tests.test_accel_julia_shd import julia_shd_executable as julia_shd_executable
from tests.test_accel_julia_dvs_backends import julia_dvs_executable as julia_dvs_executable
from tests.test_accel_go_shd_abi import recording_file as recording_file


@pytest.mark.parametrize("group", [False, True])
def test_julia_blocked_reader_dies_with_actual_parent(
    julia_shd_executable: Path, tmp_path: Path, group: bool
) -> None:
    """A contained subreaper reaps the real reader after parent or process-group death."""
    recording = tmp_path / "parent-death.h5"
    os.mkfifo(recording)
    kernel = (
        Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/shd_cli.jl"
    )
    check_reader_shutdown(
        [
            str(julia_shd_executable),
            "--startup-file=no",
            "--threads=1",
            "--check-bounds=yes",
            "--depwarn=error",
            str(kernel),
            str(recording),
            "0",
            "64",
            "libhdf5_serial.so",
        ],
        group=group,
    )


@pytest.mark.parametrize("parent", ["current", "stale"])
def test_julia_cli_enforces_own_alarm_and_expected_parent(
    julia_shd_executable: Path, tmp_path: Path, parent: str
) -> None:
    """The actual CLI independently alarms a blocked read or refuses a stale parent."""
    recording = tmp_path / "alarm.h5"
    os.mkfifo(recording)
    kernel = (
        Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/shd_cli.jl"
    )
    expected_parent = str(os.getpid()) if parent == "current" else "0"
    started = time.monotonic()
    result = subprocess.run(
        [
            str(julia_shd_executable),
            "--startup-file=no",
            "--threads=1",
            str(kernel),
            str(recording),
            "0",
            "64",
            "libhdf5_serial.so",
            expected_parent,
        ],
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


def test_julia_shd_own_alarm_terminates_blocked_output(
    julia_dvs_executable: Path, recording_file: Path
) -> None:
    """Both installed runtimes terminate an undrained SHD pipe under the supervised timer."""
    count = 8192
    with h5py.File(recording_file, "a") as handle:
        handle["spikes/times"][0] = np.zeros(count, dtype=np.float16)
        handle["spikes/units"][0] = np.ones(count, dtype=np.int16)
    kernel = (
        Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/shd_cli.jl"
    )
    started = time.monotonic()
    process = subprocess.Popen(
        [
            str(julia_dvs_executable),
            "--startup-file=no",
            "--threads=1",
            "--check-bounds=yes",
            "--depwarn=error",
            str(kernel),
            str(recording_file),
            "0",
            str(count * 32),
            "libhdf5_serial.so",
            str(os.getpid()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert process.wait(timeout=40) == -signal.SIGALRM
        elapsed = time.monotonic() - started
        output, error = process.communicate(timeout=5)
        assert 29 <= elapsed < 40
        assert output[:4] == b"SHD1" and len(output) < 20 + count * 32
        assert error == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
