# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Go DVS process lifecycle

"""Exercise native parent/group death and independent deadlines on actual blocked file opens."""

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from tests.shd_process_lifecycle_support import check_reader_shutdown
from tests.test_accel_go_dvs_recordings import _native, go_dvs_executable as go_dvs_executable


@pytest.mark.parametrize("group", [False, True])
def test_native_go_dvs_blocked_reader_is_reaped_after_parent_or_group_death(
    tmp_path: Path, go_dvs_executable: Path, group: bool
) -> None:
    """A real blocked FIFO open cannot survive the contained producer's death."""
    path = tmp_path / "blocked.npy"
    os.mkfifo(path)
    check_reader_shutdown([str(go_dvs_executable), str(path), "64"], group=group)


def test_native_go_dvs_independent_deadline_interrupts_blocked_file_open(
    tmp_path: Path, go_dvs_executable: Path
) -> None:
    """The executable's own timer terminates blocked I/O while its parent remains alive."""
    path = tmp_path / "blocked.npy"
    os.mkfifo(path)
    started = time.monotonic()
    result = subprocess.run(
        [str(go_dvs_executable), str(path), "64", str(os.getpid())],
        capture_output=True,
        timeout=35,
    )
    assert result.returncode == 124 and result.stdout == b""
    assert 29 <= time.monotonic() - started < 35


@pytest.mark.parametrize("parent", ["0", "-1", "invalid", "999999999999999999999999", "1"])
def test_native_go_dvs_invalid_or_stale_parent_refuses_before_blocking(
    tmp_path: Path, go_dvs_executable: Path, parent: str
) -> None:
    """Parent declarations refuse before an actual FIFO open can block the reader."""
    path = tmp_path / "blocked.npy"
    os.mkfifo(path)
    result = subprocess.run(
        [str(go_dvs_executable), str(path), "64", parent], capture_output=True, timeout=5
    )
    assert result.returncode == 124 and result.stdout == b""


def test_native_go_dvs_explicit_parent_success_keeps_binary_contract(
    tmp_path: Path, go_dvs_executable: Path
) -> None:
    """Guarded success writes exactly the same complete DVS event frame."""
    path = tmp_path / "events.npy"
    values = np.arange(8, dtype=np.float64).reshape((2, 4))
    np.save(path, values)
    implicit = _native(go_dvs_executable, path)
    result = subprocess.run(
        [str(go_dvs_executable), str(path), "64", str(os.getpid())],
        capture_output=True,
        check=True,
        timeout=5,
    )
    assert result.stdout[:12] == b"DVS1\x08\x00\x00\x00\x00\x00\x00\x00"
    actual = np.frombuffer(result.stdout[12:], dtype="<f8").reshape((2, 4))
    np.testing.assert_array_equal(actual, implicit)


def test_native_go_dvs_independent_deadline_interrupts_blocked_output(
    tmp_path: Path, go_dvs_executable: Path
) -> None:
    """A full undrained output pipe cannot keep the guarded command alive past its deadline."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((8192, 4)))
    started = time.monotonic()
    process = subprocess.Popen(
        [str(go_dvs_executable), str(path), str(8192 * 32), str(os.getpid())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert process.wait(timeout=35) == 124
        elapsed = time.monotonic() - started
        output, error = process.communicate(timeout=5)
        assert 29 <= elapsed < 35
        assert output[:4] == b"DVS1" and len(output) < 12 + 8192 * 32
        assert error == b""
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
