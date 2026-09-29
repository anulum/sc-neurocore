# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Unix DVS filename bytes

"""Preserve opaque Unix recording filenames through all actual public decoders."""

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from tests.test_accel_go_dvs_recordings import go_dvs_executable as go_dvs_executable
from tests.test_accel_julia_dvs_backends import julia_dvs_executable as julia_dvs_executable
from tests.test_accel_mojo_dvs import mojo_dvs_executable as mojo_dvs_executable
from tests.test_accel_rust_dvs import rust_dvs_executable as rust_dvs_executable


def _recording(tmp_path: Path) -> tuple[Path, np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    """Write an actual big-endian Fortran recording under an invalid-UTF8 Unix name."""
    path = tmp_path / os.fsdecode(b"\xff-\xfe-events.npy")
    values = np.array([[1, 2, -1, 1.002], [3, 4, 1, 2.004]], dtype=">f8", order="F")
    np.save(path, values)
    return path, np.array(values, dtype=np.float64)


@pytest.mark.parametrize("backend", ["numpy", "go", "rust", "mojo"])
def test_public_reader_preserves_opaque_unix_filename_and_storage_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    go_dvs_executable: Path,
    rust_dvs_executable: Path,
    mojo_dvs_executable: Path,
    backend: Literal["numpy", "go", "rust", "mojo"],
) -> None:
    """Each real compiled public path reads the original filename bytes without recoding."""
    path, expected = _recording(tmp_path)
    for kind, executable in (
        ("go", go_dvs_executable),
        ("rust", rust_dvs_executable),
        ("mojo", mojo_dvs_executable),
    ):
        monkeypatch.setenv(f"SC_NEUROCORE_DVS_{kind.upper()}_EXE", str(executable))
    actual = read_dvs_recording(path, backend=backend)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert actual.flags.owndata and actual.flags.writeable and actual.flags.c_contiguous


def test_public_installed_julia_runtime_preserves_opaque_unix_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, julia_dvs_executable: Path
) -> None:
    """Both actual installed Julia runtimes preserve bytes through the packaged native script."""
    path, expected = _recording(tmp_path)
    monkeypatch.setenv("SC_NEUROCORE_DVS_JULIA_EXE", str(julia_dvs_executable))
    actual = read_dvs_recording(path, backend="julia")
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert actual.flags.owndata and actual.flags.writeable and actual.flags.c_contiguous
