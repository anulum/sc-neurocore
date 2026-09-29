# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go indexed HDF5 reader parity

"""Exercise the real HDF5 shared-library ABI and its result ownership."""

import ctypes
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


class Recording(ctypes.Structure):
    """C-owned event vector and label returned by the shared reader."""

    _fields_ = [
        ("events", ctypes.POINTER(ctypes.c_double)),
        ("value_count", ctypes.c_size_t),
        ("label", ctypes.c_int64),
    ]


@pytest.fixture(scope="module")
def shd_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile and bind the actual optional HDF5 C shared exports."""
    path = tmp_path_factory.mktemp("shd-abi") / "libloaders.so"
    subprocess.run(
        [
            "go",
            "build",
            "-tags=hdf5",
            "-buildmode=c-shared",
            "-o",
            str(path),
            "./services/loaders/cshared",
        ],
        cwd=Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


def _load_library(path: Path) -> ctypes.CDLL:
    """Bind the C reader in a fresh process without Julia library interposition."""
    library = ctypes.CDLL(str(path))
    library.shd_read_c.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(Recording),
    ]
    library.shd_read_c.restype = ctypes.c_int
    library.shd_free_c.argtypes = [ctypes.POINTER(Recording)]
    library.shd_free_c.restype = None
    return library


@pytest.fixture
def recording_file(tmp_path: Path) -> Path:
    """Write one float16 recording and one empty recording with integer labels."""
    path = tmp_path / "shd.h5"
    with h5py.File(path, "w") as handle:
        times = handle.create_dataset("spikes/times", (2,), dtype=h5py.vlen_dtype(np.float16))
        units = handle.create_dataset("spikes/units", (2,), dtype=h5py.vlen_dtype(np.int16))
        times[0] = np.array([0.000333, 0.9995], dtype=np.float16)
        units[0] = np.array([34, 699], dtype=np.int16)
        times[1] = np.array([], dtype=np.float16)
        units[1] = np.array([], dtype=np.int16)
        handle["labels"] = np.array([3, 17], dtype=np.uint8)
    return path


def _read_and_release(shd_library: ctypes.CDLL, recording_file: Path) -> None:
    """Read, reject overwriting ownership, release, then read an empty row."""
    record = Recording()
    path = str(recording_file).encode()
    try:
        assert shd_library.shd_read_c(path, 0, 64, ctypes.byref(record)) == 0
        values = np.ctypeslib.as_array(record.events, shape=(record.value_count,)).copy()
        expected = np.zeros((2, 4), dtype=np.float64)
        expected[:, 0] = [34, 699]
        expected[:, 3] = np.array([0.000333, 0.9995], dtype=np.float16).astype(np.float64) * 1000
        np.testing.assert_array_equal(values.reshape(-1, 4), expected)
        assert record.label == 3
        assert shd_library.shd_read_c(path, 1, 0, ctypes.byref(record)) == -1
        np.testing.assert_array_equal(
            np.ctypeslib.as_array(record.events, shape=(record.value_count,)), values
        )
        shd_library.shd_free_c(ctypes.byref(record))
        assert not record.events and record.value_count == 0 and record.label == 0
        shd_library.shd_free_c(ctypes.byref(record))
        assert shd_library.shd_read_c(path, 1, 0, ctypes.byref(record)) == 0
        assert not record.events and record.value_count == 0 and record.label == 17
    finally:
        shd_library.shd_free_c(ctypes.byref(record))
    shd_library.shd_free_c(None)


def _refusal(shd_library: ctypes.CDLL, recording_file: Path, case: str) -> None:
    """Null arguments, integer overflow, bad files and budgets refuse safely."""
    record = Recording()
    path = None if case == "path" else str(recording_file).encode()
    if case == "missing":
        path = str(recording_file.with_name("missing.h5")).encode()
    index = 2 if case == "index" else 0
    if case == "overflow":
        index = ctypes.c_size_t(-1).value
    budget = 63 if case == "budget" else 64
    output = None if case == "output" else ctypes.byref(record)
    assert shd_library.shd_read_c(path, index, budget, output) == -1
    assert not record.events and record.value_count == 0 and record.label == 0


def test_shared_shd_result_preserves_values_and_can_be_released_and_reused(
    shd_library: Path, recording_file: Path
) -> None:
    """Run real ownership checks in a bounded fresh process, as an isolated worker."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(shd_library),
            str(recording_file),
            "read",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.parametrize("case", ["path", "output", "index", "overflow", "budget", "missing"])
def test_shared_shd_refusal_does_not_allocate_or_publish_partial_output(
    shd_library: Path, recording_file: Path, case: str
) -> None:
    """Exercise each ABI refusal through the actual library in a clean worker process."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(shd_library),
            str(recording_file),
            case,
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )


if __name__ == "__main__":
    library = _load_library(Path(sys.argv[1]))
    recording = Path(sys.argv[2])
    if sys.argv[3] == "read":
        _read_and_release(library, recording)
    else:
        _refusal(library, recording, sys.argv[3])
