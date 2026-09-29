# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust indexed SHD recording parity

"""Exercise the real Rust HDF5 reader, C ABI and public Python selection."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest

from sc_neurocore.accel.shd_recordings import read_shd_recording
from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import SampleRecord
from sc_neurocore.studio.platform.storage_event_worker_configuration import EventWorkerConfiguration
from tests.test_accel_go_shd_abi import recording_file as recording_file
from tests.test_accel_go_shd_abi import shd_library as shd_library


@pytest.fixture(scope="module")
def rust_shd_runtime(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Compile the actual cdylib and a Rust caller of the public safe reader API."""
    directory = tmp_path_factory.mktemp("rust-shd")
    source = Path(__file__).resolve().parents[1]
    crate = source / "src/sc_neurocore/accel/rust/safety/shd_native"
    subprocess.run(
        [
            "cargo",
            "build",
            "--offline",
            "--locked",
            "--release",
            "--manifest-path",
            str(crate / "Cargo.toml"),
            "--target-dir",
            str(directory),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    caller = directory / "caller.rs"
    header = "\n".join((crate / "src/lib.rs").read_text().splitlines()[:7])
    caller.write_text(
        header
        + "\n\n"
        + r"""use std::{io::{self, Write}, path::Path};
fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 { return Err(io::Error::other("path index budget required")); }
    let index = args[2].parse::<usize>().map_err(io::Error::other)?;
    let budget = args[3].parse::<usize>().map_err(io::Error::other)?;
    let row = sc_neurocore_shd::read_shd_recording(Path::new(&args[1]), index, budget)?;
    let mut output = io::stdout().lock();
    output.write_all(b"SHD1")?;
    output.write_all(&(row.events.len() as u64).to_le_bytes())?;
    output.write_all(&row.label.to_le_bytes())?;
    for value in row.events { output.write_all(&value.to_ne_bytes())?; }
    Ok(())
}
"""
    )
    flags = subprocess.check_output(["pkg-config", "--libs", "hdf5"], text=True).split()
    executable = directory / "read-shd"
    subprocess.run(
        [
            "rustc",
            "--edition=2021",
            "-Dwarnings",
            str(caller),
            "--extern",
            f"sc_neurocore_shd={directory / 'release/libsc_neurocore_shd.rlib'}",
            *flags,
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return directory / "release/libsc_neurocore_shd.so", executable


@pytest.fixture(scope="module")
def rust_shd_library(rust_shd_runtime: tuple[Path, Path]) -> Path:
    """Expose the actual built Rust C ABI artifact for worker integration tests."""
    return rust_shd_runtime[0]


@pytest.fixture
def native_rust_shd(rust_shd_library: Path) -> Iterator[None]:
    """Set and restore the operator's real Rust library selection."""
    name = "SC_NEUROCORE_SHD_RUST_LIBRARY"
    previous = os.environ.get(name)
    os.environ[name] = str(rust_shd_library)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.mark.parametrize("index", [0, 1])
def test_rust_public_reader_and_python_dispatch_preserve_exact_events(
    rust_shd_runtime: tuple[Path, Path], native_rust_shd: None, recording_file: Path, index: int
) -> None:
    """Safe Rust file API, exported ABI and manifest path agree with widened Python events."""
    result = subprocess.run(
        [str(rust_shd_runtime[1]), str(recording_file), str(index), "64"],
        check=True,
        capture_output=True,
        timeout=10,
    )
    magic, count, label = struct.unpack_from("<4sQq", result.stdout)
    assert magic == b"SHD1" and len(result.stdout) == 20 + count * 8
    rust_events = np.frombuffer(result.stdout, dtype=np.float64, offset=20).reshape(-1, 4)
    expected, expected_label = read_shd_recording(recording_file, index, backend="numpy")
    native, native_label = read_shd_recording(recording_file, index, backend="rust")
    np.testing.assert_array_equal(rust_events, expected)
    np.testing.assert_array_equal(native, expected)
    assert label == native_label == expected_label
    sample = SampleRecord("train", recording_file.name, index, label, "speaker")
    np.testing.assert_array_equal(read_event_sample(recording_file.parent, "shd", sample), expected)


@pytest.mark.parametrize(
    "case", ["read", "path", "output", "index", "overflow", "budget", "missing"]
)
def test_rust_c_abi_result_ownership_and_refusal(
    rust_shd_library: Path, recording_file: Path, case: str
) -> None:
    """Run the same real ownership/refusal protocol against the Rust allocation owner."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("test_accel_go_shd_abi.py")),
            str(rust_shd_library),
            str(recording_file),
            case,
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )


@pytest.mark.parametrize("damage", ["shape", "pair", "label", "label_overflow"])
def test_rust_reader_refuses_actual_hdf5_damage(
    native_rust_shd: None, recording_file: Path, damage: str
) -> None:
    """Native row shape, pair lengths and label representation are checked before publication."""
    with h5py.File(recording_file, "a") as handle:
        if damage == "pair":
            handle["spikes/units"][0] = np.array([1], dtype=np.int16)
        else:
            del handle["labels"]
            if damage == "shape":
                handle["labels"] = np.array([3], dtype=np.uint8)
            elif damage == "label_overflow":
                handle["labels"] = np.array([1 << 63, 17], dtype=np.uint64)
            else:
                handle["labels"] = np.array([3, 17], dtype=np.float32)
    with pytest.raises(RuntimeError, match="Rust SHD reader failed or refused"):
        read_shd_recording(recording_file, 0, backend="rust")


def test_operator_configuration_preserves_independent_rust_shd_setting(
    rust_shd_library: Path, recording_file: Path
) -> None:
    """The worker receives only the selected SHD Rust artifact, independent of N-MNIST."""
    configuration = EventWorkerConfiguration(
        dataset_root=recording_file.parent, shd_rust_library=rust_shd_library
    )
    environment = configuration.environment()
    assert environment["SC_NEUROCORE_SHD_RUST_LIBRARY"] == str(rust_shd_library)
    assert "SC_NEUROCORE_DATASET_RUST_LIBRARY" not in environment


def test_rust_go_and_python_read_actual_non_utf8_recording_name(
    native_rust_shd: None, recording_file: Path, shd_library: Path
) -> None:
    """The native transport preserves Unix filename bytes rather than forcing UTF-8."""
    renamed = recording_file.with_name(os.fsdecode(b"\xff-shd.h5"))
    recording_file.rename(renamed)
    expected, label = read_shd_recording(renamed, 0, backend="numpy")
    name = "SC_NEUROCORE_SHD_GO_LIBRARY"
    previous = os.environ.get(name)
    os.environ[name] = str(shd_library)
    try:
        for backend in ("rust", "go"):
            actual, native_label = read_shd_recording(renamed, 0, backend=backend)
            np.testing.assert_array_equal(actual, expected)
            assert native_label == label
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def test_rust_repeated_native_reads_release_hdf5_objects_and_file_descriptors(
    rust_shd_library: Path, recording_file: Path
) -> None:
    """Repeated real C calls in one process retain neither HDF5 objects nor OS descriptors."""
    source = (
        "\n".join(Path(__file__).read_text().splitlines()[:7])
        + "\n\n"
        + r"""import ctypes, os, sys
class Row(ctypes.Structure):
    'Match the public native SHD owning result layout.'

    _fields_ = [("events", ctypes.POINTER(ctypes.c_double)),
                ("value_count", ctypes.c_size_t), ("label", ctypes.c_int64)]
lib = ctypes.CDLL(sys.argv[1])
lib.shd_read_c.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(Row)]
lib.shd_read_c.restype = ctypes.c_int
lib.shd_free_c.argtypes = [ctypes.POINTER(Row)]
lib.shd_free_c.restype = None
lib.H5Fget_obj_count.argtypes = [ctypes.c_int64, ctypes.c_uint]
lib.H5Fget_obj_count.restype = ctypes.c_ssize_t
path = os.fsencode(sys.argv[2])
row = Row()
assert lib.shd_read_c(path, 0, 64, ctypes.byref(row)) == 0
lib.shd_free_c(ctypes.byref(row))
objects = lib.H5Fget_obj_count(31, 31)
assert objects >= 0
files = set(os.listdir("/proc/self/fd"))
for index, budget, expected in [(0, 64, 0), (1, 0, 0), (2, 64, -1), (0, 63, -1)] * 25:
    assert lib.shd_read_c(path, index, budget, ctypes.byref(row)) == expected
    lib.shd_free_c(ctypes.byref(row))
    assert lib.H5Fget_obj_count(31, 31) == objects
    assert set(os.listdir("/proc/self/fd")) == files
"""
    )
    subprocess.run(
        [sys.executable, "-c", source, str(rust_shd_library), str(recording_file)],
        check=True,
        capture_output=True,
        timeout=20,
    )
