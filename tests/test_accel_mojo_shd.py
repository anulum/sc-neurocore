# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo indexed SHD recording parity

"""Compare the public Mojo SHD API with actual HDF5 recordings."""

import os
import struct
import subprocess
import time
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


@pytest.fixture(scope="module")
def mojo_shd_reader(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a native caller of the public Mojo reader with exact binary value transport."""
    root = Path(__file__).resolve().parents[1]
    directory = tmp_path_factory.mktemp("mojo-shd")
    source = directory / "reader.mojo"
    source.write_text(
        "\n".join(
            (root / "src/sc_neurocore/accel/mojo/kernels/shd.mojo").read_text().splitlines()[:7]
        )
        + "\n\n"
        + _MOJO_CALLER
    )
    executable = directory / "reader"
    subprocess.run(
        [
            "mojo",
            "build",
            "--Werror",
            "--fp-mode",
            "contract=off",
            "-I",
            str(root / "src/sc_neurocore/accel/mojo/kernels"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return executable


_MOJO_CALLER = """
from shd import read_shd_recording
from std.ffi import external_call
from std.memory import Pointer
from std.sys import argv
from std.sys.terminate import exit


def run() raises:
    var args = argv()
    var result = read_shd_recording(args[1], Int(args[2]), Int(args[3]))
    var label = result[1]
    if external_call["write", Int](Int(1), Pointer(to=label), Int(8)) != 8:
        raise Error("label write failed")
    for value in result[0]:
        if external_call["write", Int](Int(1), Pointer(to=value), Int(8)) != 8:
            raise Error("event write failed")


def main():
    try:
        run()
    except:
        exit(1)
"""


@pytest.mark.parametrize("index,budget", [(0, 64), (1, 0)])
def test_mojo_shd_public_api_preserves_exact_row_and_label(
    mojo_shd_reader: Path, recording_file: Path, index: int, budget: int
) -> None:
    """Selected nonempty and empty rows retain widened double values and integer labels."""
    result = subprocess.run(
        [str(mojo_shd_reader), str(recording_file), str(index), str(budget)],
        capture_output=True,
        check=True,
        timeout=20,
    )
    label = struct.unpack_from("<q", result.stdout)[0]
    actual = np.frombuffer(result.stdout, dtype=np.float64, offset=8).reshape(-1, 4)
    expected, expected_label = read_shd_recording(recording_file, index, backend="numpy")
    np.testing.assert_array_equal(actual, expected)
    assert label == expected_label


@pytest.mark.parametrize("index,budget", [(-1, 64), (2, 64), (0, -1), (0, 63)])
def test_mojo_shd_public_api_refuses_index_and_budget(
    mojo_shd_reader: Path, recording_file: Path, index: int, budget: int
) -> None:
    """Native public index/budget refusals produce no partial successful recording."""
    result = subprocess.run(
        [str(mojo_shd_reader), str(recording_file), str(index), str(budget)],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize("damage", ["shape", "pair", "label", "overflow", "missing"])
def test_mojo_shd_public_api_refuses_real_recording_damage(
    mojo_shd_reader: Path, recording_file: Path, damage: str
) -> None:
    """Native HDF5 reader refuses mismatched vectors/counts and unrepresentable labels."""
    if damage == "missing":
        recording_file = recording_file.with_name("absent.h5")
    else:
        with h5py.File(recording_file, "a") as handle:
            if damage == "pair":
                handle["spikes/units"][0] = np.array([1], dtype=np.int16)
            else:
                del handle["labels"]
                if damage == "shape":
                    handle["labels"] = np.array([3], dtype=np.uint8)
                elif damage == "overflow":
                    handle["labels"] = np.array([1 << 63, 17], dtype=np.uint64)
                else:
                    handle["labels"] = np.array([3, 17], dtype=np.float32)
    result = subprocess.run(
        [str(mojo_shd_reader), str(recording_file), "0", "64"],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode != 0 and result.stdout == b""


@pytest.fixture(scope="module")
def mojo_shd_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the real supervised SHD CLI as an operator-owned executable."""
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path_factory.mktemp("mojo-shd-cli") / "reader"
    subprocess.run(
        [
            "mojo",
            "build",
            "--Werror",
            "--fp-mode",
            "contract=off",
            str(root / "src/sc_neurocore/accel/mojo/kernels/shd_cli.mojo"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return executable


@pytest.fixture
def selected_mojo_shd(mojo_shd_executable: Path) -> Iterator[None]:
    """Select and restore the actual compiled operator executable."""
    name = "SC_NEUROCORE_SHD_MOJO_EXE"
    previous = os.environ.get(name)
    os.environ[name] = str(mojo_shd_executable)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.mark.parametrize("index", [0, 1])
def test_public_mojo_selection_reaches_manifest_reader(
    selected_mojo_shd: None, recording_file: Path, index: int
) -> None:
    """Actual compiled CLI values and labels reach the public SHD manifest sampler."""
    expected, label = read_shd_recording(recording_file, index, backend="numpy")
    actual, actual_label = read_shd_recording(recording_file, index, backend="mojo")
    np.testing.assert_array_equal(actual, expected)
    assert actual_label == label
    sample = SampleRecord("train", recording_file.name, index, label, "speaker")
    np.testing.assert_array_equal(read_event_sample(recording_file.parent, "shd", sample), expected)


def test_selected_mojo_invalid_hdf5_library_refuses_without_fallback(
    selected_mojo_shd: None, recording_file: Path, tmp_path: Path
) -> None:
    """A declared incompatible system library fails the native read without substitution."""
    name = "SC_NEUROCORE_SHD_MOJO_HDF5_LIBRARY"
    invalid = tmp_path / "invalid.so"
    invalid.write_bytes(b"not a native library")
    previous = os.environ.get(name)
    os.environ[name] = str(invalid)
    try:
        with pytest.raises(RuntimeError, match="Mojo SHD reader failed or refused"):
            read_shd_recording(recording_file, 0, backend="mojo")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def test_mojo_operator_environment_keeps_shd_separate_from_nmnist(
    mojo_shd_executable: Path, recording_file: Path
) -> None:
    """Only the declared SHD executable/library enter the worker environment."""
    library = Path("/lib/x86_64-linux-gnu/libhdf5_serial.so").resolve()
    configuration = EventWorkerConfiguration(
        dataset_root=recording_file.parent,
        shd_mojo_executable=mojo_shd_executable,
        shd_mojo_hdf5_library=library,
    )
    environment = configuration.environment()
    assert environment["SC_NEUROCORE_SHD_MOJO_EXE"] == str(mojo_shd_executable)
    assert environment["SC_NEUROCORE_SHD_MOJO_HDF5_LIBRARY"] == str(library)
    assert "SC_NEUROCORE_DATASET_MOJO_LIBRARY" not in environment


def test_public_mojo_blocked_recording_times_out_and_reaps_reader(
    selected_mojo_shd: None, tmp_path: Path
) -> None:
    """A real blocked HDF5 FIFO read has finite lifetime and leaves no child behind."""
    recording = tmp_path / "blocked.h5"
    os.mkfifo(recording)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    before = set(children.read_text().split())
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="lifetime|failed or refused"):
        read_shd_recording(recording, 0, backend="mojo")
    assert 28 <= time.monotonic() - started <= 40
    assert set(children.read_text().split()) == before
