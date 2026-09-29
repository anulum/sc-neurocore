# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia indexed SHD recording parity

"""Read actual SHD HDF5 rows through the public Julia API and CLI."""

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
from tests.test_accel_go_shd_abi import recording_file as recording_file


@pytest.mark.parametrize("channel", ["1.11", "release"])
@pytest.mark.parametrize("index,budget", [(0, 64), (1, 0)])
def test_julia_shd_direct_cli_preserves_selected_row_and_float16_widening(
    recording_file: Path, channel: str, index: int, budget: int
) -> None:
    """Each installed Julia runtime returns exact recorded values and integer labels."""
    kernel = (
        Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/shd_cli.jl"
    )
    result = subprocess.run(
        [
            "julia",
            f"+{channel}",
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            str(kernel),
            str(recording_file),
            str(index),
            str(budget),
            "libhdf5_serial.so",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    magic, count, label = struct.unpack_from("<4sQq", result.stdout)
    assert magic == b"SHD1" and len(result.stdout) == 20 + count * 8
    actual = np.frombuffer(result.stdout, dtype=np.float64, offset=20).reshape(-1, 4)
    expected, expected_label = read_shd_recording(recording_file, index, backend="numpy")
    np.testing.assert_array_equal(actual, expected)
    assert label == expected_label


@pytest.mark.parametrize("index,budget", [(-1, 64), (2, 64), (0, -1), (0, 63)])
def test_julia_shd_cli_refuses_index_and_budget_without_partial_output(
    recording_file: Path, index: int, budget: int
) -> None:
    """Bad recording indices and matrix budgets fail before binary output starts."""
    kernel = (
        Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/shd_cli.jl"
    )
    result = subprocess.run(
        [
            "julia",
            "+1.11",
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            str(kernel),
            str(recording_file),
            str(index),
            str(budget),
            "libhdf5_serial.so",
        ],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 1 and result.stdout == b""


@pytest.mark.parametrize("damage", ["shape", "pair", "label", "overflow", "missing"])
def test_julia_shd_cli_refuses_real_hdf5_format_damage(recording_file: Path, damage: str) -> None:
    """Pair lengths, recording counts and int64 label representation agree with other readers."""
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
    kernel = (
        Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/shd_cli.jl"
    )
    result = subprocess.run(
        [
            "julia",
            "+1.11",
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            str(kernel),
            str(recording_file),
            "0",
            "64",
            "libhdf5_serial.so",
        ],
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 1 and result.stdout == b""


def test_public_julia_shd_return_type_is_inferred(recording_file: Path) -> None:
    """The real public Julia file reader retains a concrete event/label return type."""
    kernel = Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/julia/datasets/shd.jl"
    subprocess.run(
        [
            "julia",
            "+1.11",
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            "-e",
            "using Test; include(ARGS[1]); sample=@inferred SHDRecordings.read_shd_recording(ARGS[2],0;maximum_bytes=64); @test sample.label==3; @test length(sample.events)==8",
            str(kernel),
            str(recording_file),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )


@pytest.fixture(scope="module")
def julia_shd_executable() -> Path:
    """Resolve an already-installed Julia executable, avoiding Juliaup in runtime reads."""
    path = subprocess.check_output(
        [
            "julia",
            "+1.11",
            "--startup-file=no",
            "-e",
            "print(joinpath(Sys.BINDIR, Base.julia_exename()))",
        ],
        text=True,
        timeout=20,
    )
    executable = Path(path).resolve()
    assert executable.is_absolute() and executable.is_file()
    return executable


@pytest.fixture
def selected_julia_shd(julia_shd_executable: Path) -> Iterator[None]:
    """Set and restore the actual operator-owned standalone runtime setting."""
    name = "SC_NEUROCORE_SHD_JULIA_EXE"
    previous = os.environ.get(name)
    os.environ[name] = str(julia_shd_executable)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.mark.parametrize("index", [0, 1])
def test_public_julia_selection_reaches_manifest_reader(
    selected_julia_shd: None, recording_file: Path, index: int
) -> None:
    """The public selector returns exact values and routes its actual label to the sample reader."""
    expected, label = read_shd_recording(recording_file, index, backend="numpy")
    actual, actual_label = read_shd_recording(recording_file, index, backend="julia")
    np.testing.assert_array_equal(actual, expected)
    assert actual_label == label
    sample = SampleRecord("train", recording_file.name, index, label, "speaker")
    np.testing.assert_array_equal(read_event_sample(recording_file.parent, "shd", sample), expected)


def test_selected_julia_invalid_hdf5_library_refuses_without_fallback(
    selected_julia_shd: None, recording_file: Path, tmp_path: Path
) -> None:
    """An existing incompatible native library fails a selected runtime instead of decoding in Python."""
    invalid = tmp_path / "unreadable.so"
    invalid.write_bytes(b"not a native library")
    name = "SC_NEUROCORE_SHD_JULIA_HDF5_LIBRARY"
    previous = os.environ.get(name)
    os.environ[name] = str(invalid)
    try:
        with pytest.raises(RuntimeError, match="Julia SHD reader failed or refused"):
            read_shd_recording(recording_file, 0, backend="julia")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def test_public_julia_blocked_recording_times_out_and_reaps_reader(
    selected_julia_shd: None, tmp_path: Path
) -> None:
    """A real blocked HDF5 open has a bounded lifetime and leaves no child process."""
    recording = tmp_path / "blocked.h5"
    os.mkfifo(recording)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    before = set(children.read_text().split())
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="lifetime|failed or refused"):
        read_shd_recording(recording, 0, backend="julia")
    elapsed = time.monotonic() - started
    assert 28 <= elapsed <= 40, elapsed
    assert set(children.read_text().split()) == before
