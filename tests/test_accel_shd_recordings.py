# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go indexed HDF5 reader parity

"""Exercise public indexed SHD reads and native worker separation on real files."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pytest

from sc_neurocore.accel.shd_recordings import read_shd_recording
from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import SampleRecord
from sc_neurocore.studio.platform.storage_event_worker_configuration import EventWorkerConfiguration
from tests.test_accel_go_shd_abi import recording_file as recording_file
from tests.test_accel_go_shd_abi import shd_library as shd_library


@pytest.fixture
def native_shd(shd_library: Path) -> Iterator[None]:
    """Select the compiled native library through the actual operator environment."""
    name = "SC_NEUROCORE_SHD_GO_LIBRARY"
    previous = os.environ.get(name)
    os.environ[name] = str(shd_library)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.mark.parametrize("index", [0, 1])
def test_public_native_and_numpy_rows_match_and_reach_manifest_reader(
    native_shd: None, recording_file: Path, index: int
) -> None:
    """The native subprocess returns exact writable values even with Julia in its parent."""
    expected, label = read_shd_recording(recording_file, index, backend="numpy")
    actual, native_label = read_shd_recording(recording_file, index, backend="go")
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64 and actual.flags.writeable
    assert native_label == label
    sample = SampleRecord("train", recording_file.name, index, label, "speaker")
    np.testing.assert_array_equal(read_event_sample(recording_file.parent, "shd", sample), expected)


@pytest.mark.parametrize("backend", ["go", "numpy"])
@pytest.mark.parametrize("index,budget", [(-1, 64), (2, 64), (0, -1), (0, 63), (True, 64)])
def test_public_shd_refuses_indices_and_result_budgets(
    native_shd: None, recording_file: Path, backend: Literal["go", "numpy"], index: int, budget: int
) -> None:
    """Neither backend publishes partial results for invalid indices or budgets."""
    with pytest.raises((ValueError, RuntimeError)):
        read_shd_recording(recording_file, index, backend=backend, maximum_bytes=budget)


@pytest.mark.parametrize("backend", ["go", "numpy"])
@pytest.mark.parametrize("damage", ["shape", "pair", "label", "label_overflow", "missing"])
def test_public_shd_malformed_recording_contract_agrees(
    native_shd: None, recording_file: Path, backend: Literal["go", "numpy"], damage: str
) -> None:
    """Real HDF5 shape, pair and label damage refuses across the public readers."""
    if damage == "missing":
        recording_file = recording_file.with_name("missing.h5")
    else:
        with h5py.File(recording_file, "a") as handle:
            if damage == "shape":
                del handle["labels"]
                handle["labels"] = np.array([3], dtype=np.uint8)
            elif damage == "pair":
                handle["spikes/units"][0] = np.array([0], dtype=np.int16)
            else:
                del handle["labels"]
                handle["labels"] = (
                    np.array([1 << 63, 17], dtype=np.uint64)
                    if damage == "label_overflow"
                    else np.array([3, 17], dtype=np.float32)
                )
    with pytest.raises((ValueError, RuntimeError, OSError)):
        read_shd_recording(recording_file, 0, backend=backend)


def test_manifest_reader_refuses_actual_label_drift(native_shd: None, recording_file: Path) -> None:
    """The selected HDF5 row cannot silently disagree with its manifest label."""
    sample = SampleRecord("train", recording_file.name, 0, 4, "speaker")
    with pytest.raises(ValueError, match="label does not match"):
        read_event_sample(recording_file.parent, "shd", sample)


def test_selected_unreadable_native_library_never_falls_back(
    native_shd: None, recording_file: Path, tmp_path: Path
) -> None:
    """An actual non-library file is an operator error, even when NumPy could read."""
    invalid = tmp_path / "broken.so"
    invalid.write_bytes(b"not an ELF library")
    previous = os.environ["SC_NEUROCORE_SHD_GO_LIBRARY"]
    os.environ["SC_NEUROCORE_SHD_GO_LIBRARY"] = str(invalid)
    try:
        with pytest.raises(RuntimeError, match="failed or refused"):
            read_shd_recording(recording_file, 0)
    finally:
        os.environ["SC_NEUROCORE_SHD_GO_LIBRARY"] = previous


def test_isolated_operator_configuration_carries_only_declared_shd_library(
    shd_library: Path, recording_file: Path
) -> None:
    """Trusted worker settings carry the native reader independently of N-MNIST."""
    configuration = EventWorkerConfiguration(
        dataset_root=recording_file.parent, shd_go_library=shd_library
    )
    environment = configuration.environment()
    assert environment["SC_NEUROCORE_SHD_GO_LIBRARY"] == str(shd_library)
    assert "SC_NEUROCORE_DATASET_GO_LIBRARY" not in environment


def test_native_blocked_file_read_has_finite_lifetime_and_is_reaped(
    native_shd: None, tmp_path: Path
) -> None:
    """A real FIFO blocks native file open until the reader lifetime stops its process."""
    pipe = tmp_path / "blocked.h5"
    os.mkfifo(pipe)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    before = set(children.read_text().split())
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="lifetime|failed or refused"):
        read_shd_recording(pipe, 0, backend="go")
    elapsed = time.monotonic() - started
    assert 28 <= elapsed <= 40, elapsed
    assert set(children.read_text().split()) == before
