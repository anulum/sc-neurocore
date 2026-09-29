# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public native DVS backend selection

"""Exercise actual Mojo command selection, eager/lazy camera paths and operator configuration."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from sc_neurocore.datasets import load_dvs_cifar10
from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import SampleRecord
from sc_neurocore.studio.platform.storage_event_worker_configuration import EventWorkerConfiguration
from tests.test_accel_mojo_dvs import mojo_dvs_executable as mojo_dvs_executable


@pytest.mark.parametrize("backend", ["auto", "numpy", "mojo"])
def test_public_selected_backend_and_dataset_paths_keep_owned_values(
    tmp_path: Path,
    mojo_dvs_executable: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["auto", "numpy", "mojo"],
) -> None:
    """Real environment declarations reach native decoding through direct, eager and lazy APIs."""
    path = tmp_path / "train/0/events.npy"
    path.parent.mkdir(parents=True)
    values = np.array([[1, 2, 1, 1.002], [3, 4, 0, 2.004]], dtype=">f8", order="F")
    np.save(path, values)
    monkeypatch.setenv("SC_NEUROCORE_DVS_MOJO_EXE", str(mojo_dvs_executable))
    result = read_dvs_recording(path, backend=backend)
    np.testing.assert_array_equal(result, values)
    assert result.flags.owndata and result.flags.writeable and result.flags.c_contiguous
    eager, labels = load_dvs_cifar10(tmp_path)
    lazy = read_event_sample(
        tmp_path, "dvs_cifar10", SampleRecord("train", "train/0/events.npy", 0, 0, "recording")
    )
    np.testing.assert_array_equal(eager[0], values)
    np.testing.assert_array_equal(lazy, values)
    np.testing.assert_array_equal(labels, [0])


def test_declared_native_recording_refusal_never_falls_back(
    tmp_path: Path, mojo_dvs_executable: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected command refusing a damaged recording propagates failure through both dataset paths."""
    path = tmp_path / "train/0/events.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((2, 4)))
    path.write_bytes(path.read_bytes()[:-1])
    monkeypatch.setenv("SC_NEUROCORE_DVS_MOJO_EXE", str(mojo_dvs_executable))
    with pytest.raises(RuntimeError, match="failed or refused"):
        load_dvs_cifar10(tmp_path)
    with pytest.raises(RuntimeError, match="failed or refused"):
        read_event_sample(
            tmp_path, "dvs_cifar10", SampleRecord("train", "train/0/events.npy", 0, 0, "recording")
        )


@pytest.mark.parametrize("declared", ["", "relative-mojo", "/absent-dvs-mojo", "/"])
def test_invalid_native_declaration_refuses_even_with_valid_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: str
) -> None:
    """Invalid native settings cannot silently select NumPy despite a readable valid recording."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((2, 4)))
    monkeypatch.setenv("SC_NEUROCORE_DVS_MOJO_EXE", declared)
    with pytest.raises(RuntimeError, match="executable"):
        read_dvs_recording(path)
    np.testing.assert_array_equal(read_dvs_recording(path, backend="numpy"), np.zeros((2, 4)))


def test_explicit_mojo_without_operator_selection_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit Mojo cannot discover a runtime or downgrade to an undeclared reference backend."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((0, 4)))
    monkeypatch.delenv("SC_NEUROCORE_DVS_MOJO_EXE", raising=False)
    with pytest.raises(RuntimeError, match="requires"):
        read_dvs_recording(path, backend="mojo")


def test_operator_configuration_emits_declared_dvs_setting_only(
    tmp_path: Path, mojo_dvs_executable: Path
) -> None:
    """The strict operator-owned worker model emits the same executable selection as the reader."""
    declared = EventWorkerConfiguration(
        dataset_root=tmp_path, dvs_mojo_executable=mojo_dvs_executable
    )
    assert declared.environment()["SC_NEUROCORE_DVS_MOJO_EXE"] == str(mojo_dvs_executable)
    assert (
        "SC_NEUROCORE_DVS_MOJO_EXE"
        not in EventWorkerConfiguration(dataset_root=tmp_path).environment()
    )
    for path in (Path("relative"), tmp_path, tmp_path / "missing"):
        with pytest.raises(ValueError):
            EventWorkerConfiguration(dataset_root=tmp_path, dvs_mojo_executable=path)


def test_native_public_reader_reaps_its_actual_fifo_timeout(
    tmp_path: Path, mojo_dvs_executable: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A public native read cannot retain its child after the guarded FIFO deadline."""
    import os

    fifo = tmp_path / "blocked.npy"
    os.mkfifo(fifo)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    before = set(children.read_text().split())
    monkeypatch.setenv("SC_NEUROCORE_DVS_MOJO_EXE", str(mojo_dvs_executable))
    with pytest.raises(RuntimeError, match="lifetime|failed or refused"):
        read_dvs_recording(fifo)
    assert set(children.read_text().split()) == before
