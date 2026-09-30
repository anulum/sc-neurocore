# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public native DVS backend selection

"""Exercise actual Julia runtime selection, eager/lazy camera paths and operator configuration."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from sc_neurocore.datasets import load_dvs_cifar10
from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import SampleRecord
from sc_neurocore.studio.platform.storage_event_worker_configuration import EventWorkerConfiguration
import subprocess
from tests.julia_runtimes import require_julia_runtime


@pytest.fixture(scope="module", params=["1.11", "release"])
def julia_dvs_executable(request: pytest.FixtureRequest) -> Path:
    """Resolve each installed runtime directly, keeping Juliaup out of production reads."""
    channel = str(request.param)
    result = subprocess.check_output(
        [
            str(require_julia_runtime(channel)),
            "--startup-file=no",
            "-e",
            "print(joinpath(Sys.BINDIR, Base.julia_exename()))",
        ],
        text=True,
        timeout=20,
    )
    executable = Path(result).resolve()
    assert executable.is_absolute() and executable.is_file()
    return executable


@pytest.mark.parametrize("backend", ["auto", "numpy", "julia"])
def test_public_selected_backend_and_dataset_paths_keep_owned_values(
    tmp_path: Path,
    julia_dvs_executable: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["auto", "numpy", "julia"],
) -> None:
    """Real environment declarations reach native decoding through direct, eager and lazy APIs."""
    path = tmp_path / "train/0/events.npy"
    path.parent.mkdir(parents=True)
    values = np.array([[1, 2, 1, 1.002], [3, 4, 0, 2.004]], dtype=">f8", order="F")
    np.save(path, values)
    monkeypatch.setenv("SC_NEUROCORE_DVS_JULIA_EXE", str(julia_dvs_executable))
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
    tmp_path: Path, julia_dvs_executable: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected command refusing a damaged recording propagates failure through both dataset paths."""
    path = tmp_path / "train/0/events.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((2, 4)))
    path.write_bytes(path.read_bytes()[:-1])
    monkeypatch.setenv("SC_NEUROCORE_DVS_JULIA_EXE", str(julia_dvs_executable))
    with pytest.raises(RuntimeError, match="failed or refused"):
        load_dvs_cifar10(tmp_path)
    with pytest.raises(RuntimeError, match="failed or refused"):
        read_event_sample(
            tmp_path, "dvs_cifar10", SampleRecord("train", "train/0/events.npy", 0, 0, "recording")
        )


@pytest.mark.parametrize("declared", ["", "relative-go", "/absent-dvs-go", "/"])
def test_invalid_native_declaration_refuses_even_with_valid_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: str
) -> None:
    """Invalid native settings cannot silently select NumPy despite a readable valid recording."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((2, 4)))
    monkeypatch.setenv("SC_NEUROCORE_DVS_JULIA_EXE", declared)
    with pytest.raises(RuntimeError, match="executable"):
        read_dvs_recording(path)
    np.testing.assert_array_equal(read_dvs_recording(path, backend="numpy"), np.zeros((2, 4)))


def test_explicit_julia_without_operator_selection_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit Julia cannot discover a runtime or downgrade to an undeclared reference backend."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((0, 4)))
    monkeypatch.delenv("SC_NEUROCORE_DVS_JULIA_EXE", raising=False)
    with pytest.raises(RuntimeError, match="requires"):
        read_dvs_recording(path, backend="julia")


def test_operator_configuration_emits_declared_dvs_setting_only(
    tmp_path: Path, julia_dvs_executable: Path
) -> None:
    """The strict operator-owned worker model emits the same executable selection as the reader."""
    declared = EventWorkerConfiguration(
        dataset_root=tmp_path, dvs_julia_executable=julia_dvs_executable
    )
    assert declared.environment()["SC_NEUROCORE_DVS_JULIA_EXE"] == str(julia_dvs_executable)
    assert (
        "SC_NEUROCORE_DVS_JULIA_EXE"
        not in EventWorkerConfiguration(dataset_root=tmp_path).environment()
    )
    for path in (Path("relative"), tmp_path, tmp_path / "missing"):
        with pytest.raises(ValueError):
            EventWorkerConfiguration(dataset_root=tmp_path, dvs_julia_executable=path)


def test_native_public_reader_reaps_its_actual_fifo_timeout(
    tmp_path: Path, julia_dvs_executable: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A public native read cannot retain its child after the guarded FIFO deadline."""
    import os

    fifo = tmp_path / "blocked.npy"
    os.mkfifo(fifo)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    before = set(children.read_text().split())
    monkeypatch.setenv("SC_NEUROCORE_DVS_JULIA_EXE", str(julia_dvs_executable))
    with pytest.raises(RuntimeError, match="lifetime|failed or refused"):
        read_dvs_recording(fifo)
    assert set(children.read_text().split()) == before


@pytest.mark.parametrize("other", ["GO", "RUST"])
def test_ambiguous_native_declarations_require_explicit_backend(
    tmp_path: Path, julia_dvs_executable: Path, monkeypatch: pytest.MonkeyPatch, other: str
) -> None:
    """Two declared runtimes refuse auto while explicit Julia and NumPy remain deterministic."""
    path = tmp_path / "events.npy"
    np.save(path, np.zeros((2, 4)))
    monkeypatch.setenv("SC_NEUROCORE_DVS_JULIA_EXE", str(julia_dvs_executable))
    monkeypatch.setenv(f"SC_NEUROCORE_DVS_{other}_EXE", "unavailable")
    with pytest.raises(RuntimeError, match="one native declaration"):
        read_dvs_recording(path)
    np.testing.assert_array_equal(read_dvs_recording(path, backend="julia"), np.zeros((2, 4)))
    np.testing.assert_array_equal(read_dvs_recording(path, backend="numpy"), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="one native executable"):
        EventWorkerConfiguration(
            dataset_root=tmp_path,
            dvs_go_executable=julia_dvs_executable,
            dvs_julia_executable=julia_dvs_executable,
        )


def test_julia_nonexecutable_operator_path_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An actual nonexecutable file cannot be selected by the public reader or operator model."""
    executable = tmp_path / "nonexecutable"
    executable.write_bytes(b"not executable")
    executable.chmod(0o600)
    monkeypatch.setenv("SC_NEUROCORE_DVS_JULIA_EXE", str(executable))
    with pytest.raises(RuntimeError, match="executable"):
        read_dvs_recording(tmp_path / "missing.npy", backend="julia")
    with pytest.raises(ValueError, match="unavailable"):
        EventWorkerConfiguration(dataset_root=tmp_path, dvs_julia_executable=executable)
