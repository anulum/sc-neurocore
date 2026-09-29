# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native isolated event training state parity

"""Compare complete checkpoint states from actual native and NumPy worker generations."""

import hashlib
import os
import subprocess
import sys

import numpy as np
from pathlib import Path
from typing import Literal

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK
from sc_neurocore.studio.training_contract import resolve_training_config
from tests.event_dataset_support import write_nmnist, write_shd
from tests.studio_storage_generation_runs import api_runtime
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority
from tests.studio_storage_launcher_support import shutdown, start
from tests.test_accel_event_recordings import go_recording_library as go_recording_library
from tests.test_accel_event_recordings import mojo_recording_library as mojo_recording_library
from tests.test_accel_event_recordings import native_recording_library as native_recording_library
from tests.test_accel_event_recordings import rust_recording_library as rust_recording_library
from tests.test_accel_go_dvs_recordings import go_dvs_executable as go_dvs_executable
from tests.test_accel_go_shd_abi import shd_library as shd_library
from tests.test_accel_julia_shd import julia_shd_executable as julia_shd_executable
from tests.test_accel_mojo_shd import mojo_shd_executable as mojo_shd_executable
from tests.test_accel_rust_dvs import rust_dvs_executable as rust_dvs_executable
from tests.test_accel_rust_shd import rust_shd_library as rust_shd_library
from tests.test_accel_rust_shd import rust_shd_runtime as rust_shd_runtime
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_isolated_jobs import _configuration, request_on


def _julia_runtime() -> tuple[Path, Path]:
    """Return the Julia executable and JuliaCall project the worker is given.

    Each comes from its ``PYTHON_JULIACALL_*`` variable when set; otherwise the
    newest installed Julia 1.11 and the project JuliaCall itself uses in this
    environment, so the case does not depend on an undeclared shell setting.
    """
    configured = os.environ.get("PYTHON_JULIACALL_EXE")
    installed = sorted((Path.home() / ".julia/juliaup").glob("julia-1.11.*/bin/julia"))
    assert configured or installed, "Julia 1.11 required; set PYTHON_JULIACALL_EXE"
    executable = Path(configured) if configured else installed[-1]
    configured = os.environ.get("PYTHON_JULIACALL_PROJECT")
    project = Path(configured) if configured else Path(sys.prefix) / "julia_env"
    assert (project / "Project.toml").is_file(), (
        f"no JuliaCall project at {project}; set PYTHON_JULIACALL_PROJECT"
    )
    return executable, project


def _compare_isolated_training(
    base: Path,
    tmp_path: Path,
    authority: Authority,
    native_recording_library: tuple[Literal["go", "rust", "mojo", "julia"], Path],
    *,
    dataset: Literal["nmnist", "shd", "dvs_cifar10"] = "nmnist",
) -> None:
    """Trusted native settings survive launch and reproduce all model, optimiser and RNG state."""
    backend, library = native_recording_library
    root = tmp_path / "recordings"
    if dataset == "shd":
        write_shd(root, {"train": [0, 0, 1, 1, 2, 2]})
    elif dataset == "dvs_cifar10":
        for label, count in ((0, 3), (1, 2)):
            directory = root / "train" / str(label)
            directory.mkdir(parents=True)
            for index in range(count):
                np.save(
                    directory / f"{index}.npy",
                    np.array(
                        [[label, index, 1, 1.002], [label, index, 0, 2.004]], dtype=np.float64
                    ),
                )
    else:
        write_nmnist(root, {"train": {0: 3, 1: 2}})
    manifest = build_manifest(dataset, root, version="generated-format-fixture")
    contract = EventTrainingContract(
        manifest,
        group_split(manifest, fractions={"train": 0.6, "evaluation": 0.4}, seed=7),
        EventBinning(
            1.0,
            4,
            700 if dataset == "shd" else 128 if dataset == "dvs_cifar10" else 34,
            1 if dataset == "shd" else 128 if dataset == "dvs_cifar10" else 34,
            "merge",
        ),
        "train",
        "evaluation",
    )
    config = resolve_training_config(
        {
            "dataset": dataset,
            "epochs": 1,
            "batch_size": 3,
            "hidden": [4],
            "timesteps": 4,
            "seed": 7,
            "event_data": contract.to_dict(),
        }
    ).to_public_dict()
    source = Path(__file__).resolve().parents[1]
    for installed in (
        "src/sc_neurocore/accel/rust/safety/libnmnist.so",
        "src/sc_neurocore/accel/mojo/kernels/libnmnist.so",
        "src/sc_neurocore/accel/go/services/loaders/libloaders.so",
        "src/sc_neurocore/accel/go/services/loaders/libshd.so",
        "src/sc_neurocore/accel/rust/safety/libshd.so",
    ):
        assert not (source / installed).exists(), (
            "NumPy baseline needs an installation without default native libraries"
        )
    checkpoints: list[Path] = []
    for selected in (True, False):
        operator: dict[str, object] = {"dataset_root": str(root)}
        if selected:
            if backend == "julia" and dataset == "nmnist":
                operator["julia"] = {
                    "executable": str(_julia_runtime()[0]),
                    "project": str(library),
                    "handle_signals": "no",
                }
            else:
                operator[
                    (
                        f"shd_{backend}_executable"
                        if backend in ("julia", "mojo")
                        else f"shd_{backend}_library"
                    )
                    if dataset == "shd"
                    else f"dvs_{backend}_executable"
                    if dataset == "dvs_cifar10"
                    else f"{backend}_library"
                ] = str(library)
        launcher = start(base, event_input=operator)
        try:
            manager = IsolatedJobManager(
                api_runtime(base, authority),
                _configuration(base),
                allowed_kinds=frozenset({"training"}),
                default_timeout_seconds=120.0,
            )
            with request_on("/api/training/start"):
                submitted = manager.submit_process_task(
                    kind="training",
                    owner="studio-training",
                    request_id="native-parity",
                    task_path=TRAINING_PROCESS_TASK,
                    payload=config,
                    training_config=config,
                )
                completed = manager.wait(submitted.job_id, timeout_seconds=120.0)
            assert completed.status == "completed", completed.error
            with request_on("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}", "GET"):
                saved = manager.read_artifact(submitted.job_id, "training/model_state.pt")
            assert hashlib.sha256(saved.payload).hexdigest() == saved.artifact.sha256
            output = tmp_path / f"{backend}-{'native' if selected else 'numpy'}.pt"
            output.write_bytes(saved.payload)
            checkpoints.append(output)
            with request_on("/api/studio/jobs/status", "GET"):
                assert manager.unreaped_workers == ()
            assert manager.generation_failures == {}
        finally:
            shutdown(launcher)
    subprocess.run(
        [
            sys.executable,
            str(source / "studio/frontend/e2e/event_training_state.py"),
            *(str(path) for path in checkpoints),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    operator = {"dataset_root": str(root)}
    if backend == "julia" and dataset == "nmnist":
        altered = tmp_path / "julia-project"
        altered.mkdir()
        for name in ("Project.toml", "Manifest.toml"):
            (altered / name).write_bytes((library / name).read_bytes())
        operator["julia"] = {
            "executable": str(_julia_runtime()[0]),
            "project": str(altered),
            "handle_signals": "no",
        }
    else:
        altered = tmp_path / f"{backend}-altered.so"
        altered.write_bytes(library.read_bytes())
        if (backend in ("julia", "mojo") and dataset == "shd") or dataset == "dvs_cifar10":
            altered.chmod(0o700)
        operator[
            (
                f"shd_{backend}_executable"
                if backend in ("julia", "mojo")
                else f"shd_{backend}_library"
            )
            if dataset == "shd"
            else f"dvs_{backend}_executable"
            if dataset == "dvs_cifar10"
            else f"{backend}_library"
        ] = str(altered)
    launcher = start(base, event_input=operator)
    try:
        if backend == "julia" and dataset == "nmnist":
            (altered / "Project.toml").rename(altered / "Project.unavailable")
        else:
            altered.write_bytes(b"invalid native library")
        manager = IsolatedJobManager(
            api_runtime(base, authority),
            _configuration(base),
            allowed_kinds=frozenset({"training"}),
            default_timeout_seconds=120.0,
        )
        with request_on("/api/training/start"):
            submitted = manager.submit_process_task(
                kind="training",
                owner="studio-training",
                request_id="native-refusal",
                task_path=TRAINING_PROCESS_TASK,
                payload=config,
                training_config=config,
            )
            refused = manager.wait(submitted.job_id, timeout_seconds=120.0)
        assert refused.status == "failed", refused
        assert not any(
            item.relative_path == "training/model_state.pt" for item in refused.artifacts
        )
        with request_on("/api/studio/jobs/status", "GET"):
            assert manager.unreaped_workers == ()
        assert manager.generation_failures == {}
    finally:
        shutdown(launcher)


def test_native_isolated_checkpoint_matches_numpy(
    base: Path,
    tmp_path: Path,
    authority: Authority,
    native_recording_library: tuple[Literal["go", "rust", "mojo"], Path],
) -> None:
    """Compiled decoders match full NumPy saved state and refuse altered libraries."""
    _compare_isolated_training(base, tmp_path, authority, native_recording_library)


def test_julia_isolated_checkpoint_matches_numpy(
    base: Path,
    tmp_path: Path,
    authority: Authority,
) -> None:
    """Explicit offline Julia runtime reproduces saved state and refuses a removed project."""
    project = _julia_runtime()[1]
    _compare_isolated_training(base, tmp_path, authority, ("julia", project))


def test_go_shd_isolated_checkpoint_matches_numpy(
    base: Path, tmp_path: Path, authority: Authority, shd_library: Path
) -> None:
    """SHD native subprocess wiring preserves all saved state and refuses library damage."""
    _compare_isolated_training(base, tmp_path, authority, ("go", shd_library), dataset="shd")


def test_rust_shd_isolated_checkpoint_matches_numpy(
    base: Path, tmp_path: Path, authority: Authority, rust_shd_library: Path
) -> None:
    """Rust SHD worker wiring preserves every saved state and refuses altered native code."""
    _compare_isolated_training(base, tmp_path, authority, ("rust", rust_shd_library), dataset="shd")


def test_standalone_julia_shd_isolated_checkpoint_matches_numpy(
    base: Path, tmp_path: Path, authority: Authority, julia_shd_executable: Path
) -> None:
    """Explicit standalone Julia SHD worker matches all saved state and refuses executable corruption."""
    _compare_isolated_training(
        base, tmp_path, authority, ("julia", julia_shd_executable), dataset="shd"
    )


def test_standalone_mojo_shd_isolated_checkpoint_matches_numpy(
    base: Path, tmp_path: Path, authority: Authority, mojo_shd_executable: Path
) -> None:
    """The actual declared Mojo CLI preserves the full saved state and refuses corruption."""
    _compare_isolated_training(
        base, tmp_path, authority, ("mojo", mojo_shd_executable), dataset="shd"
    )


def test_go_dvs_isolated_checkpoint_matches_numpy(
    base: Path, tmp_path: Path, authority: Authority, go_dvs_executable: Path
) -> None:
    """Guarded Go DVS selection preserves full trained state and refuses executable corruption."""
    _compare_isolated_training(
        base, tmp_path, authority, ("go", go_dvs_executable), dataset="dvs_cifar10"
    )


def test_rust_dvs_isolated_checkpoint_matches_numpy(
    base: Path, tmp_path: Path, authority: Authority, rust_dvs_executable: Path
) -> None:
    """Guarded Rust DVS preserves full trained state and refuses executable corruption."""
    _compare_isolated_training(
        base, tmp_path, authority, ("rust", rust_dvs_executable), dataset="dvs_cifar10"
    )
