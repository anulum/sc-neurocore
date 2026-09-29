# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native event decoding in actual Studio training workers

"""Verify native decoding across HTTP admission, child training and saved state."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import pytest
from starlette.testclient import TestClient

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.event_training_data import DATASET_ROOT_ENV
from sc_neurocore.studio.platform import StudioRuntimeSettings
from tests.event_dataset_support import write_nmnist
from tests.test_accel_event_recordings import native_recording_library as native_recording_library
from tests.test_accel_event_recordings import go_recording_library as go_recording_library
from tests.test_accel_event_recordings import rust_recording_library as rust_recording_library
from tests.test_accel_event_recordings import mojo_recording_library as mojo_recording_library


def _train(client: TestClient, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Start one actual child and observe its HTTP terminal result."""
    response = client.post("/api/training/start", json=config)
    assert response.status_code == 200, response.text
    job_id = str(response.json()["job_id"])
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/training/status/{job_id}")
        assert response.status_code == 200, response.text
        status = response.json()
        if status["status"] in {"completed", "failed", "stopped"}:
            return job_id, status
        time.sleep(0.1)
    raise AssertionError(f"training worker did not finish: {job_id}")


def test_native_child_matches_numpy_saved_state_and_refuses_missing_library(
    tmp_path: Path,
    native_recording_library: tuple[Literal["go", "rust", "mojo", "julia"], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native and NumPy readers produce identical full state in real CPU training."""
    backend, library = native_recording_library
    library_variable = (
        "PYTHON_JULIACALL_PROJECT"
        if backend == "julia"
        else f"SC_NEUROCORE_DATASET_{backend.upper()}_LIBRARY"
    )
    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "0")
    monkeypatch.delenv("SC_NEUROCORE_DATASET_GO_LIBRARY", raising=False)
    monkeypatch.delenv("SC_NEUROCORE_DATASET_RUST_LIBRARY", raising=False)
    monkeypatch.delenv("SC_NEUROCORE_DATASET_MOJO_LIBRARY", raising=False)
    root = tmp_path / "recordings"
    write_nmnist(root, {"train": {0: 3, 1: 2}})
    manifest = build_manifest("nmnist", root, version="generated-format-fixture")
    contract = EventTrainingContract(
        manifest,
        group_split(manifest, fractions={"train": 0.6, "evaluation": 0.4}, seed=7),
        EventBinning(1.0, 4, 34, 34, "merge"),
        "train",
        "evaluation",
    )
    monkeypatch.setenv(DATASET_ROOT_ENV, str(root))
    app = create_app(
        StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs"), job_default_timeout_seconds=60)
    )
    config = {
        "dataset": "nmnist",
        "epochs": 1,
        "batch_size": 3,
        "hidden": [4],
        "timesteps": 4,
        "seed": 7,
        "event_data": contract.to_dict(),
    }
    state_files: list[Path] = []
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for native in (True, False):
            if native:
                monkeypatch.setenv(library_variable, str(library))
                if backend == "julia":
                    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "1")
            else:
                if backend == "julia":
                    monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "0")
                else:
                    monkeypatch.delenv(library_variable)
            job_id, status = _train(client, config)
            assert status["status"] == "completed", status
            checkpoint = client.get(f"/api/training/checkpoint/{job_id}")
            assert checkpoint.status_code == 200, checkpoint.text
            assert checkpoint.json()["config"]["event_data"] == contract.to_dict()
            artifact = client.get(f"/api/studio/jobs/{job_id}/artifacts/training/model_state.pt")
            assert artifact.status_code == 200, artifact.text
            assert (
                hashlib.sha256(artifact.content).hexdigest()
                == artifact.headers["x-studio-artifact-sha256"]
            )
            output = tmp_path / f"{job_id}.pt"
            output.write_bytes(artifact.content)
            state_files.append(output)
        comparator = (
            Path(__file__).resolve().parents[1] / "studio/frontend/e2e/event_training_state.py"
        )
        subprocess.run(
            [sys.executable, str(comparator), str(state_files[0]), str(state_files[1])],
            check=True,
            capture_output=True,
            timeout=30,
        )
        monkeypatch.setenv(library_variable, str(tmp_path / "absent.so"))
        if backend == "julia":
            monkeypatch.setenv("SC_NEUROCORE_DATASET_JULIA_ENABLED", "1")
        failed_id, failed_status = _train(client, config)
        assert failed_status["status"] == "failed", failed_status
        assert failed_id not in {path.stem for path in state_files}
