# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event training through HTTP and real workers

"""Train generated SHD recordings through the actual Studio process surface."""

from __future__ import annotations

import json
import io
import os
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.event_training_budget import admit_event_training_input
from sc_neurocore.studio.event_training_data import DATASET_ROOT_ENV
from sc_neurocore.studio.platform import StudioRuntimeSettings
from tests.event_dataset_support import write_shd


@pytest.fixture
def experiment(tmp_path: Path) -> Iterator[tuple[TestClient, EventTrainingContract]]:
    """Configure a real operator corpus and a separate disposable job ledger."""
    root = tmp_path / "recordings"
    write_shd(root, {"train": [0, 0, 1, 1, 2, 2, 3, 3], "test": [4]})
    manifest = build_manifest("shd", root, version="generated-format-fixture")
    contract = EventTrainingContract(
        manifest,
        group_split(manifest, fractions={"train": 0.5, "evaluation": 0.5}, seed=7),
        EventBinning(1.0, 4, 700, 1, "merge"),
        "train",
        "evaluation",
    )
    previous = os.environ.get(DATASET_ROOT_ENV)
    os.environ[DATASET_ROOT_ENV] = str(root)
    app = create_app(
        StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"), job_default_timeout_seconds=60.0
        )
    )
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            yield client, contract
    finally:
        if previous is None:
            os.environ.pop(DATASET_ROOT_ENV, None)
        else:
            os.environ[DATASET_ROOT_ENV] = previous


def test_http_event_training_exports_replayable_input_digests(
    experiment: tuple[TestClient, EventTrainingContract],
) -> None:
    """Actual child training retains every input digest in its checkpoint."""
    client, contract = experiment
    response = client.post(
        "/api/training/start",
        json={
            "dataset": "shd",
            "epochs": 1,
            "batch_size": 3,
            "hidden": [4],
            "timesteps": 4,
            "seed": 7,
            "event_data": contract.to_dict(),
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 60.0
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/training/status/{job_id}")
        assert response.status_code == 200, response.text
        status = response.json()
        if status["status"] in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.1)
    assert status["status"] == "completed", status
    assert status["final_metrics"]["train_loss"] > 0
    assert status["final_metrics"]["val_loss"] > 0
    exported = client.get(f"/api/training/checkpoint/{job_id}")
    assert exported.status_code == 200, exported.text
    checkpoint = exported.json()
    assert checkpoint["config"]["event_data"] == contract.to_dict()
    assert checkpoint["weight_checkpoint"]["architecture"] == "700->4->20"
    assert checkpoint["config"]["event_data"]["digests"] == {
        "manifest": contract.manifest.digest,
        "split": contract.split.digest,
        "encoder": contract.encoder.digest,
    }
    imported = client.post("/api/training/checkpoint/import", json=checkpoint)
    assert imported.status_code == 200, imported.text
    assert imported.json()["config"] == checkpoint["config"]
    manager = cast(FastAPI, client.app).state.studio_job_manager
    assert manager.record(job_id).execution_model == "process"
    events = [
        json.loads(line)
        for line in manager.read_artifact(job_id, "training/events.jsonl")
        .payload.decode()
        .splitlines()
    ]
    assert events[0]["data"]["input_data"] == contract.receipt()
    assert events[0]["data"]["input_admission"] == admit_event_training_input(contract, 3)


def test_http_unconfigured_dataset_is_refused_before_job_allocation(
    experiment: tuple[TestClient, EventTrainingContract],
) -> None:
    """The HTTP surface cannot allocate an event job without operator data."""
    client, contract = experiment
    before = client.get("/api/training/jobs").json()
    del os.environ[DATASET_ROOT_ENV]
    response = client.post(
        "/api/training/start",
        json={"dataset": "shd", "timesteps": 4, "event_data": contract.to_dict()},
    )
    assert response.status_code == 422, response.text
    assert client.get("/api/training/jobs").json() == before


def test_http_oversized_event_window_is_refused_before_job_allocation(
    experiment: tuple[TestClient, EventTrainingContract],
) -> None:
    """A huge valid temporal declaration fails at admission, without a worker."""
    client, contract = experiment
    oversized = replace(contract, encoder=EventBinning(1.0, 10**18, 700, 1, "merge"))
    before = client.get("/api/training/jobs").json()
    response = client.post(
        "/api/training/start",
        json={"dataset": "shd", "timesteps": 10**18, "event_data": oversized.to_dict()},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["field"] == "event_data"
    assert "reduce batch_size or timesteps" in response.json()["detail"]["reason"]
    assert client.get("/api/training/jobs").json() == before


def _wait_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    """Wait for this specific HTTP job to reach its declared terminal state."""
    deadline = time.monotonic() + 60.0
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/training/status/{job_id}")
        assert response.status_code == 200, response.text
        status = response.json()
        if status["status"] in {"completed", "failed", "stopped"}:
            return status
        time.sleep(0.1)
    raise AssertionError(f"event training job did not finish: {status}")


def _assert_same_saved_state(left: object, right: object) -> None:
    """Compare every saved tensor and primitive, independent of zip framing."""
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_same_saved_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_same_saved_state(first, second)
    else:
        assert left == right


def test_event_exact_resume_matches_uninterrupted_training_and_refuses_changed_encoder(
    experiment: tuple[TestClient, EventTrainingContract],
) -> None:
    """Continue a saved optimiser/RNG position only for its original event input."""
    client, contract = experiment
    config = {
        "dataset": "shd",
        "epochs": 1,
        "batch_size": 3,
        "hidden": [4],
        "timesteps": 4,
        "seed": 7,
        "event_data": contract.to_dict(),
    }
    source = client.post("/api/training/start", json=config)
    assert source.status_code == 200, source.text
    source_id = source.json()["job_id"]
    assert _wait_terminal(client, source_id)["status"] == "completed"
    continued_config = {**config, "epochs": 2}
    attached = client.post(
        "/api/studio/training/weight-restore/attach",
        json={"source_job_id": source_id, "config": continued_config, "mode": "exact_resume"},
    )
    assert attached.status_code == 200, attached.text
    resumed = _wait_terminal(client, attached.json()["job_id"])
    assert resumed["status"] == "completed", resumed
    uninterrupted = client.post("/api/training/start", json=continued_config)
    assert uninterrupted.status_code == 200, uninterrupted.text
    baseline = _wait_terminal(client, uninterrupted.json()["job_id"])
    assert baseline["status"] == "completed", baseline
    assert resumed["final_metrics"] == baseline["final_metrics"]
    manager = cast(FastAPI, client.app).state.studio_job_manager
    # Each archive passes its own digest check; compare its complete saved
    # state rather than zip framing that can differ after materialisation.
    resumed_bytes = manager.read_artifact(
        attached.json()["job_id"], "training/model_state.pt"
    ).payload
    baseline_bytes = manager.read_artifact(
        uninterrupted.json()["job_id"], "training/model_state.pt"
    ).payload
    _assert_same_saved_state(
        torch.load(io.BytesIO(resumed_bytes), weights_only=True, map_location="cpu"),
        torch.load(io.BytesIO(baseline_bytes), weights_only=True, map_location="cpu"),
    )
    changed = replace(contract, encoder=EventBinning(2.0, 4, 700, 1, "merge"))
    before = client.get("/api/training/jobs").json()
    refused = client.post(
        "/api/studio/training/weight-restore/attach",
        json={
            "source_job_id": source_id,
            "config": {**continued_config, "event_data": changed.to_dict()},
            "mode": "exact_resume",
        },
    )
    assert refused.status_code == 422, refused.text
    assert "source manifest, split and encoder unchanged" in refused.text
    assert client.get("/api/training/jobs").json() == before
    del os.environ[DATASET_ROOT_ENV]
    unavailable = client.post(
        "/api/studio/training/weight-restore/attach",
        json={"source_job_id": source_id, "config": continued_config, "mode": "warm_start"},
    )
    assert unavailable.status_code == 422, unavailable.text
    assert client.get("/api/training/jobs").json() == before
