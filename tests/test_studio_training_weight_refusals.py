# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training weight refusals over real workers and persisted state

"""Distinguish authored weight refusals from real ledger corruption over HTTP."""

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioJobManager, StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs_ledger_schema import LEDGER_FILENAME
from sc_neurocore.studio.platform.training_weights import (
    build_training_weight_restore_plan,
    training_architecture_fingerprint,
)
from sc_neurocore.studio.training_refusals import TrainingRefusal

_ROUTES = (
    "/api/studio/training/weight-restore",
    "/api/studio/training/weight-restore/attach",
    "/api/studio/training/weight-restore/attach/live",
)
_CONFIG: dict[str, object] = {
    "dataset": "synthetic",
    "epochs": 1,
    "batch_size": 1024,
    "hidden": [4],
    "timesteps": 1,
}
Experiment = tuple[TestClient, StudioJobManager, str, str, Path]


def _object(value: object) -> dict[str, object]:
    """Read a stored or HTTP JSON object after validating its key shape."""
    assert isinstance(value, dict) and all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Experiment]:
    """Train source weights and keep a real bounded target worker available."""
    root = tmp_path_factory.mktemp("training-weight-refusals")
    app = create_app(
        StudioRuntimeSettings(
            job_root_path=str(root / "jobs"),
            audit_log_path=str(root / "audit.jsonl"),
            job_default_timeout_seconds=120.0,
        )
    )
    with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
        manager = cast(FastAPI, client.app).state.studio_job_manager
        assert isinstance(manager, StudioJobManager)
        started = client.post("/api/training/start", json=_CONFIG)
        assert started.status_code == 200, started.text
        source = _object(started.json())["job_id"]
        assert isinstance(source, str)
        completed = manager.wait(source, timeout_seconds=45.0)
        assert completed.status == "completed", completed.public_error
        target_response = client.post("/api/training/start", json={**_CONFIG, "epochs": 100000})
        assert target_response.status_code == 200, target_response.text
        target = _object(target_response.json())["job_id"]
        assert isinstance(target, str)
        try:
            deadline = time.monotonic() + 10.0
            while manager.record(target).status == "pending":
                assert time.monotonic() < deadline, "target worker did not start"
                time.sleep(0.01)
            assert manager.record(target).status == "running"
            yield client, manager, source, target, root
        finally:
            client.post("/api/training/stop", json={"job_id": target})
            manager.wait(target, timeout_seconds=10.0)


def _body(source: str, target: str, route: str) -> dict[str, object]:
    """Build the documented body for each actual lifecycle route."""
    if route == _ROUTES[1]:
        return {"source_job_id": source, "config": _CONFIG}
    if route == _ROUTES[2]:
        return {"source_job_id": source, "target_job_id": target}
    return {"source_job_id": source}


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("schema_version", "unsupported", "Training weight checkpoint schema is unsupported."),
        ("framework", "unsupported", "Training weight checkpoint framework is unsupported."),
        ("format", "unsupported", "Training weight checkpoint format is unsupported."),
        ("architecture", "", "Training weight checkpoint requires architecture."),
        ("final_metrics", [], "Training weight checkpoint metrics must be an object."),
        (
            "parameter_count",
            "caller-text-xyz",
            "Training weight checkpoint parameter count is invalid.",
        ),
        (
            "config_sha256",
            "caller-text-xyz",
            "Training weight checkpoint config digest is invalid.",
        ),
    ],
)
def test_actual_stored_checkpoint_refusal_is_authored(
    experiment: Experiment, route: str, field: str, value: object, reason: str
) -> None:
    """All three routes preserve useful refusals without admitting a new job."""
    client, manager, source, target, root = experiment
    with sqlite3.connect(root / "jobs" / LEDGER_FILENAME) as connection:
        original = connection.execute(
            "SELECT result FROM jobs WHERE job_id=?", (source,)
        ).fetchone()[0]
        result = _object(json.loads(original))
        checkpoint = _object(result["weight_checkpoint"])
        checkpoint[field] = value
        changed = json.dumps(result)
        before_count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        connection.execute("UPDATE jobs SET result=? WHERE job_id=?", (changed, source))
        connection.commit()
        try:
            response = client.post(route, json=_body(source, target, route))
            assert response.status_code == 422, response.text
            assert _object(response.json())["detail"] == reason
            assert "caller-text-xyz" not in response.text
            assert (
                connection.execute("SELECT result FROM jobs WHERE job_id=?", (source,)).fetchone()[
                    0
                ]
                == changed
            )
            assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == before_count
        finally:
            connection.execute("UPDATE jobs SET result=? WHERE job_id=?", (original, source))
    assert manager.record(target).status == "running"


@pytest.mark.parametrize("route", _ROUTES)
def test_actual_ledger_corruption_is_a_generic_server_failure(
    experiment: Experiment, route: str
) -> None:
    """Malformed stored JSON stays intact and never becomes a caller refusal."""
    client, _, source, target, root = experiment
    artifact = root / "jobs" / source / "training" / "model_state.pt"
    original_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with sqlite3.connect(root / "jobs" / LEDGER_FILENAME) as connection:
        original = connection.execute(
            "SELECT result FROM jobs WHERE job_id=?", (source,)
        ).fetchone()[0]
        corrupt = "{caller-text-xyz"
        connection.execute("UPDATE jobs SET result=? WHERE job_id=?", (corrupt, source))
        connection.commit()
        try:
            response = client.post(route, json=_body(source, target, route))
            assert response.status_code == 500, response.text
            assert _object(response.json())["detail"] == "Internal error"
            assert "caller-text-xyz" not in response.text
            assert "JSON" not in response.text
            assert (
                connection.execute("SELECT result FROM jobs WHERE job_id=?", (source,)).fetchone()[
                    0
                ]
                == corrupt
            )
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == original_hash
        finally:
            connection.execute("UPDATE jobs SET result=? WHERE job_id=?", (original, source))


def test_public_restore_plan_refusal_remains_value_error(experiment: Experiment) -> None:
    """The public planner retains ValueError compatibility and explicit provenance."""
    _, manager, source, _, _ = experiment
    result = _object(manager.record(source).result)
    with pytest.raises(ValueError, match="config digest mismatch") as caught:
        build_training_weight_restore_plan(
            source_job_id=source,
            source_status="completed",
            weight_checkpoint=_object(result["weight_checkpoint"]),
            expected_config_sha256="0" * 64,
        )
    assert isinstance(caught.value, TrainingRefusal)


def test_real_trained_weights_still_restore(experiment: Experiment) -> None:
    """The actual bounded restore worker still emits verified materialisation."""
    client, manager, source, target, _ = experiment
    response = client.post(_ROUTES[0], json=_body(source, target, _ROUTES[0]))
    assert response.status_code == 200, response.text
    body = _object(response.json())
    materialization = _object(body["materialization"])
    assert isinstance(materialization["loaded_key_count"], int)
    assert materialization["loaded_key_count"] > 0
    job_id = body["job_id"]
    assert isinstance(job_id, str)
    assert manager.record(job_id).status == "completed"


@pytest.mark.parametrize("route", _ROUTES)
def test_invalid_job_identifier_unicode_uses_field_validation(
    experiment: Experiment, route: str
) -> None:
    """Real HTTP field validation rejects invalid Unicode before ledger access."""
    client, _, _, target, _ = experiment
    response = client.post(
        route,
        content=json.dumps(_body("sj_\ud800", target, route)),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422, response.text
    detail = _object(response.json())["detail"]
    assert isinstance(detail, list) and len(detail) == 1
    error = _object(detail[0])
    assert error["type"] == "string_unicode"
    assert error["loc"] == ["body", "source_job_id"]
    assert "UnicodeEncodeError" not in response.text
    assert "surrogates not allowed" not in response.text


def test_actual_trained_weights_warm_start_a_new_worker(experiment: Experiment) -> None:
    """The attach route trains from real source weights and seals its evidence."""
    client, manager, source, target, _ = experiment
    response = client.post(_ROUTES[1], json=_body(source, target, _ROUTES[1]))
    assert response.status_code == 200, response.text
    job_id = _object(response.json())["job_id"]
    assert isinstance(job_id, str)
    completed = manager.wait(job_id, timeout_seconds=45.0)
    assert completed.status == "completed", completed.error
    evidence = _object(
        json.loads(manager.read_artifact(job_id, "training/weight-restore-attach.json").payload)
    )
    assert evidence["mode"] == "warm_start"
    assert evidence["source_job_id"] == source


def test_actual_live_attach_is_applied_by_the_running_worker(experiment: Experiment) -> None:
    """A real running trainer consumes the HTTP command and writes live evidence."""
    client, manager, source, target, _ = experiment
    response = client.post(_ROUTES[2], json=_body(source, target, _ROUTES[2]))
    assert response.status_code == 200, response.text
    assert _object(response.json())["status"] == "attach_requested"
    deadline = time.monotonic() + 20.0
    while True:
        payload, _ = manager.read_live_artifact_bytes(
            target, "training/weight-restore-attach.json", offset=0
        )
        if payload:
            break
        assert manager.record(target).status == "running"
        assert time.monotonic() < deadline, "worker did not apply the attach"
        time.sleep(0.02)
    evidence = _object(json.loads(payload))
    assert evidence["mode"] == "live"
    assert evidence["source_job_id"] == source
    assert evidence["target_job_id"] == target


def test_public_fingerprint_conversion_is_not_marked_as_authored() -> None:
    """A real integer conversion fault keeps its unmarked library provenance."""
    with pytest.raises(ValueError, match="invalid literal") as caught:
        training_architecture_fingerprint({"hidden": ["caller-text-xyz"]})
    assert not isinstance(caught.value, TrainingRefusal)
