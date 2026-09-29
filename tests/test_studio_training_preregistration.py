# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Preregistered training acceptance criterion

"""A training run is judged against the criterion stored before it started.

The criterion travels in the training request, is digested and stored with
the job's configuration at submission, and the finished run is judged on its
unrounded validation metric through the public HTTP surface.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.training_config_storage import prepare_training_config
from sc_neurocore.studio.training_contract import TrainingConfigError, resolve_training_config
from sc_neurocore.studio.training_preregistration import (
    PREREGISTRATION_SCHEMA_VERSION,
    RATIONALE_MAX_CHARACTERS,
    TrainingPreregistration,
    resolve_training_preregistration,
)

_FAST_RUN = {"dataset": "synthetic", "epochs": 1, "batch_size": 32, "timesteps": 4}
_CRITERION = {
    "metric": "val_accuracy",
    "threshold": 0.5,
    "rationale": "above chance on ten classes",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A Studio client for the training routes."""
    pytest.importorskip("torch")
    return TestClient(create_app(), base_url="http://127.0.0.1")


class TestCriterion:
    def test_resolution_keeps_the_declared_criterion(self) -> None:
        criterion = resolve_training_preregistration(_CRITERION)
        assert criterion == TrainingPreregistration(
            "val_accuracy", 0.5, "above chance on ten classes"
        )
        assert criterion.direction == "at_least"
        assert resolve_training_preregistration({"metric": "val_loss", "threshold": 2}) == (
            TrainingPreregistration("val_loss", 2.0, "")
        )
        assert resolve_training_preregistration(None) is None

    def test_the_stored_form_resolves_again_and_a_changed_one_is_refused(self) -> None:
        criterion = resolve_training_preregistration(_CRITERION)
        assert criterion is not None
        stored = criterion.to_public_dict()
        assert stored["schema_version"] == PREREGISTRATION_SCHEMA_VERSION
        assert len(str(stored["sha256"])) == 64
        assert resolve_training_preregistration(stored) == criterion
        with pytest.raises(ValueError, match="digest does not match"):
            resolve_training_preregistration({**stored, "threshold": 0.4})

    def test_the_digest_binds_every_field(self) -> None:
        base = TrainingPreregistration("val_accuracy", 0.5, "hypothesis")
        others = [
            TrainingPreregistration("val_loss", 0.5, "hypothesis"),
            TrainingPreregistration("val_accuracy", 0.6, "hypothesis"),
            TrainingPreregistration("val_accuracy", 0.5, "another hypothesis"),
        ]
        assert len({base.sha256, *(other.sha256 for other in others)}) == 4

    @pytest.mark.parametrize(
        "value,message",
        [
            ("val_accuracy >= 0.5", "must be an object"),
            ({**_CRITERION, "when": "later"}, "unknown preregistration field"),
            ({**_CRITERION, "schema_version": "v0"}, "not the preregistration contract"),
            ({"threshold": 0.5}, "metric must be one of"),
            ({"metric": "train_accuracy", "threshold": 0.5}, "metric must be one of"),
            ({"metric": "val_accuracy"}, "threshold must be a number"),
            ({"metric": "val_accuracy", "threshold": True}, "threshold must be a number"),
            ({"metric": "val_accuracy", "threshold": "0.5"}, "threshold must be a number"),
            ({"metric": "val_accuracy", "threshold": math.nan}, "threshold must be finite"),
            ({"metric": "val_loss", "threshold": math.inf}, "threshold must be finite"),
            ({"metric": "val_loss", "threshold": -0.1}, "threshold must be finite"),
            ({"metric": "val_accuracy", "threshold": 1.01}, "threshold must be finite"),
            ({"metric": "conversion_accuracy_drop", "threshold": 1.01}, "threshold must be finite"),
            (
                {"metric": "conversion_accuracy_drop", "threshold": -0.01},
                "threshold must be finite",
            ),
            ({**_CRITERION, "rationale": 7}, "rationale must be text"),
            (
                {**_CRITERION, "rationale": "x" * (RATIONALE_MAX_CHARACTERS + 1)},
                "rationale must be text",
            ),
        ],
    )
    def test_an_unjudgeable_criterion_is_refused(self, value: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            resolve_training_preregistration(value)

    def test_the_verdict_follows_the_metric_direction(self) -> None:
        accuracy = TrainingPreregistration("val_accuracy", 0.5, "")
        loss = TrainingPreregistration("val_loss", 0.5, "")
        assert accuracy.judge(0.5)["passed"] is True
        assert accuracy.judge(0.4999)["passed"] is False
        assert loss.judge(0.5)["passed"] is True
        assert loss.judge(0.5001)["passed"] is False
        drop = TrainingPreregistration("conversion_accuracy_drop", 0.05, "")
        assert drop.direction == "at_most"
        assert drop.judge(-0.02)["passed"] is True and drop.judge(0.0501)["passed"] is False
        verdict = loss.judge(math.nan)
        assert verdict["passed"] is False and verdict["observed"] is None
        assert accuracy.judge(math.inf)["passed"] is False
        assert accuracy.judge(0.7) == {
            "direction": "at_least",
            "metric": "val_accuracy",
            "observed": 0.7,
            "passed": True,
            "preregistration_sha256": accuracy.sha256,
            "schema_version": PREREGISTRATION_SCHEMA_VERSION,
            "threshold": 0.5,
        }


class TestTrainingContract:
    def test_the_criterion_is_part_of_the_stored_configuration(self) -> None:
        resolved = resolve_training_config({**_FAST_RUN, "preregistration": _CRITERION})
        criterion = resolved.preregistration
        assert criterion is not None
        public = resolved.to_public_dict()
        assert public["preregistration"] == criterion.to_public_dict()
        assert resolve_training_config(public) == resolved
        stored, _ = prepare_training_config(public)
        assert '"preregistration"' in stored and criterion.sha256 in stored

    def test_a_run_without_a_criterion_stores_none(self) -> None:
        public = resolve_training_config(_FAST_RUN).to_public_dict()
        assert "preregistration" not in public

    def test_an_invalid_criterion_is_refused_by_field(self) -> None:
        with pytest.raises(TrainingConfigError) as refusal:
            resolve_training_config({**_FAST_RUN, "preregistration": {"metric": "val_loss"}})
        assert refusal.value.field == "preregistration"


def _run(client: TestClient, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Start a training run, wait for completion and return status and checkpoint."""
    response = client.post("/api/training/start", json=config)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 300.0
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/training/status/{job_id}").json()
        if status.get("status") in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.2)
    assert status.get("status") == "completed", status
    return status, client.get(f"/api/training/checkpoint/{job_id}").json()


class TestHttpSurface:
    def test_an_invalid_criterion_is_refused_before_a_job_exists(self, client: TestClient) -> None:
        response = client.post(
            "/api/training/start",
            json={**_FAST_RUN, "preregistration": {"metric": "val_accuracy", "threshold": 2}},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert (
            detail["error"] == "training_config_rejected" and detail["field"] == "preregistration"
        )

    @pytest.mark.parametrize(
        "criterion,passed",
        [
            ({"metric": "val_accuracy", "threshold": 0.0}, True),
            ({"metric": "val_loss", "threshold": 0.0}, False),
        ],
    )
    def test_a_finished_run_is_judged_on_its_stored_criterion(
        self, client: TestClient, criterion: dict[str, Any], passed: bool
    ) -> None:
        status, checkpoint = _run(client, {**_FAST_RUN, "preregistration": criterion})
        stored = checkpoint["config"]["preregistration"]
        verdict = status["preregistration_verdict"]
        assert verdict["preregistration_sha256"] == stored["sha256"]
        assert verdict["metric"] == criterion["metric"] and verdict["passed"] is passed
        rounded = status["final_metrics"][criterion["metric"]]
        assert abs(verdict["observed"] - rounded) <= 1e-4
        assert verdict["threshold"] == float(criterion["threshold"])

    def test_a_run_without_a_criterion_has_no_verdict(self, client: TestClient) -> None:
        status, checkpoint = _run(client, _FAST_RUN)
        assert status["preregistration_verdict"] is None
        assert "preregistration" not in checkpoint["config"]
