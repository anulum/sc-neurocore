# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training configuration fidelity

"""The Studio trains what was asked for, or refuses before it starts.

Measured through the public HTTP surface before the contract existed: a request
for hidden widths ``[128, 64]`` on ``cifar10`` with surrogate
``not_a_real_surrogate`` completed successfully, and its exported checkpoint
recorded that request beside the architecture ``64->128->128->10`` — 128 twice,
64 nowhere, on synthetic data, with the default surrogate, all sealed under a
``config_sha256``. These cases hold the three halves of the repair: the widths
are honoured, the unsupported names are refused, and a run is replayable from
the seed the checkpoint records.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.training_contract import (
    SUPPORTED_DATASETS,
    SUPPORTED_SURROGATES,
    TRAINING_CONFIG_SCHEMA_VERSION,
    ResolvedTrainingConfig,
    TrainingConfigError,
    resolve_training_config,
)

pytest.importorskip("torch")

#: A run small enough to finish in a test and large enough to be a real run.
_FAST_RUN = {"dataset": "synthetic", "epochs": 1, "batch_size": 32, "timesteps": 4}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A Studio client for the training routes."""
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _run(client: TestClient, config: dict[str, Any]) -> dict[str, Any]:
    """Start a training run and wait for its terminal state."""
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
    checkpoint: dict[str, Any] = client.get(f"/api/training/checkpoint/{job_id}").json()
    return checkpoint


class TestResolution:
    def test_each_requested_width_is_kept_in_order(self) -> None:
        resolved = resolve_training_config({"hidden": [128, 64, 32]})

        assert resolved.hidden_widths == (128, 64, 32)
        assert resolved.architecture(64, 10) == "64->128->64->32->10"

    def test_no_hidden_layer_is_a_request_not_a_mistake(self) -> None:
        """``hidden: []`` has always built the direct input-to-output layer."""
        resolved = resolve_training_config({"hidden": []})

        assert resolved.hidden_widths == ()
        assert resolved.architecture(64, 10) == "64->10"

    def test_a_resolved_configuration_resolves_again_unchanged(self) -> None:
        """It is submitted to the runner as-is, so it has to round trip."""
        resolved = resolve_training_config({"hidden": [16], "seed": 5})

        assert resolve_training_config(resolved.to_public_dict()) == resolved

    def test_the_defaults_are_a_complete_runnable_configuration(self) -> None:
        resolved = resolve_training_config({})

        assert isinstance(resolved, ResolvedTrainingConfig)
        assert resolved.dataset in SUPPORTED_DATASETS
        assert resolved.surrogate in SUPPORTED_SURROGATES

    @pytest.mark.parametrize(
        ("payload", "field"),
        [
            ({"dataset": "cifar10"}, "dataset"),
            ({"dataset": 7}, "dataset"),
            ({"surrogate": "not_a_real_surrogate"}, "surrogate"),
            ({"hidden": [0]}, "hidden"),
            ({"hidden": [8, -3]}, "hidden"),
            ({"hidden": [8, True]}, "hidden"),
            ({"hidden": "128,64"}, "hidden"),
            ({"epochs": 0}, "epochs"),
            ({"epochs": 1.5}, "epochs"),
            ({"batch_size": -1}, "batch_size"),
            ({"timesteps": 0}, "timesteps"),
            ({"lr": 0.0}, "lr"),
            ({"lr": float("inf")}, "lr"),
            ({"max_grad_norm": -1.0}, "max_grad_norm"),
            ({"learn_beta": "yes"}, "learn_beta"),
            ({"seed": -1}, "seed"),
            ({"hiddens": [8]}, "config"),
            ({"lr": "fast"}, "lr"),
            ({"max_grad_norm": "none"}, "max_grad_norm"),
            ({"seed": "random"}, "seed"),
            ({"schema_version": "studio.training-config.v99"}, "schema_version"),
        ],
    )
    def test_an_unrunnable_request_is_refused_by_field(
        self, payload: dict[str, Any], field: str
    ) -> None:
        with pytest.raises(TrainingConfigError) as raised:
            resolve_training_config(payload)

        assert raised.value.field == field
        detail = raised.value.to_public_detail()
        assert detail["error"] == "training_config_rejected"
        assert detail["schema_version"] == TRAINING_CONFIG_SCHEMA_VERSION

    def test_a_closed_set_refusal_names_what_is_supported(self) -> None:
        """A caller must not have to guess which of eleven fields is wrong."""
        with pytest.raises(TrainingConfigError) as raised:
            resolve_training_config({"dataset": "cifar10"})

        assert raised.value.supported == SUPPORTED_DATASETS
        assert "synthetic" in str(raised.value)

    def test_a_request_that_is_not_an_object_is_refused(self) -> None:
        """A list or a string never reached a field check."""
        with pytest.raises(TrainingConfigError, match="must be an object"):
            resolve_training_config([1, 2, 3])  # type: ignore[arg-type]

    def test_zero_gradient_clipping_stays_available(self) -> None:
        """Clipping to zero runs the loop without learning; it is not an error."""
        assert resolve_training_config({"max_grad_norm": 0.0}).max_grad_norm == 0.0


class TestHttpSurface:
    def test_an_unsupported_dataset_is_refused_before_a_job_exists(
        self, client: TestClient
    ) -> None:
        response = client.post("/api/training/start", json={**_FAST_RUN, "dataset": "cifar10"})

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "training_config_rejected"
        assert detail["field"] == "dataset"
        assert detail["supported"] == list(SUPPORTED_DATASETS)
        assert "job_id" not in response.json()

    def test_an_unknown_surrogate_is_refused(self, client: TestClient) -> None:
        response = client.post(
            "/api/training/start", json={**_FAST_RUN, "surrogate": "not_a_real_surrogate"}
        )

        assert response.status_code == 422
        assert response.json()["detail"]["field"] == "surrogate"

    def test_the_capability_routes_offer_only_what_the_runner_accepts(
        self, client: TestClient
    ) -> None:
        """A name the Studio advertises must not be one it would reject."""
        offered = {row["name"] for row in client.get("/api/training/surrogates").json()}

        assert offered == set(SUPPORTED_SURROGATES)
        for name in offered:
            assert resolve_training_config({"surrogate": name}).surrogate == name

    def test_requested_widths_reach_the_checkpoint(self, client: TestClient) -> None:
        """The acceptance case: [128, 64] builds that network, never [128, 128]."""
        checkpoint = _run(client, {**_FAST_RUN, "hidden": [128, 64]})

        assert checkpoint["weight_checkpoint"]["architecture"] == "64->128->64->10"
        assert checkpoint["config"]["hidden"] == [128, 64]

    def test_the_checkpoint_records_the_configuration_that_ran(self, client: TestClient) -> None:
        """The sealed config block and the built network cannot disagree.

        Before the contract, the block recorded the request and the
        architecture recorded something else, both under one digest.
        """
        checkpoint = _run(client, {**_FAST_RUN, "hidden": [32, 16]})

        config = checkpoint["config"]
        widths = "->".join(str(width) for width in config["hidden"])
        assert widths in checkpoint["weight_checkpoint"]["architecture"]
        assert config["schema_version"] == TRAINING_CONFIG_SCHEMA_VERSION
        assert config["seed"] == 0

    def test_a_seeded_run_replays_and_a_different_seed_does_not(self, client: TestClient) -> None:
        """The seed the checkpoint records is the seed that produced the run."""
        first = _run(client, {**_FAST_RUN, "hidden": [16], "seed": 7})
        again = _run(client, {**_FAST_RUN, "hidden": [16], "seed": 7})
        other = _run(client, {**_FAST_RUN, "hidden": [16], "seed": 8})

        assert first["final_metrics"] == again["final_metrics"]
        assert first["config"]["seed"] == 7
        assert other["final_metrics"] != first["final_metrics"]
