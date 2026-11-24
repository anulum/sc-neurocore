# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints codegen

"""The export endpoints: what they return, and what they refuse."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from tests.studio_endpoints_support import (
    MODEL,
)
from tests.studio_endpoints_support import client as client


class TestCodegenEndpoint:
    """The Studio code-export endpoint."""

    def test_codegen_model(self, client: TestClient) -> None:
        """A model export pins the request and embeds the experiment hash and a replay script."""
        r = client.post(
            "/api/codegen",
            json={
                "mode": "model",
                "model_name": MODEL,
                "params": {},
                "dt": 0.1,
                "duration": 100,
                "current": 10,
                "protocol": "step",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["request"] == {
            "name": MODEL,
            "params": {},
            "dt": 0.1,
            "duration": 100.0,
            "current": 10.0,
            "protocol": "step",
            "frequency_hz": 10.0,
            "trial": "replay",
        }
        assert data["experiment_sha256"] in data["script"]
        assert "replay_pack" in data["replay_script"]
        # The script goes through the contract instead of guessing a step call.
        assert "step(current=" not in data["script"]

    def test_codegen_ode(self, client: TestClient) -> None:
        """An ODE export pins the equations and the initial state."""
        r = client.post(
            "/api/codegen",
            json={
                "mode": "ode",
                "equations": ["dv/dt = I"],
                "params": {},
                "init": {"v": 0},
                "dt": 0.1,
                "duration": 100,
                "current": 10,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["request"]["equations"] == ["dv/dt = I"]
        assert data["request"]["init"] == {"v": 0.0}
        assert "script" in data

    def test_codegen_requires_the_mode_discriminator(self, client: TestClient) -> None:
        """An export without its mode is rejected with 422."""
        r = client.post("/api/codegen", json={"model_name": MODEL, "duration": 10})
        assert r.status_code == 422

    def test_codegen_rejects_a_field_of_the_other_branch(self, client: TestClient) -> None:
        """An export with a field of the other mode is rejected with 422."""
        r = client.post(
            "/api/codegen",
            json={"mode": "model", "model_name": MODEL, "equations": ["dv/dt = I"]},
        )
        assert r.status_code == 422

    def test_codegen_refuses_an_experiment_the_contract_rejects(self, client: TestClient) -> None:
        """An export of a rejected experiment is refused with 422."""
        r = client.post(
            "/api/codegen",
            json={"mode": "model", "model_name": MODEL, "params": {"not_a_parameter": 1.0}},
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["error"] in {"invalid_model_input", "experiment_rejected"}

    def test_codegen_refuses_an_unknown_model_instead_of_exporting_it(
        self, client: TestClient
    ) -> None:
        """An export of an unknown model is refused with 422."""
        r = client.post(
            "/api/codegen",
            json={"mode": "model", "model_name": "NoSuchNeuronExistsHere"},
        )
        assert r.status_code == 422


class TestReplayPackEndpoint:
    """The Studio replay-pack export endpoint."""

    def test_pack_size_refusal_has_actionable_detail(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real export route preserves the pack refusal rather than a generic error."""
        monkeypatch.setattr("sc_neurocore.studio.replay_pack.MAX_PACK_BYTES", 16)
        response = client.post(
            "/api/export/replay-pack",
            json={
                "mode": "model",
                "model_name": MODEL,
                "duration": 2,
                "dt": 0.1,
                "current": 10,
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"] == {
            "error": "replay_refused",
            "stage": "schema",
            "reason": "replay pack exceeds the file size limit",
            "differences": [],
        }

    def test_the_pack_seals_the_experiment_and_its_expectation(self, client: TestClient) -> None:
        """The replay pack seals the experiment identity, expectation, environment and runner."""
        r = client.post(
            "/api/export/replay-pack",
            json={
                "mode": "model",
                "model_name": MODEL,
                "dt": 0.1,
                "duration": 100,
                "current": 10,
                "protocol": "step",
            },
        )
        assert r.status_code == 200
        pack = r.json()
        assert pack["schema_version"] == "studio.replay-pack.v2"
        assert len(pack["experiment_identity_sha256"]) == 64
        assert pack["expectation"]["n_steps"] == 1000
        assert pack["expectation"]["spikes"] == sorted(pack["expectation"]["spikes"])
        assert pack["environment"]["package_version"]
        assert "replay_pack" in pack["runner"]["command"]
        # A saved pack is a JSON document: it must survive the round trip the
        # browser download and the Python runner put it through.
        assert json.loads(json.dumps(pack)) == pack

    def test_the_pack_matches_the_simulation_the_same_request_produces(
        self, client: TestClient
    ) -> None:
        """The pack's expectation matches the simulation of the same request."""
        body = {
            "mode": "model",
            "model_name": MODEL,
            "dt": 0.1,
            "duration": 100,
            "current": 10,
            "protocol": "step",
        }
        pack = client.post("/api/export/replay-pack", json=body).json()
        run = client.post(
            "/api/models/simulate",
            json={key: value for key, value in body.items() if key != "mode"},
        ).json()

        assert pack["expectation"]["spikes"] == run["spikes"]
        assert pack["expectation"]["final_state"] == run["final_state"]
        assert pack["experiment_sha256"] == run["experiment"]["experiment_sha256"]

    def test_the_pack_endpoint_refuses_a_rejected_experiment(self, client: TestClient) -> None:
        """A replay pack of a rejected experiment is refused with 422."""
        r = client.post(
            "/api/export/replay-pack",
            json={"mode": "model", "model_name": MODEL, "dt": 0.1, "duration": 0.01},
        )
        assert r.status_code == 422
