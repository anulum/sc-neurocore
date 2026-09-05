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

from tests.studio_endpoints_support import *  # noqa: F403


class TestCodegenEndpoint:
    def test_codegen_model(self, client):
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

    def test_codegen_ode(self, client):
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

    def test_codegen_requires_the_mode_discriminator(self, client):
        r = client.post("/api/codegen", json={"model_name": MODEL, "duration": 10})
        assert r.status_code == 422

    def test_codegen_rejects_a_field_of_the_other_branch(self, client):
        r = client.post(
            "/api/codegen",
            json={"mode": "model", "model_name": MODEL, "equations": ["dv/dt = I"]},
        )
        assert r.status_code == 422

    def test_codegen_refuses_an_experiment_the_contract_rejects(self, client):
        r = client.post(
            "/api/codegen",
            json={"mode": "model", "model_name": MODEL, "params": {"not_a_parameter": 1.0}},
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["error"] in {"invalid_model_input", "experiment_rejected"}

    def test_codegen_refuses_an_unknown_model_instead_of_exporting_it(self, client):
        r = client.post(
            "/api/codegen",
            json={"mode": "model", "model_name": "NoSuchNeuronExistsHere"},
        )
        assert r.status_code == 422


class TestReplayPackEndpoint:
    def test_the_pack_seals_the_experiment_and_its_expectation(self, client):
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
        assert pack["schema_version"] == "studio.replay-pack.v1"
        assert len(pack["experiment_identity_sha256"]) == 64
        assert pack["expectation"]["n_steps"] == 1000
        assert pack["expectation"]["spikes"] == sorted(pack["expectation"]["spikes"])
        assert pack["environment"]["package_version"]
        assert "replay_pack" in pack["runner"]["command"]
        # A saved pack is a JSON document: it must survive the round trip the
        # browser download and the Python runner put it through.
        assert json.loads(json.dumps(pack)) == pack

    def test_the_pack_matches_the_simulation_the_same_request_produces(self, client):
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

    def test_the_pack_endpoint_refuses_a_rejected_experiment(self, client):
        r = client.post(
            "/api/export/replay-pack",
            json={"mode": "model", "model_name": MODEL, "dt": 0.1, "duration": 0.01},
        )
        assert r.status_code == 422
