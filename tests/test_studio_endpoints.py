# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio endpoint coverage (network, analysis, codegen)

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

MODEL = "AdExNeuron"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


class TestModelEndpoints:
    def test_list_models(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 100

    def test_model_detail(self, client):
        r = client.get(f"/api/models/{MODEL}")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == MODEL
        assert "params" in data
        assert "state_vars" in data
        assert "category" in data

    def test_model_detail_not_found(self, client):
        r = client.get("/api/models/NonexistentModel")
        assert r.status_code == 404

    def test_simulate_model(self, client):
        r = client.post(
            "/api/models/simulate",
            json={
                "name": MODEL,
                "duration": 50.0,
                "current": 10.0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "time" in data
        assert "states" in data


class TestNetworkEndpoint:
    def test_network_default(self, client):
        r = client.post("/api/network/ei", json={})
        assert r.status_code == 200
        data = r.json()
        assert "spike_times" in data
        assert "spike_neurons" in data
        assert "rate_time" in data
        assert data["n_total"] == 100

    def test_network_custom(self, client):
        r = client.post(
            "/api/network/ei",
            json={
                "n_exc": 40,
                "n_inh": 10,
                "duration": 50.0,
                "ext_rate": 20.0,
                "w_ee": 0.05,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_exc"] == 40
        assert data["n_inh"] == 10
        assert data["n_total"] == 50


class TestFICurveEndpoint:
    def test_fi_curve_model(self, client):
        r = client.post(
            "/api/fi-curve",
            json={
                "model_name": MODEL,
                "duration": 30.0,
                "i_min": 0,
                "i_max": 20,
                "i_steps": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "currents" in data
        assert "rates" in data
        assert len(data["currents"]) == 3


class TestSensitivityEndpoint:
    def test_sensitivity_model(self, client):
        r = client.post(
            "/api/sensitivity",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "current": 10.0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "base_rate" in data
        assert "sensitivities" in data


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
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "script" in data
        assert "oneliner" in data
        assert MODEL in data["script"]

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
        assert "script" in data


class TestPresetsEndpoint:
    def test_list_presets(self, client):
        r = client.get("/api/presets")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 10

    def test_get_preset(self, client):
        r = client.get("/api/presets")
        presets = r.json()
        first = presets[0]
        r2 = client.get(f"/api/presets/{first['id']}")
        assert r2.status_code == 200
        data = r2.json()
        assert "id" in data


class TestBifurcationEndpoint:
    def test_bifurcation_model(self, client):
        r = client.post(
            "/api/bifurcation",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "current": 10.0,
                "params": {"v_rest": -65.0},
                "sweep_param": "v_rest",
                "sweep_min": -75,
                "sweep_max": -55,
                "sweep_steps": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "param_values" in data
        assert "attractors" in data


class TestCharacterizeEndpoint:
    def test_characterize_model(self, client):
        r = client.post(
            "/api/characterize",
            json={
                "name": MODEL,
                "dt": 0.5,
                "duration": 20.0,
                "current": 10.0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "pattern" in data
        assert "fi_curve" in data
        assert "top_sensitivities" in data


class TestFreqResponseEndpoint:
    def test_freq_response(self, client):
        r = client.post(
            "/api/freq-response",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "current": 10.0,
                "amplitude": 10,
                "freq_min": 1,
                "freq_max": 50,
                "n_freqs": 3,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "frequencies_hz" in data
        assert "rates" in data


class TestMultiSimulate:
    def test_multi_simulate(self, client):
        r = client.post(
            "/api/multi-simulate",
            json=[
                {"name": MODEL, "duration": 20, "current": 10},
                {"name": "ChayNeuron", "duration": 20, "current": 10},
            ],
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert all("time" in d for d in data)


class TestCacheStats:
    def test_cache_stats(self, client):
        r = client.get("/api/cache/stats")
        assert r.status_code == 200
        data = r.json()
        assert "hits" in data
        assert "misses" in data
        assert "size" in data
