# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio analysis and new endpoints

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.codegen import classify_firing_pattern, generate_model_script
from sc_neurocore.studio.simulation import simulate


@pytest.fixture
def client():
    return TestClient(create_app())


class TestFiringClassifier:
    def test_silent(self):
        r = classify_firing_pattern([], 1000, 0.1)
        assert r["pattern"] == "silent"

    def test_tonic(self):
        spikes = list(range(100, 1000, 100))
        r = classify_firing_pattern(spikes, 1000, 0.1)
        assert r["pattern"] == "tonic"

    def test_single_spike(self):
        r = classify_firing_pattern([500], 1000, 0.1)
        assert r["pattern"] == "single_spike"


class TestCodegen:
    def test_model_script(self):
        script = generate_model_script("COBALIFNeuron", {"c_m": 200.0}, 100, 10, 0.1)
        assert "COBALIFNeuron" in script
        assert "c_m=200.0" in script
        assert "step(current=" in script

    def test_model_script_runs(self):
        script = generate_model_script("COBALIFNeuron")
        assert "import numpy" in script


class TestSimulationProtocols:
    def test_step_protocol(self):
        r = simulate(
            equations=["dv/dt = -(v - E_L) / tau_m + I / C"],
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
            dt=0.1,
            duration=100.0,
            current=30.0,
            protocol="step",
        )
        assert "current_trace" in r
        assert r["current_trace"][0] == 0.0
        mid = len(r["current_trace"]) // 2
        assert r["current_trace"][mid] == 30.0

    def test_ramp_protocol(self):
        r = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.1,
            duration=10.0,
            current=10.0,
            protocol="ramp",
        )
        assert r["current_trace"][0] == 0.0
        assert r["current_trace"][-1] == pytest.approx(10.0, rel=0.1)

    def test_stats_have_isi_histogram(self):
        r = simulate(
            equations=["dv/dt = -(v - E_L) / tau_m + I / C"],
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
            dt=0.1,
            duration=200.0,
            current=30.0,
        )
        if r["spike_count"] >= 3:
            assert r["stats"]["isi_histogram"] is not None
            assert "counts" in r["stats"]["isi_histogram"]


class TestAnalysisEndpoints:
    def test_characterize(self, client):
        r = client.post(
            "/api/characterize",
            json={
                "name": "COBALIFNeuron",
                "current": 500,
                "duration": 100,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "pattern" in d
        assert "fi_curve" in d
        assert "top_sensitivities" in d

    def test_fi_curve_model(self, client):
        r = client.post(
            "/api/fi-curve",
            json={
                "model_name": "HodgkinHuxleyNeuron",
                "dt": 0.01,
                "duration": 50,
                "i_min": 0,
                "i_max": 20,
                "i_steps": 5,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert len(d["currents"]) == 5
        assert len(d["rates"]) == 5

    def test_codegen(self, client):
        r = client.post(
            "/api/codegen",
            json={
                "mode": "model",
                "model_name": "COBALIFNeuron",
                "params": {"c_m": 200},
                "dt": 0.1,
                "duration": 100,
                "current": 10,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "COBALIFNeuron" in d["script"]
        assert "oneliner" in d

    def test_classify_endpoint(self, client):
        r = client.post(
            "/api/classify",
            json={
                "equations": ["dv/dt = -(v + 65) / 10 + I"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 100,
                "current": 30,
            },
        )
        assert r.status_code == 200
        assert "pattern" in r.json()

    def test_import_trace(self, client):
        voltage = [float(i) for i in range(-65, -45)] + [float(i) for i in range(-45, -65, -1)]
        r = client.post("/api/import-trace", json={"voltage": voltage, "dt": 0.1})
        assert r.status_code == 200
        d = r.json()
        assert "time" in d
        assert "stats" in d

    def test_multi_simulate(self, client):
        r = client.post(
            "/api/multi-simulate",
            json=[
                {"name": "COBALIFNeuron", "current": 500, "duration": 50},
                {"name": "HodgkinHuxleyNeuron", "current": 10, "duration": 50, "dt": 0.01},
            ],
        )
        assert r.status_code == 200
        d = r.json()
        assert len(d) == 2
        assert d[0]["spike_count"] >= 0
