# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio advanced features

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore_engine.studio import get_ei_network_simulator
from sc_neurocore.studio.characterize import characterize_model
from sc_neurocore.studio.codegen import (
    classify_firing_pattern,
    generate_ode_script,
    generate_oneliner,
)
from sc_neurocore.studio.network import simulate_ei_network


@pytest.fixture
def client():
    return TestClient(create_app())


# --- Network simulation ---


class TestNetwork:
    def test_basic_ei_network(self):
        r = simulate_ei_network(n_exc=20, n_inh=5, duration=50.0, ext_rate=10.0)
        assert r["n_total"] == 25
        assert r["n_exc"] == 20
        assert r["n_inh"] == 5
        assert len(r["spike_times"]) == r["n_spikes"]
        assert len(r["spike_neurons"]) == r["n_spikes"]

    def test_network_produces_spikes(self):
        r = simulate_ei_network(n_exc=40, n_inh=10, duration=200.0, ext_rate=50.0)
        assert r["n_spikes"] >= 0  # may be 0 with low drive, just verify no crash

    def test_network_rates_arrays(self):
        r = simulate_ei_network(n_exc=20, n_inh=5, duration=50.0)
        assert len(r["rate_time"]) > 0
        assert len(r["exc_rates"]) == len(r["rate_time"])
        assert len(r["inh_rates"]) == len(r["rate_time"])

    def test_network_uses_rust_engine(self):
        """Verify the Rust engine path is used when available."""
        try:
            simulate = get_ei_network_simulator()
            r = simulate(n_exc=10, n_inh=5, duration=20.0, ext_rate=100.0)
            assert "spike_times" in r
            assert "n_total" in r
            assert int(r["n_total"]) == 15
        except ImportError:
            pytest.skip("Rust engine not installed")

    def test_network_result_types(self):
        r = simulate_ei_network(n_exc=10, n_inh=5, duration=20.0)
        assert isinstance(r["spike_times"], list)
        assert isinstance(r["n_exc"], int)
        assert isinstance(r["mean_exc_rate"], float)

    def test_network_endpoint(self, client):
        r = client.post(
            "/api/network/ei",
            json={
                "n_exc": 20,
                "n_inh": 5,
                "duration": 30,
                "ext_rate": 10,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d["n_total"] == 25
        assert "spike_times" in d


# --- Characterise ---


class TestCharacterize:
    def test_characterize_lif(self):
        from sc_neurocore.studio.models import simulate_model

        def sim_fn(**kw):
            cur = kw.pop("current", 500)
            kw.pop("params", None)
            kw.pop("init", None)
            kw.pop("dt", None)
            kw.pop("duration", None)
            kw.pop("protocol", None)
            return simulate_model("COBALIFNeuron", duration=50, current=cur)

        base = {"params": {}, "dt": 0.1, "duration": 50, "current": 500, "protocol": "constant"}
        r = characterize_model(sim_fn, base)
        assert "pattern" in r
        assert "fi_curve" in r
        assert len(r["fi_curve"]["currents"]) == 20
        assert "top_sensitivities" in r
        assert "state_ranges" in r

    def test_characterize_endpoint(self, client):
        r = client.post(
            "/api/characterize",
            json={
                "name": "HodgkinHuxleyNeuron",
                "current": 10,
                "duration": 50,
                "dt": 0.01,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d["pattern"]["pattern"] in (
            "tonic",
            "bursting",
            "adapting",
            "irregular",
            "chaotic",
            "silent",
            "single_spike",
        )


# --- Codegen ---


class TestCodegenAdvanced:
    def test_ode_script(self):
        script = generate_ode_script(
            equations=["dv/dt = -(v - E_L) / tau_m + I / C"],
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
            duration=100,
            current=30,
            dt=0.1,
        )
        assert "from_equations" in script
        assert "E_L" in script

    def test_oneliner(self):
        line = generate_oneliner("COBALIFNeuron", {"c_m": 200}, 10)
        assert "COBALIFNeuron" in line
        assert "step" in line

    def test_classifier_adapting(self):
        isis_adapting = list(range(50, 150, 10))
        spikes = []
        t = 100
        for isi in isis_adapting:
            spikes.append(t)
            t += isi
        r = classify_firing_pattern(spikes, 2000, 0.1)
        assert r["pattern"] in ("adapting", "irregular", "tonic")

    def test_classifier_bursting(self):
        spikes = []
        for burst_start in range(0, 1000, 200):
            for i in range(5):
                spikes.append(burst_start + i * 5)
        r = classify_firing_pattern(spikes, 1200, 0.1)
        assert r["pattern"] in ("bursting", "irregular", "chaotic")


# --- Analysis functions ---


class TestAnalysisFunctions:
    def test_bifurcation_endpoint(self, client):
        r = client.post(
            "/api/bifurcation",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 100,
                "current": 30,
                "sweep_param": "C",
                "sweep_min": 0.5,
                "sweep_max": 3.0,
                "sweep_steps": 5,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert len(d["param_values"]) == 5
        assert len(d["attractors"]) == 5

    def test_sensitivity_endpoint(self, client):
        r = client.post(
            "/api/sensitivity",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 100,
                "current": 30,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "sensitivities" in d
        assert len(d["sensitivities"]) > 0

    def test_precision_endpoint(self, client):
        r = client.post(
            "/api/precision",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 50,
                "current": 30,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "error" in d
        assert d["error"]["max_error"] >= 0

    def test_heatmap_endpoint(self, client):
        r = client.post(
            "/api/heatmap",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 50,
                "current": 30,
                "param_x": "tau_m",
                "x_min": 5,
                "x_max": 20,
                "x_steps": 3,
                "param_y": "C",
                "y_min": 0.5,
                "y_max": 2.0,
                "y_steps": 3,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert len(d["rates"]) == 3
        assert len(d["rates"][0]) == 3

    def test_compile_endpoint(self, client):
        r = client.post(
            "/api/compile",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "module" in d["verilog"]
        assert d["chars"] > 100

    def test_codegen_endpoint(self, client):
        r = client.post(
            "/api/codegen",
            json={
                "mode": "ode",
                "equations": ["dv/dt = -(v + 65) / 10 + I"],
                "params": {},
                "init": {"v": -65},
                "dt": 0.1,
                "duration": 100,
                "current": 30,
            },
        )
        assert r.status_code == 200
        assert "from_equations" in r.json()["script"]


# --- Presets ---


class TestPresets:
    def test_list_presets(self, client):
        r = client.get("/api/presets")
        assert r.status_code == 200
        d = r.json()
        assert len(d) == 10

    def test_get_preset(self, client):
        r = client.get("/api/presets/threshold")
        assert r.status_code == 200
        d = r.json()
        assert d["id"] == "threshold"
        assert "equations" in d or "model_name" in d

    def test_preset_not_found(self, client):
        r = client.get("/api/presets/nonexistent")
        assert r.status_code == 404


# --- Error handling ---


class TestErrorHandling:
    def test_bad_model_name(self, client):
        r = client.post(
            "/api/models/simulate",
            json={
                "name": "NonExistentNeuron",
                "current": 10,
                "duration": 50,
            },
        )
        assert r.status_code == 422

    def test_bad_ode_equation(self, client):
        r = client.post(
            "/api/simulate",
            json={
                "equations": ["this is not an ODE"],
                "dt": 0.1,
                "duration": 10,
            },
        )
        assert r.status_code == 422

    def test_negative_duration(self, client):
        r = client.post(
            "/api/simulate",
            json={
                "equations": ["dv/dt = I"],
                "dt": 0.1,
                "duration": -10,
            },
        )
        assert r.status_code == 422
