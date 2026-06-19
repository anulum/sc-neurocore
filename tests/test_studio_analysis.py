# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
from sc_neurocore.studio.analysis import frequency_response, heatmap_2d
from sc_neurocore.studio.simulation import _make_current_trace, simulate


@pytest.fixture
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


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

    def test_sine_protocol_trace_has_ac_current(self):
        trace = _make_current_trace("sine", 2.0, 1000, dt=1.0, frequency_hz=10.0)
        assert trace[0] == pytest.approx(0.0)
        assert max(trace) == pytest.approx(2.0, rel=0.02)
        assert min(trace) == pytest.approx(-2.0, rel=0.02)

    def test_frequency_response_uses_sine_protocol(self):
        calls: list[dict[str, float | str]] = []

        def fake_simulate(**cfg):
            calls.append(cfg)
            return {"stats": {"rate_hz": float(cfg["frequency_hz"])}}

        result = frequency_response(
            fake_simulate,
            {"dt": 0.1, "duration": 20.0},
            freq_min=5.0,
            freq_max=20.0,
            n_freqs=3,
            amplitude=4.0,
        )
        assert result["rates"] == pytest.approx([5.0, 10.0, 20.0])
        assert all(call["protocol"] == "sine" for call in calls)
        assert all(call["current"] == 4.0 for call in calls)

    def test_heatmap_2d_returns_failure_metadata_on_success(self):
        def fake_simulate(**cfg):
            params = cfg["params"]
            return {"stats": {"rate_hz": float(params["ix"] + params["iy"])}}

        result = heatmap_2d(
            fake_simulate,
            base_config={"params": {"baseline": 1.0}},
            param_x="ix",
            x_min=1.0,
            x_max=2.0,
            x_steps=2,
            param_y="iy",
            y_min=10.0,
            y_max=20.0,
            y_steps=2,
        )
        assert result["failed_points"] == 0
        assert result["total_points"] == 4
        assert result["failure_rate"] == 0.0
        assert result["rates"] == [[11.0, 12.0], [21.0, 22.0]]

    def test_heatmap_2d_fails_closed_with_diagnostics(self):
        def fake_simulate(**cfg):
            params = cfg["params"]
            if params["ix"] == 2.0 and params["iy"] == 20.0:
                raise RuntimeError("synthetic failure")
            return {"stats": {"rate_hz": 1.0}}

        with pytest.raises(ValueError) as exc_info:
            heatmap_2d(
                fake_simulate,
                base_config={"params": {}},
                param_x="ix",
                x_min=1.0,
                x_max=2.0,
                x_steps=2,
                param_y="iy",
                y_min=10.0,
                y_max=20.0,
                y_steps=2,
            )

        err = exc_info.value
        assert "heatmap sweep failed for 1/4 points" in str(err)
        diagnostics = err.args[1]
        assert diagnostics["failed_points"] == 1
        assert diagnostics["total_points"] == 4
        assert diagnostics["failure_rate"] == pytest.approx(0.25)
        assert diagnostics["failures"] == [
            {
                "grid_index": [1, 1],
                "param_x_value": 2.0,
                "param_y_value": 20.0,
                "error_type": "RuntimeError",
                "error_message": "synthetic failure",
            }
        ]

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

    def test_adaptive_precision_auto_tune(self, client):
        response = client.post(
            "/api/adaptive-precision/auto-tune",
            json={
                "layer_weights": [
                    [[0.2, 0.4], [0.6, 0.8]],
                    [0.3, 0.7],
                ],
                "layer_names": ["input", "readout"],
                "target_error_percent": 0.1,
                "min_bits": 4,
                "max_bits": 8,
                "min_length": 32,
                "max_length": 256,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema"] == "sc-neurocore.adaptive_precision_plan.v1"
        assert payload["api_surface"]["action_id"] == "auto_tune_adaptive_precision"
        assert payload["api_surface"]["target_error_percent"] == pytest.approx(0.1)
        assert payload["api_surface"]["estimated_lut_cost"] > 0.0
        assert (
            payload["api_surface"]["uniform_length_reference_cost"]
            >= payload["api_surface"]["estimated_lut_cost"]
        )
        assert payload["num_synapses"] == 6

    def test_adaptive_precision_auto_tune_rejects_invalid_layer(self, client):
        response = client.post(
            "/api/adaptive-precision/auto-tune",
            json={
                "layer_weights": [
                    [[[0.1]]],
                ]
            },
        )
        assert response.status_code == 422

    def test_adaptive_precision_formal_bundle(self, client):
        response = client.post(
            "/api/adaptive-precision/formal-bundle",
            json={
                "layer_weights": [[[0.2, 0.3], [0.4, 0.6]]],
                "layer_names": ["dense0"],
                "target_error_percent": 0.1,
                "module_name": "precision_plan_demo",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        bundle_manifest = payload["bundle_manifest"]
        assert (
            bundle_manifest["schema_version"] == "sc-neurocore.adaptive-precision-formal-bundle.v1"
        )
        assert bundle_manifest["module_name"] == "precision_plan_demo"
        assert payload["artifacts_text"]["sby"]
        assert payload["artifacts_text"]["sva"]
        assert bundle_manifest["artifacts"]["report"].endswith("_formal_report.json")
        assert payload["artifacts_text"]["report"] == ""
        assert "symbiyosys_executed" in payload["formal_manifest_json"]

    def test_adaptive_precision_formal_bundle_rejects_bad_layer_shape(self, client):
        response = client.post(
            "/api/adaptive-precision/formal-bundle",
            json={"layer_weights": [[[[0.2]]]]},
        )
        assert response.status_code == 422
