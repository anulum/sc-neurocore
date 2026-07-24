# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis analysis endpoints

"""Focused suite: TestAnalysisEndpoints from former test_studio_analysis.py."""

from __future__ import annotations

from tests.studio_analysis_support import *  # noqa: F403


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
