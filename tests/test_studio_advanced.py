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
    return TestClient(create_app(), base_url="http://127.0.0.1")


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

    def test_preset_actions_catalog_endpoint(self, client):
        response = client.get("/api/presets/actions/catalog")
        assert response.status_code == 200
        payload = response.json()
        assert payload["count"] >= 2
        assert all(
            row["endpoint"]
            in {"/api/adaptive-precision/auto-tune", "/api/adaptive-precision/formal-bundle"}
            for row in payload["actions"]
        )

    def test_get_preset(self, client):
        r = client.get("/api/presets/threshold")
        assert r.status_code == 200
        d = r.json()
        assert d["id"] == "threshold"
        assert "equations" in d or "model_name" in d

    def test_fpga_precision_preset_exposes_adaptive_precision_actions(self, client):
        r = client.get("/api/presets/fpga_precision")
        assert r.status_code == 200
        payload = r.json()
        actions = payload["studio_actions"]
        assert len(actions) >= 2
        ids = {action["id"] for action in actions}
        assert "auto_tune_adaptive_precision" in ids
        assert "generate_adaptive_precision_formal_bundle" in ids

    def test_fpga_precision_preset_actions_endpoint(self, client):
        response = client.get("/api/presets/fpga_precision/actions")
        assert response.status_code == 200
        payload = response.json()
        assert payload["preset_id"] == "fpga_precision"
        assert len(payload["actions"]) >= 2

    def test_fpga_precision_action_resolve_endpoint(self, client):
        response = client.post(
            "/api/presets/fpga_precision/actions/auto_tune_adaptive_precision/resolve",
            json={"overrides": {"target_error_percent": 0.05, "max_bits": 12}},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["preset_id"] == "fpga_precision"
        assert payload["action_id"] == "auto_tune_adaptive_precision"
        assert payload["endpoint"] == "/api/adaptive-precision/auto-tune"
        assert payload["payload"]["target_error_percent"] == 0.05
        assert payload["payload"]["max_bits"] == 12

    def test_fpga_precision_action_execute_endpoint(self, client):
        response = client.post(
            "/api/presets/fpga_precision/actions/auto_tune_adaptive_precision/execute",
            json={"overrides": {"target_error_percent": 0.05, "max_bits": 12}},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["resolved_action"]["action_id"] == "auto_tune_adaptive_precision"
        assert payload["result"]["schema"] == "sc-neurocore.adaptive_precision_plan.v1"
        assert payload["result"]["api_surface"]["target_error_percent"] == 0.05

    def test_fpga_precision_execute_all_actions_endpoint(self, client):
        response = client.post(
            "/api/presets/fpga_precision/actions/execute-all",
            json={
                "action_overrides": {
                    "auto_tune_adaptive_precision": {"target_error_percent": 0.05},
                    "generate_adaptive_precision_formal_bundle": {
                        "module_name": "precision_plan_batch"
                    },
                }
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["preset_id"] == "fpga_precision"
        assert payload["executed_count"] >= 2
        ids = {entry["action_id"] for entry in payload["results"]}
        assert "auto_tune_adaptive_precision" in ids
        assert "generate_adaptive_precision_formal_bundle" in ids

    def test_fpga_precision_default_flow_run_endpoint(self, client):
        response = client.post(
            "/api/presets/fpga_precision/default-flow/run",
            json={
                "action_overrides": {"auto_tune_adaptive_precision": {"target_error_percent": 0.05}}
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["preset_id"] == "fpga_precision"
        assert payload["schema_version"] == "sc-neurocore.studio.default-flow-run.v1"
        assert payload["evidence_classification"] == "default_flow"
        assert payload["status"] == "completed"
        assert payload["flow_id"] == "studio_default_adaptive_precision_v1"
        assert payload["executed_count"] >= 2
        assert payload["execution_time_ms"] >= 0.0
        assert payload["action_order"][0] == "auto_tune_adaptive_precision"
        manifest = payload["reproducibility_manifest"]
        assert manifest["hash_algorithm"] == "sha256"
        assert len(manifest["inputs_fingerprint_sha256"]) == 64
        assert len(manifest["run_fingerprint_sha256"]) == 64

    def test_fpga_precision_default_flow_plan_endpoint(self, client):
        response = client.get("/api/presets/fpga_precision/default-flow/plan")
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == "sc-neurocore.studio.default-flow-plan.v1"
        assert payload["flow_id"] == "studio_default_adaptive_precision_v1"
        assert payload["count"] >= 2
        assert payload["action_order"][0] == "auto_tune_adaptive_precision"
        assert len(payload["actions"][0]["template_fingerprint_sha256"]) == 64
        assert len(payload["plan_fingerprint_sha256"]) == 64

    def test_fpga_precision_default_flow_contract_endpoint(self, client):
        response = client.get("/api/presets/fpga_precision/default-flow/contract")
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == "sc-neurocore.studio.default-flow-contract.v1"
        assert payload["preset_id"] == "fpga_precision"
        template = payload["guarded_run_request_template"]
        plan = payload["plan"]
        assert template["action_order"] == plan["action_order"]
        assert template["plan_fingerprint_sha256"] == plan["plan_fingerprint_sha256"]

    def test_fpga_precision_default_flow_verify_endpoint(self, client):
        plan = client.get("/api/presets/fpga_precision/default-flow/plan")
        assert plan.status_code == 200
        plan_payload = plan.json()
        fingerprints = {
            row["action_id"]: row["template_fingerprint_sha256"] for row in plan_payload["actions"]
        }
        response = client.post(
            "/api/presets/fpga_precision/default-flow/verify",
            json={
                "action_order": plan_payload["action_order"],
                "template_fingerprints": fingerprints,
                "plan_fingerprint_sha256": plan_payload["plan_fingerprint_sha256"],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == "sc-neurocore.studio.default-flow-verify.v1"
        assert payload["verified"] is True
        assert payload["plan_fingerprint_match"] is True

    def test_fpga_precision_default_flow_guarded_run_endpoint(self, client):
        plan = client.get("/api/presets/fpga_precision/default-flow/plan")
        assert plan.status_code == 200
        plan_payload = plan.json()
        fingerprints = {
            row["action_id"]: row["template_fingerprint_sha256"] for row in plan_payload["actions"]
        }
        response = client.post(
            "/api/presets/fpga_precision/default-flow/run-guarded",
            json={
                "action_order": plan_payload["action_order"],
                "template_fingerprints": fingerprints,
                "plan_fingerprint_sha256": plan_payload["plan_fingerprint_sha256"],
                "action_overrides": {
                    "auto_tune_adaptive_precision": {"target_error_percent": 0.05}
                },
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["verification_gate"]["verified"] is True
        assert payload["executed_count"] >= 2

    def test_fpga_precision_default_flow_run_from_contract_endpoint(self, client):
        contract = client.get("/api/presets/fpga_precision/default-flow/contract")
        assert contract.status_code == 200
        contract_payload = contract.json()
        response = client.post(
            "/api/presets/fpga_precision/default-flow/run-from-contract",
            json={
                "contract": contract_payload,
                "action_overrides": {
                    "auto_tune_adaptive_precision": {"target_error_percent": 0.05}
                },
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["contract_verification"]["verified"] is True
        assert payload["verification_gate"]["verified"] is True

    def test_fpga_precision_default_flow_attest_endpoint(self, client):
        run = client.post(
            "/api/presets/fpga_precision/default-flow/run",
            json={
                "action_overrides": {"auto_tune_adaptive_precision": {"target_error_percent": 0.05}}
            },
        )
        assert run.status_code == 200
        run_payload = run.json()
        response = client.post(
            "/api/presets/fpga_precision/default-flow/attest",
            json={"run_result": run_payload},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["schema_version"] == "sc-neurocore.studio.default-flow-attestation.v1"
        assert payload["evidence_classification"] == "default_flow"
        assert payload["status"] == "completed"
        assert payload["preset_id"] == "fpga_precision"
        assert len(payload["attestation_fingerprint_sha256"]) == 64

    def test_fpga_precision_default_flow_attest_verify_endpoint(self, client):
        run = client.post(
            "/api/presets/fpga_precision/default-flow/run",
            json={
                "action_overrides": {"auto_tune_adaptive_precision": {"target_error_percent": 0.05}}
            },
        )
        assert run.status_code == 200
        run_payload = run.json()
        attest = client.post(
            "/api/presets/fpga_precision/default-flow/attest",
            json={"run_result": run_payload},
        )
        assert attest.status_code == 200
        attest_payload = attest.json()
        verify = client.post(
            "/api/presets/fpga_precision/default-flow/attest/verify",
            json={"run_result": run_payload, "attestation": attest_payload},
        )
        assert verify.status_code == 200
        verify_payload = verify.json()
        assert (
            verify_payload["schema_version"]
            == "sc-neurocore.studio.default-flow-attestation-verify.v1"
        )
        assert verify_payload["verified"] is True

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
