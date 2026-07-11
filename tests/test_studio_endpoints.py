# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
from sc_neurocore.studio.platform import (
    DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS,
    DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION,
    DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS,
    StudioRuntimeSettings,
)

MODEL = "AdExNeuron"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app, base_url="http://127.0.0.1")


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

    def test_model_detail_internal_failure_is_500(self, client, monkeypatch):
        import sc_neurocore.studio.api.catalogue as catalogue_routes

        def _boom(_name: str):
            raise RuntimeError("metadata exploded")

        monkeypatch.setattr(catalogue_routes, "get_model_detail", _boom)
        r = client.get(f"/api/models/{MODEL}")
        assert r.status_code == 500
        assert r.json()["detail"] == "Internal error"

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
    def test_preset_actions_catalog(self, client):
        response = client.get("/api/presets/actions/catalog")
        assert response.status_code == 200
        payload = response.json()
        assert payload["count"] == len(payload["actions"])
        assert payload["count"] >= 2

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

    def test_fpga_precision_preset_has_studio_actions(self, client):
        response = client.get("/api/presets/fpga_precision")
        assert response.status_code == 200
        payload = response.json()
        assert "studio_actions" in payload
        assert payload["studio_actions"][0]["endpoint"].startswith("/api/adaptive-precision/")

    def test_fpga_precision_actions_endpoint(self, client):
        response = client.get("/api/presets/fpga_precision/actions")
        assert response.status_code == 200
        payload = response.json()
        assert payload["preset_id"] == "fpga_precision"
        assert isinstance(payload["actions"], list)
        assert payload["actions"][0]["endpoint"].startswith("/api/adaptive-precision/")

    def test_fpga_precision_action_resolve_rejects_unknown_override(self, client):
        response = client.post(
            "/api/presets/fpga_precision/actions/auto_tune_adaptive_precision/resolve",
            json={"overrides": {"unknown_key": 123}},
        )
        assert response.status_code == 422

    def test_fpga_precision_action_execute_rejects_unknown_override(self, client):
        response = client.post(
            "/api/presets/fpga_precision/actions/auto_tune_adaptive_precision/execute",
            json={"overrides": {"unknown_key": 123}},
        )
        assert response.status_code == 422

    def test_fpga_precision_execute_all_rejects_bad_override_shape(self, client):
        response = client.post(
            "/api/presets/fpga_precision/actions/execute-all",
            json={"action_overrides": {"auto_tune_adaptive_precision": "invalid"}},
        )
        assert response.status_code == 422

    def test_fpga_precision_default_flow_rejects_bad_override_shape(self, client):
        response = client.post(
            "/api/presets/fpga_precision/default-flow/run",
            json={"action_overrides": {"auto_tune_adaptive_precision": "invalid"}},
        )
        assert response.status_code == 422

    def test_fpga_precision_default_flow_fingerprint_stable_for_same_input(self, client):
        payload = {
            "action_overrides": {"auto_tune_adaptive_precision": {"target_error_percent": 0.05}}
        }
        first = client.post("/api/presets/fpga_precision/default-flow/run", json=payload)
        second = client.post("/api/presets/fpga_precision/default-flow/run", json=payload)
        assert first.status_code == 200
        assert second.status_code == 200
        first_body = first.json()
        second_body = second.json()
        assert (
            first_body["reproducibility_manifest"]["inputs_fingerprint_sha256"]
            == second_body["reproducibility_manifest"]["inputs_fingerprint_sha256"]
        )

    def test_default_flow_plan_not_found(self, client):
        response = client.get("/api/presets/nonexistent/default-flow/plan")
        assert response.status_code == 404

    def test_default_flow_contract_not_found(self, client):
        response = client.get("/api/presets/nonexistent/default-flow/contract")
        assert response.status_code == 404

    def test_default_flow_verify_detects_drift(self, client):
        response = client.post(
            "/api/presets/fpga_precision/default-flow/verify",
            json={
                "action_order": ["generate_adaptive_precision_formal_bundle"],
                "template_fingerprints": {"generate_adaptive_precision_formal_bundle": "0" * 64},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["order_match"] is False
        assert payload["fingerprints_match"] is False
        assert payload["verified"] is False

    def test_default_flow_verify_rejects_invalid_fingerprint_format(self, client):
        response = client.post(
            "/api/presets/fpga_precision/default-flow/verify",
            json={
                "action_order": ["auto_tune_adaptive_precision"],
                "template_fingerprints": {"auto_tune_adaptive_precision": "BAD_HASH"},
                "plan_fingerprint_sha256": "also_bad",
            },
        )
        assert response.status_code == 422

    def test_default_flow_guarded_run_rejects_drift(self, client):
        response = client.post(
            "/api/presets/fpga_precision/default-flow/run-guarded",
            json={
                "action_order": ["generate_adaptive_precision_formal_bundle"],
                "template_fingerprints": {"generate_adaptive_precision_formal_bundle": "0" * 64},
                "plan_fingerprint_sha256": "0" * 64,
            },
        )
        assert response.status_code == 422

    def test_default_flow_guarded_run_requires_plan_fingerprint(self, client):
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
            },
        )
        assert response.status_code == 422

    def test_default_flow_guarded_run_rejects_plan_fingerprint_mismatch(self, client):
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
                "plan_fingerprint_sha256": "f" * 64,
            },
        )
        assert response.status_code == 422

    def test_default_flow_guarded_run_rejects_invalid_fingerprint_format(self, client):
        response = client.post(
            "/api/presets/fpga_precision/default-flow/run-guarded",
            json={
                "action_order": ["auto_tune_adaptive_precision"],
                "template_fingerprints": {"auto_tune_adaptive_precision": "BAD_HASH"},
                "plan_fingerprint_sha256": "bad",
            },
        )
        assert response.status_code == 422

    def test_default_flow_run_from_contract_rejects_drift(self, client):
        contract = client.get("/api/presets/fpga_precision/default-flow/contract")
        assert contract.status_code == 200
        payload = contract.json()
        payload["guarded_run_request_template"]["plan_fingerprint_sha256"] = "f" * 64
        response = client.post(
            "/api/presets/fpga_precision/default-flow/run-from-contract",
            json={"contract": payload, "action_overrides": {}},
        )
        assert response.status_code == 422

    def test_default_flow_attest_rejects_missing_repro_manifest(self, client):
        response = client.post(
            "/api/presets/fpga_precision/default-flow/attest",
            json={
                "run_result": {
                    "preset_id": "fpga_precision",
                    "flow_id": "studio_default_adaptive_precision_v1",
                }
            },
        )
        assert response.status_code == 422

    def test_default_flow_attest_verify_detects_tampered_attestation(self, client):
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
        attest_payload["attestation_fingerprint_sha256"] = "0" * 64
        verify = client.post(
            "/api/presets/fpga_precision/default-flow/attest/verify",
            json={"run_result": run_payload, "attestation": attest_payload},
        )
        assert verify.status_code == 200
        verify_payload = verify.json()
        assert verify_payload["verified"] is False
        assert verify_payload["checks"]["attestation_fingerprint_match"] is False


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
                "sweep_steps": 5,
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

    def test_empty_multi_simulate_returns_empty_result(self, client):
        """An empty bounded batch is a valid no-op."""

        response = client.post("/api/multi-simulate", json=[])

        assert response.status_code == 200
        assert response.json() == []


class TestCacheStats:
    def test_cache_stats(self, client):
        r = client.get("/api/cache/stats")
        assert r.status_code == 200
        data = r.json()
        assert "hits" in data
        assert "misses" in data
        assert "size" in data

    def test_simulation_cache_reuses_and_evicts_results(self, monkeypatch):
        """The bounded simulation cache reuses hits and evicts its oldest entry."""
        import sc_neurocore.studio.api.simulation as simulation_routes

        cache = simulation_routes._SimCache(maxsize=1)
        monkeypatch.setattr(simulation_routes, "_cache", cache)
        local_client = TestClient(create_app(), base_url="http://127.0.0.1")
        payload = {
            "current": 1.0,
            "dt": 0.1,
            "duration": 1.0,
            "equations": ["dv/dt = I"],
            "init": {"v": 0.0},
        }

        first = local_client.post("/api/simulate", json=payload)
        cached = local_client.post("/api/simulate", json=payload)
        second = local_client.post("/api/simulate", json={**payload, "current": 2.0})
        evicted = local_client.post("/api/simulate", json=payload)

        assert [response.status_code for response in (first, cached, second, evicted)] == [
            200,
            200,
            200,
            200,
        ]
        assert cached.json() == first.json()
        assert cache.hits == 1
        assert cache.misses == 3
        assert len(cache._cache) == 1


def test_import_trace_rejects_empty_voltage(client: TestClient) -> None:
    """Trace import requires a non-empty voltage vector."""

    response = client.post("/api/import-trace", json={"voltage": [], "dt": 0.1})

    assert response.status_code == 422
    assert response.json()["detail"] == "Expected {voltage: [...], dt: float}"


def _budget_client(
    *,
    max_sync_analysis_steps_per_simulation: int = (
        DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION
    ),
    max_sync_analysis_total_steps: int = DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS,
    max_sync_analysis_simulations: int = DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS,
) -> TestClient:
    """Build a TestClient whose Studio app uses tightened analysis budgets."""
    settings = StudioRuntimeSettings(
        max_sync_analysis_steps_per_simulation=max_sync_analysis_steps_per_simulation,
        max_sync_analysis_total_steps=max_sync_analysis_total_steps,
        max_sync_analysis_simulations=max_sync_analysis_simulations,
    )
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


class TestAnalysisBudgetEnforcement:
    def test_heatmap_rejected_over_step_budget(self) -> None:
        client = _budget_client(
            max_sync_analysis_steps_per_simulation=1_000,
            max_sync_analysis_simulations=1_000,
        )
        r = client.post(
            "/api/heatmap",
            json={
                "model_name": MODEL,
                "duration": 200.0,
                "dt": 0.001,
                "param_x": "v_rest",
                "x_min": -75,
                "x_max": -55,
                "x_steps": 3,
                "param_y": "a",
                "y_min": 0,
                "y_max": 5,
                "y_steps": 3,
            },
        )
        assert r.status_code == 422
        assert r.json()["detail"]["limit"] == "steps_per_simulation"
        assert r.json()["detail"]["allowed"] == 1_000
        assert r.json()["detail"]["projected"] == 200_000

    def test_sensitivity_rejected_over_simulation_budget(self) -> None:
        client = _budget_client(max_sync_analysis_simulations=5)
        params = {f"p{i}": float(i + 1) for i in range(10)}
        r = client.post(
            "/api/sensitivity",
            json={"model_name": MODEL, "duration": 20.0, "params": params},
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["limit"] == "simulations"
        assert detail["projected"] == 1 + 2 * 10
        assert detail["allowed"] == 5

    def test_multi_simulate_rejected_over_simulation_budget(self) -> None:
        client = _budget_client(max_sync_analysis_simulations=1)
        r = client.post(
            "/api/multi-simulate",
            json=[
                {"name": MODEL, "duration": 20.0, "current": 10.0},
                {"name": "ChayNeuron", "duration": 20.0, "current": 10.0},
            ],
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["limit"] == "simulations"
        assert detail["projected"] == 2
        assert detail["allowed"] == 1

    def test_compare_invalid_cost_fields_reaches_payload_validation(
        self,
        client: TestClient,
    ) -> None:
        invalid_config = {
            "equations": ["dv/dt = I"],
            "init": {"v": 0.0},
            "duration": "not-a-number",
            "dt": "not-a-number",
        }
        r = client.post(
            "/api/compare",
            json={"config_a": invalid_config, "config_b": invalid_config},
        )
        assert r.status_code == 422
        assert r.json()["detail"] == "Invalid input"

    def test_bifurcation_rejected_for_non_positive_timestep(self, client: TestClient) -> None:
        r = client.post(
            "/api/bifurcation",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "dt": 0.0,
                "sweep_param": "v_rest",
                "sweep_min": -75,
                "sweep_max": -55,
                "sweep_steps": 5,
            },
        )
        assert r.status_code == 422
        assert r.json()["detail"]["limit"] == "timestep"

    def test_normal_sensitivity_within_default_budget_passes(self, client: TestClient) -> None:
        # The default budget admits ordinary analysis requests unchanged.
        r = client.post(
            "/api/sensitivity",
            json={"model_name": MODEL, "duration": 20.0, "current": 10.0},
        )
        assert r.status_code == 200
        assert "sensitivities" in r.json()


class TestAnalysisMetadataConsistency:
    def test_characterize_attaches_analysis_metadata(self, client: TestClient) -> None:
        r = client.post(
            "/api/characterize",
            json={"name": MODEL, "dt": 0.5, "duration": 20.0, "current": 10.0},
        )
        assert r.status_code == 200
        metadata = r.json()["analysis_metadata"]
        assert metadata["schema_version"] == "studio.analysis-result.v1"
        assert metadata["analysis_type"] == "characterize"
        assert metadata["source"] == "model"

    def test_multi_simulate_attaches_run_metadata_per_result(self, client: TestClient) -> None:
        r = client.post(
            "/api/multi-simulate",
            json=[
                {"name": MODEL, "duration": 20, "current": 10},
                {"name": "ChayNeuron", "duration": 20, "current": 10},
            ],
        )
        assert r.status_code == 200
        results = r.json()
        assert len(results) == 2
        for result in results:
            assert result["run_metadata"]["schema_version"] == "studio.simulation-run.v1"
            assert result["run_metadata"]["source"] == "model"
