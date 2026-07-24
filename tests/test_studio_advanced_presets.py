# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio advanced presets

"""Focused suite: TestPresets from former test_studio_advanced.py."""

from __future__ import annotations

from tests.studio_advanced_support import *  # noqa: F403


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
