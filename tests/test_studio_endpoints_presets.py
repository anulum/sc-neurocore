# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints presets

"""Focused suite: TestPresetsEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403


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
