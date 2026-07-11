# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio preset and default-flow routes

"""Resolve, execute, verify, and attest Studio preset action flows."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.api.adaptive_precision import (
    _execute_resolved_preset_action,
    _is_executable_preset_action_endpoint,
)
from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.preset_flows import (
    _build_default_flow_plan_payload,
    _execute_default_flow_with_overrides,
    _resolve_action_payload,
    _sha256_json,
)
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    PresetActionResolveRequest,
    PresetActionsExecuteAllRequest,
    PresetDefaultFlowAttestationVerifyRequest,
    PresetDefaultFlowAttestRequest,
    PresetDefaultFlowGuardedRunRequest,
    PresetDefaultFlowRunFromContractRequest,
    PresetDefaultFlowRunRequest,
    PresetDefaultFlowVerifyRequest,
)
from sc_neurocore.studio.presets import (
    get_preset,
    get_preset_action,
    get_preset_actions,
    list_preset_action_catalog,
    list_presets,
)


def build_presets_router(context: StudioApiContext) -> APIRouter:
    """Build the preset and default-flow router over shared Studio runtime state."""
    router = APIRouter()

    @router.get("/api/presets")
    def api_presets() -> Any:
        return list_presets()

    @router.get("/api/presets/actions/catalog")
    def api_preset_actions_catalog() -> Any:
        catalog = list_preset_action_catalog()
        executable = [
            row for row in catalog if _is_executable_preset_action_endpoint(row.get("endpoint"))
        ]
        return {"actions": executable, "count": len(executable)}

    @router.get("/api/presets/{preset_id}")
    def api_preset(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        return p

    @router.get("/api/presets/{preset_id}/actions")
    def api_preset_actions(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        return {
            "preset_id": preset_id,
            "actions": get_preset_actions(preset_id),
        }

    @router.post("/api/presets/{preset_id}/actions/{action_id}/resolve")
    def api_preset_action_resolve(
        preset_id: str, action_id: str, req: PresetActionResolveRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        action = get_preset_action(preset_id, action_id)
        if not action:
            raise HTTPException(
                status_code=404,
                detail=f"Action '{action_id}' not found for preset '{preset_id}'",
            )
        template = action.get("payload_template")
        if not isinstance(template, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Action '{action_id}' does not define a payload template",
            )
        return _safe(
            lambda: _resolve_action_payload(preset_id, action_id, action, template, req.overrides)
        )

    @router.post("/api/presets/{preset_id}/actions/{action_id}/execute")
    def api_preset_action_execute(
        preset_id: str, action_id: str, req: PresetActionResolveRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        action = get_preset_action(preset_id, action_id)
        if not action:
            raise HTTPException(
                status_code=404,
                detail=f"Action '{action_id}' not found for preset '{preset_id}'",
            )
        template = action.get("payload_template")
        if not isinstance(template, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Action '{action_id}' does not define a payload template",
            )

        def fn() -> dict[str, Any]:
            resolved = _resolve_action_payload(
                preset_id, action_id, action, template, req.overrides
            )
            result = _execute_resolved_preset_action(resolved)
            return {"resolved_action": resolved, "result": result}

        return _safe(fn)

    @router.post("/api/presets/{preset_id}/actions/execute-all")
    def api_preset_actions_execute_all(preset_id: str, req: PresetActionsExecuteAllRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            actions = get_preset_actions(preset_id)
            executable_actions = [
                action
                for action in actions
                if _is_executable_preset_action_endpoint(action.get("endpoint"))
            ]
            results: list[dict[str, Any]] = []
            for action in executable_actions:
                action_id = action.get("id")
                if not isinstance(action_id, str):
                    continue
                template = action.get("payload_template")
                if not isinstance(template, dict):
                    raise ValueError(f"action '{action_id}' does not define a payload template")
                overrides = req.action_overrides.get(action_id, {})
                resolved = _resolve_action_payload(
                    preset_id, action_id, action, template, overrides
                )
                result = _execute_resolved_preset_action(resolved)
                results.append(
                    {
                        "action_id": action_id,
                        "resolved_action": resolved,
                        "result": result,
                    }
                )
            return {
                "preset_id": preset_id,
                "executed_count": len(results),
                "results": results,
            }

        return _safe(fn)

    @router.post("/api/presets/{preset_id}/default-flow/run")
    def api_preset_default_flow_run(preset_id: str, req: PresetDefaultFlowRunRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            return _execute_default_flow_with_overrides(preset_id, req.action_overrides)

        return _safe(fn)

    @router.get("/api/presets/{preset_id}/default-flow/plan")
    def api_preset_default_flow_plan(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        return _safe(lambda: _build_default_flow_plan_payload(preset_id))

    @router.get("/api/presets/{preset_id}/default-flow/contract")
    def api_preset_default_flow_contract(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            plan = _build_default_flow_plan_payload(preset_id)
            template_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            return {
                "schema_version": "sc-neurocore.studio.default-flow-contract.v1",
                "preset_id": preset_id,
                "flow_id": plan["flow_id"],
                "plan": plan,
                "guarded_run_request_template": {
                    "action_order": plan["action_order"],
                    "template_fingerprints": template_fingerprints,
                    "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                    "action_overrides": {},
                },
            }

        return _safe(fn)

    @router.post("/api/presets/{preset_id}/default-flow/verify")
    def api_preset_default_flow_verify(preset_id: str, req: PresetDefaultFlowVerifyRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            plan = _build_default_flow_plan_payload(preset_id)
            expected_order = plan["action_order"]
            expected_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            expected_plan_fingerprint = plan["plan_fingerprint_sha256"]
            order_match = req.action_order == expected_order
            fingerprints_match = req.template_fingerprints == expected_fingerprints
            plan_fingerprint_match = req.plan_fingerprint_sha256 == expected_plan_fingerprint
            return {
                "schema_version": "sc-neurocore.studio.default-flow-verify.v1",
                "preset_id": preset_id,
                "flow_id": plan["flow_id"],
                "order_match": order_match,
                "fingerprints_match": fingerprints_match,
                "plan_fingerprint_match": plan_fingerprint_match,
                "verified": order_match and fingerprints_match and plan_fingerprint_match,
                "expected_action_order": expected_order,
                "expected_template_fingerprints": expected_fingerprints,
                "expected_plan_fingerprint_sha256": expected_plan_fingerprint,
            }

        return _safe(fn)

    @router.post("/api/presets/{preset_id}/default-flow/run-guarded")
    def api_preset_default_flow_run_guarded(
        preset_id: str, req: PresetDefaultFlowGuardedRunRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            plan = _build_default_flow_plan_payload(preset_id)
            expected_order = plan["action_order"]
            expected_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            expected_plan_fingerprint = plan["plan_fingerprint_sha256"]
            order_match = req.action_order == expected_order
            fingerprints_match = req.template_fingerprints == expected_fingerprints
            plan_fingerprint_match = req.plan_fingerprint_sha256 == expected_plan_fingerprint
            if not (order_match and fingerprints_match and plan_fingerprint_match):
                raise ValueError("default-flow plan verification failed; refresh plan before run")
            run_payload = _execute_default_flow_with_overrides(preset_id, req.action_overrides)
            run_payload["verification_gate"] = {
                "order_match": order_match,
                "fingerprints_match": fingerprints_match,
                "plan_fingerprint_match": plan_fingerprint_match,
                "verified": True,
            }
            return run_payload

        return _safe(fn)

    @router.post("/api/presets/{preset_id}/default-flow/run-from-contract")
    def api_preset_default_flow_run_from_contract(
        preset_id: str, req: PresetDefaultFlowRunFromContractRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            contract = req.contract
            if contract.get("schema_version") != "sc-neurocore.studio.default-flow-contract.v1":
                raise ValueError("unsupported contract schema version")
            if contract.get("preset_id") != preset_id:
                raise ValueError("contract preset_id mismatch")

            guarded = contract.get("guarded_run_request_template")
            if not isinstance(guarded, dict):
                raise ValueError("contract missing guarded_run_request_template")
            action_order = guarded.get("action_order")
            template_fingerprints = guarded.get("template_fingerprints")
            plan_fingerprint = guarded.get("plan_fingerprint_sha256")
            if not isinstance(action_order, list) or not isinstance(template_fingerprints, dict):
                raise ValueError("invalid guarded run template contract")
            if not isinstance(plan_fingerprint, str):
                raise ValueError("invalid plan fingerprint in contract")

            plan = _build_default_flow_plan_payload(preset_id)
            expected_order = plan["action_order"]
            expected_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            expected_plan_fingerprint = plan["plan_fingerprint_sha256"]

            order_match = action_order == expected_order
            fingerprints_match = template_fingerprints == expected_fingerprints
            plan_fingerprint_match = plan_fingerprint == expected_plan_fingerprint
            if not (order_match and fingerprints_match and plan_fingerprint_match):
                raise ValueError("contract drift detected; refresh contract before run")

            run_payload = _execute_default_flow_with_overrides(preset_id, req.action_overrides)
            run_payload["verification_gate"] = {
                "order_match": order_match,
                "fingerprints_match": fingerprints_match,
                "plan_fingerprint_match": plan_fingerprint_match,
                "verified": True,
            }
            run_payload["contract_verification"] = {
                "schema_version": "sc-neurocore.studio.default-flow-contract-verify.v1",
                "contract_schema_version": contract["schema_version"],
                "verified": True,
            }
            return run_payload

        return _safe(fn)

    @router.post("/api/presets/{preset_id}/default-flow/attest")
    def api_preset_default_flow_attest(preset_id: str, req: PresetDefaultFlowAttestRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            run_result = req.run_result
            if run_result.get("preset_id") != preset_id:
                raise ValueError("run_result preset_id mismatch")
            if run_result.get("flow_id") != "studio_default_adaptive_precision_v1":
                raise ValueError("unsupported flow_id for attestation")
            repro = run_result.get("reproducibility_manifest")
            if not isinstance(repro, dict):
                raise ValueError("run_result missing reproducibility_manifest")
            run_fingerprint = repro.get("run_fingerprint_sha256")
            inputs_fingerprint = repro.get("inputs_fingerprint_sha256")
            if not isinstance(run_fingerprint, str) or not isinstance(inputs_fingerprint, str):
                raise ValueError("run_result reproducibility fingerprint missing")
            if len(run_fingerprint) != 64 or len(inputs_fingerprint) != 64:
                raise ValueError("run_result reproducibility fingerprint must be sha256 hex")

            plan = _build_default_flow_plan_payload(preset_id)
            attestation_payload = {
                "schema_version": "sc-neurocore.studio.default-flow-attestation.v1",
                "evidence_classification": "default_flow",
                "status": "completed",
                "preset_id": preset_id,
                "flow_id": "studio_default_adaptive_precision_v1",
                "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                "inputs_fingerprint_sha256": inputs_fingerprint,
                "run_fingerprint_sha256": run_fingerprint,
            }
            attestation_payload["attestation_fingerprint_sha256"] = _sha256_json(
                attestation_payload
            )
            return attestation_payload

        return _safe(fn)

    @router.post("/api/presets/{preset_id}/default-flow/attest/verify")
    def api_preset_default_flow_attest_verify(
        preset_id: str, req: PresetDefaultFlowAttestationVerifyRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            run_result = req.run_result
            attestation = req.attestation
            if run_result.get("preset_id") != preset_id:
                raise ValueError("run_result preset_id mismatch")
            if run_result.get("flow_id") != "studio_default_adaptive_precision_v1":
                raise ValueError("unsupported flow_id for attestation verification")
            repro = run_result.get("reproducibility_manifest")
            if not isinstance(repro, dict):
                raise ValueError("run_result missing reproducibility_manifest")
            run_fingerprint = repro.get("run_fingerprint_sha256")
            inputs_fingerprint = repro.get("inputs_fingerprint_sha256")
            if not isinstance(run_fingerprint, str) or not isinstance(inputs_fingerprint, str):
                raise ValueError("run_result reproducibility fingerprint missing")
            if len(run_fingerprint) != 64 or len(inputs_fingerprint) != 64:
                raise ValueError("run_result reproducibility fingerprint must be sha256 hex")

            plan = _build_default_flow_plan_payload(preset_id)
            expected_attestation_base = {
                "schema_version": "sc-neurocore.studio.default-flow-attestation.v1",
                "evidence_classification": "default_flow",
                "status": "completed",
                "preset_id": preset_id,
                "flow_id": "studio_default_adaptive_precision_v1",
                "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                "inputs_fingerprint_sha256": inputs_fingerprint,
                "run_fingerprint_sha256": run_fingerprint,
            }
            expected_attestation_fingerprint = _sha256_json(expected_attestation_base)

            schema_match = (
                isinstance(attestation, dict)
                and attestation.get("schema_version")
                == "sc-neurocore.studio.default-flow-attestation.v1"
            )
            plan_match = (
                schema_match
                and attestation.get("plan_fingerprint_sha256") == plan["plan_fingerprint_sha256"]
            )
            inputs_match = (
                schema_match and attestation.get("inputs_fingerprint_sha256") == inputs_fingerprint
            )
            run_match = (
                schema_match and attestation.get("run_fingerprint_sha256") == run_fingerprint
            )
            attestation_fingerprint_match = (
                schema_match
                and attestation.get("attestation_fingerprint_sha256")
                == expected_attestation_fingerprint
            )

            return {
                "schema_version": "sc-neurocore.studio.default-flow-attestation-verify.v1",
                "preset_id": preset_id,
                "verified": bool(
                    schema_match
                    and plan_match
                    and inputs_match
                    and run_match
                    and attestation_fingerprint_match
                ),
                "checks": {
                    "schema_match": bool(schema_match),
                    "plan_fingerprint_match": bool(plan_match),
                    "inputs_fingerprint_match": bool(inputs_match),
                    "run_fingerprint_match": bool(run_match),
                    "attestation_fingerprint_match": bool(attestation_fingerprint_match),
                },
                "expected_plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                "expected_attestation_fingerprint_sha256": expected_attestation_fingerprint,
            }

        return _safe(fn)

    return router
