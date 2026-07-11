# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio preset flow contracts

"""Plan, execute, fingerprint, and attest deterministic preset flows."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from sc_neurocore.studio.api.adaptive_precision import (
    _execute_resolved_preset_action,
    _is_executable_preset_action_endpoint,
)
from sc_neurocore.studio.presets import get_preset_actions


def _resolve_action_payload(
    preset_id: str,
    action_id: str,
    action: dict[str, Any],
    payload_template: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    unknown = set(overrides) - set(payload_template)
    if unknown:
        bad = ", ".join(sorted(unknown))
        raise ValueError(f"unknown override keys: {bad}")

    resolved = dict(payload_template)
    for key, value in overrides.items():
        template_value = payload_template[key]
        if template_value is not None and not isinstance(value, type(template_value)):
            raise ValueError(f"override type mismatch for key '{key}'")
        resolved[key] = value

    return {
        "preset_id": preset_id,
        "action_id": action_id,
        "method": action.get("method"),
        "endpoint": action.get("endpoint"),
        "payload": resolved,
    }


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _default_flow_actions(preset_id: str) -> list[dict[str, Any]]:
    actions = get_preset_actions(preset_id)
    return [
        action
        for action in actions
        if _is_executable_preset_action_endpoint(action.get("endpoint"))
    ]


def _build_default_flow_plan_payload(preset_id: str) -> dict[str, Any]:
    actions = _default_flow_actions(preset_id)
    plan_actions: list[dict[str, Any]] = []
    for action in actions:
        action_id = action.get("id")
        endpoint = action.get("endpoint")
        method = action.get("method")
        template = action.get("payload_template")
        if not isinstance(action_id, str) or not isinstance(endpoint, str):
            continue
        if not isinstance(template, dict):
            raise ValueError(f"action '{action_id}' does not define a payload template")
        plan_actions.append(
            {
                "action_id": action_id,
                "endpoint": endpoint,
                "method": method if isinstance(method, str) else None,
                "template_keys": sorted(template.keys()),
                "template_fingerprint_sha256": _sha256_json(template),
            }
        )
    base_payload = {
        "schema_version": "sc-neurocore.studio.default-flow-plan.v1",
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": [row["action_id"] for row in plan_actions],
        "actions": plan_actions,
        "count": len(plan_actions),
    }
    plan_contract = {
        "preset_id": base_payload["preset_id"],
        "flow_id": base_payload["flow_id"],
        "action_order": base_payload["action_order"],
        "actions": [
            {
                "action_id": row["action_id"],
                "endpoint": row["endpoint"],
                "method": row["method"],
                "template_fingerprint_sha256": row["template_fingerprint_sha256"],
            }
            for row in plan_actions
        ],
    }
    base_payload["plan_fingerprint_sha256"] = _sha256_json(plan_contract)
    return base_payload


def _execute_default_flow_with_overrides(
    preset_id: str, action_overrides: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    started = time.perf_counter()
    actions = _default_flow_actions(preset_id)
    results: list[dict[str, Any]] = []
    action_order: list[str] = []
    for action in actions:
        action_id = action.get("id")
        if not isinstance(action_id, str):
            continue
        template = action.get("payload_template")
        if not isinstance(template, dict):
            raise ValueError(f"action '{action_id}' does not define a payload template")
        overrides = action_overrides.get(action_id, {})
        resolved = _resolve_action_payload(preset_id, action_id, action, template, overrides)
        result = _execute_resolved_preset_action(resolved)
        action_order.append(action_id)
        results.append(
            {
                "action_id": action_id,
                "resolved_action": resolved,
                "result": result,
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    deterministic_results = [
        {
            "action_id": row["action_id"],
            "resolved_action": row["resolved_action"],
            "result": row["result"],
        }
        for row in results
    ]
    reproducibility_inputs = {
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": action_order,
        "resolved_actions": [row["resolved_action"] for row in deterministic_results],
    }
    reproducibility_run = {
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": action_order,
        "results": deterministic_results,
    }
    return {
        "schema_version": "sc-neurocore.studio.default-flow-run.v1",
        "evidence_classification": "default_flow",
        "status": "completed",
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": action_order,
        "executed_count": len(results),
        "execution_time_ms": elapsed_ms,
        "results": results,
        "reproducibility_manifest": {
            "hash_algorithm": "sha256",
            "inputs_fingerprint_sha256": _sha256_json(reproducibility_inputs),
            "run_fingerprint_sha256": _sha256_json(reproducibility_run),
        },
    }
