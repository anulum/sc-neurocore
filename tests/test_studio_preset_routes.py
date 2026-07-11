# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio preset route failure-contract tests

"""Exercise preset-flow validation through the public Studio API."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api import preset_flows
from sc_neurocore.studio.api import presets as preset_routes
from sc_neurocore.studio.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a client for the public Studio preset routes."""
    return TestClient(create_app(), base_url="http://127.0.0.1")


@pytest.fixture(scope="module")
def valid_plan(client: TestClient) -> dict[str, Any]:
    """Return the current public default-flow plan."""
    response = client.get("/api/presets/fpga_precision/default-flow/plan")
    assert response.status_code == 200
    return dict(response.json())


@pytest.fixture(scope="module")
def valid_contract(client: TestClient) -> dict[str, Any]:
    """Return the current public default-flow contract."""
    response = client.get("/api/presets/fpga_precision/default-flow/contract")
    assert response.status_code == 200
    return dict(response.json())


@pytest.fixture(scope="module")
def valid_run(client: TestClient) -> dict[str, Any]:
    """Return one completed public default-flow run."""
    response = client.post(
        "/api/presets/fpga_precision/default-flow/run",
        json={"action_overrides": {}},
    )
    assert response.status_code == 200
    return dict(response.json())


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown": 1},
        {"target_error_percent": "not-a-number"},
    ],
)
def test_preset_action_resolve_rejects_invalid_overrides(
    client: TestClient,
    overrides: dict[str, Any],
) -> None:
    response = client.post(
        "/api/presets/fpga_precision/actions/auto_tune_adaptive_precision/resolve",
        json={"overrides": overrides},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"


@pytest.mark.parametrize(
    ("method", "route", "payload"),
    [
        ("get", "/api/presets/missing/actions", None),
        ("post", "/api/presets/missing/actions/action/resolve", {"overrides": {}}),
        ("post", "/api/presets/missing/actions/action/execute", {"overrides": {}}),
        ("post", "/api/presets/missing/actions/execute-all", {"action_overrides": {}}),
        ("post", "/api/presets/missing/default-flow/run", {"action_overrides": {}}),
        ("get", "/api/presets/missing/default-flow/plan", None),
        ("get", "/api/presets/missing/default-flow/contract", None),
        (
            "post",
            "/api/presets/missing/default-flow/verify",
            {
                "action_order": [],
                "template_fingerprints": {},
                "plan_fingerprint_sha256": None,
            },
        ),
        (
            "post",
            "/api/presets/missing/default-flow/run-guarded",
            {
                "action_order": [],
                "template_fingerprints": {},
                "plan_fingerprint_sha256": "0" * 64,
                "action_overrides": {},
            },
        ),
        (
            "post",
            "/api/presets/missing/default-flow/run-from-contract",
            {"contract": {}, "action_overrides": {}},
        ),
        ("post", "/api/presets/missing/default-flow/attest", {"run_result": {}}),
        (
            "post",
            "/api/presets/missing/default-flow/attest/verify",
            {"run_result": {}, "attestation": {}},
        ),
    ],
)
def test_preset_subroutes_reject_missing_preset(
    client: TestClient,
    method: str,
    route: str,
    payload: dict[str, Any] | None,
) -> None:
    response = client.get(route) if method == "get" else client.post(route, json=payload)

    assert response.status_code == 404


@pytest.mark.parametrize("suffix", ["resolve", "execute"])
def test_preset_action_routes_reject_missing_action(
    client: TestClient,
    suffix: str,
) -> None:
    response = client.post(
        f"/api/presets/fpga_precision/actions/missing/{suffix}",
        json={"overrides": {}},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("suffix", ["resolve", "execute"])
def test_preset_action_routes_reject_missing_template(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    monkeypatch.setattr(
        preset_routes,
        "get_preset_action",
        lambda _preset_id, _action_id: {
            "endpoint": "/api/adaptive-precision/auto-tune",
            "id": "missing_template",
            "method": "POST",
            "payload_template": None,
        },
    )

    response = client.post(
        f"/api/presets/fpga_precision/actions/missing_template/{suffix}",
        json={"overrides": {}},
    )

    assert response.status_code == 422


def test_execute_all_skips_action_without_string_id(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preset_routes,
        "get_preset_actions",
        lambda _preset_id: [
            {
                "endpoint": "/api/adaptive-precision/auto-tune",
                "id": 7,
                "method": "POST",
                "payload_template": {},
            }
        ],
    )

    response = client.post(
        "/api/presets/fpga_precision/actions/execute-all",
        json={"action_overrides": {}},
    )

    assert response.status_code == 200
    assert response.json()["executed_count"] == 0


def test_execute_all_rejects_missing_action_template(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preset_routes,
        "get_preset_actions",
        lambda _preset_id: [
            {
                "endpoint": "/api/adaptive-precision/auto-tune",
                "id": "missing_template",
                "method": "POST",
                "payload_template": None,
            }
        ],
    )

    response = client.post(
        "/api/presets/fpga_precision/actions/execute-all",
        json={"action_overrides": {}},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"


def test_default_flow_plan_preserves_missing_method(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preset_flows,
        "get_preset_actions",
        lambda _preset_id: [
            {
                "endpoint": "/api/adaptive-precision/auto-tune",
                "id": "action_without_method",
                "method": None,
                "payload_template": {},
            }
        ],
    )

    response = client.get("/api/presets/fpga_precision/default-flow/plan")

    assert response.status_code == 200
    assert response.json()["actions"][0]["method"] is None


@pytest.mark.parametrize(
    ("route", "method"),
    [
        ("/api/presets/fpga_precision/default-flow/plan", "get"),
        ("/api/presets/fpga_precision/default-flow/run", "post"),
    ],
)
def test_default_flow_rejects_malformed_action_templates(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    method: str,
) -> None:
    monkeypatch.setattr(
        preset_flows,
        "get_preset_actions",
        lambda _preset_id: [
            {
                "endpoint": "/api/adaptive-precision/auto-tune",
                "id": 7,
                "method": "POST",
                "payload_template": {},
            },
            {
                "endpoint": "/api/adaptive-precision/auto-tune",
                "id": "malformed_action",
                "method": "POST",
                "payload_template": None,
            },
        ],
    )

    response = client.get(route) if method == "get" else client.post(route, json={})

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"


def test_guarded_run_rejects_stale_plan(
    client: TestClient,
    valid_plan: dict[str, Any],
) -> None:
    fingerprints = {
        row["action_id"]: row["template_fingerprint_sha256"] for row in valid_plan["actions"]
    }
    response = client.post(
        "/api/presets/fpga_precision/default-flow/run-guarded",
        json={
            "action_order": list(reversed(valid_plan["action_order"])),
            "template_fingerprints": fingerprints,
            "plan_fingerprint_sha256": valid_plan["plan_fingerprint_sha256"],
            "action_overrides": {},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "preset",
        "missing_guarded",
        "invalid_guarded",
        "invalid_fingerprint",
        "drift",
    ],
)
def test_run_from_contract_rejects_invalid_contract(
    client: TestClient,
    valid_contract: dict[str, Any],
    case: str,
) -> None:
    contract = deepcopy(valid_contract)
    guarded = contract["guarded_run_request_template"]
    if case == "schema":
        contract["schema_version"] = "unsupported"
    elif case == "preset":
        contract["preset_id"] = "other"
    elif case == "missing_guarded":
        contract.pop("guarded_run_request_template")
    elif case == "invalid_guarded":
        guarded["action_order"] = "invalid"
    elif case == "invalid_fingerprint":
        guarded["plan_fingerprint_sha256"] = 7
    else:
        guarded["plan_fingerprint_sha256"] = "0" * 64

    response = client.post(
        "/api/presets/fpga_precision/default-flow/run-from-contract",
        json={"contract": contract, "action_overrides": {}},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"


@pytest.mark.parametrize(
    "case",
    ["preset", "flow", "missing_manifest", "missing_fingerprint", "short_fingerprint"],
)
@pytest.mark.parametrize("suffix", ["attest", "attest/verify"])
def test_attestation_routes_reject_invalid_run_result(
    client: TestClient,
    valid_run: dict[str, Any],
    case: str,
    suffix: str,
) -> None:
    run_result = deepcopy(valid_run)
    manifest = run_result["reproducibility_manifest"]
    if case == "preset":
        run_result["preset_id"] = "other"
    elif case == "flow":
        run_result["flow_id"] = "unsupported"
    elif case == "missing_manifest":
        run_result.pop("reproducibility_manifest")
    elif case == "missing_fingerprint":
        manifest["run_fingerprint_sha256"] = None
    else:
        manifest["run_fingerprint_sha256"] = "short"
    payload: dict[str, Any] = {"run_result": run_result}
    if suffix == "attest/verify":
        payload["attestation"] = {}

    response = client.post(
        f"/api/presets/fpga_precision/default-flow/{suffix}",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"


def test_attestation_verify_reports_mismatched_attestation(
    client: TestClient,
    valid_run: dict[str, Any],
) -> None:
    response = client.post(
        "/api/presets/fpga_precision/default-flow/attest/verify",
        json={"run_result": valid_run, "attestation": {}},
    )

    assert response.status_code == 200
    assert response.json()["verified"] is False
