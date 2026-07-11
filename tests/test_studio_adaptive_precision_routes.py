# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio adaptive-precision route tests

"""Exercise adaptive-precision success and validation through Studio routes."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api import presets as preset_routes
from sc_neurocore.studio.app import create_app


@pytest.fixture
def client() -> TestClient:
    """Return a client for adaptive-precision Studio routes."""
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_adaptive_precision_auto_tune_route(client: TestClient) -> None:
    response = client.post(
        "/api/adaptive-precision/auto-tune",
        json={"layer_weights": [[0.1, 0.2, 0.4]]},
    )

    assert response.status_code == 200
    assert response.json()["schema"] == "sc-neurocore.adaptive_precision_plan.v1"


@pytest.mark.parametrize("layer", [[], ["nan"]])
def test_adaptive_precision_rejects_invalid_layer_weights(
    client: TestClient,
    layer: list[object],
) -> None:
    response = client.post(
        "/api/adaptive-precision/auto-tune",
        json={"layer_weights": [layer]},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"


def test_preset_action_execute_rejects_unknown_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preset_routes,
        "get_preset_action",
        lambda _preset_id, _action_id: {
            "endpoint": "/api/not-executable",
            "id": "unknown_endpoint",
            "method": "POST",
            "payload_template": {},
        },
    )

    response = client.post(
        "/api/presets/fpga_precision/actions/unknown_endpoint/execute",
        json={"overrides": {}},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid input"
