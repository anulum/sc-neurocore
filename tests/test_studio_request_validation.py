# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio request-validation HTTP regressions

"""Exercise JSON rendering of rejected requests through real Studio routes."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Provide the actual Studio app with server exceptions exposed to tests."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as connection:
        yield connection


@pytest.mark.parametrize(
    ("route", "payload"),
    [
        ("/api/studio/auth/login", {"username": "operator", "password": "\ud800"}),
        ("/api/studio/identity/browser-users/operator/password", {"password": "\udfff"}),
        (
            "/api/studio/identity/browser-users",
            {
                "username": "analyst",
                "principal_id": "analyst",
                "roles": ["studio.viewer"],
                "password": "\ud800",
            },
        ),
        ("/api/models/simulate", {"model_name": {"\ud800": ["\udfff", "váhy 🧠"]}}),
    ],
)
def test_unpaired_unicode_in_rejected_requests_remains_422(
    client: TestClient, route: str, payload: dict[str, object]
) -> None:
    """Malformed Unicode receives field errors without failing response encoding."""
    response = client.post(
        route,
        content=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"]
    assert "UnicodeEncodeError" not in response.text
    assert "surrogates not allowed" not in response.text
    assert "\\\\ud" in response.text


def test_validation_preserves_valid_unicode_and_nested_values(client: TestClient) -> None:
    """Normal Unicode and nested JSON values retain their field-error input."""
    value = {"váhy 🧠": ["neurón", 1, True, None]}
    response = client.post("/api/models/simulate", json={"model_name": value})

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["input"] == value for error in errors)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_validation_input_stays_json(client: TestClient, number: str) -> None:
    """The existing non-finite rendering contract remains a structured rejection."""
    response = client.post(
        "/api/models/simulate",
        content='{"model_name":"AdaptiveThresholdIFNeuron","dt":' + number + "}",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["loc"][-1] == "dt" and isinstance(error["input"], str) for error in errors)
