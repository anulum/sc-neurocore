# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — parameter sweeps run under the drive the request states

"""Bifurcation and 2-D sweeps honour the requested protocol.

The bifurcation sweep ran every point under a sine drive whatever the request
said (its schema had no protocol field, so the Studio's protocol was dropped),
and the heatmap always used a constant one. For the default catalogue model
the sine's negative half-cycle pushed the membrane out of the model's safety
bounds at t = 62.4 ms at any swept value and any step size, so the Studio's
bifurcation view failed for it every time.
"""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

MODEL = "ATypeKNeuron"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _bifurcation(client: TestClient, **extra: Any) -> Any:
    body = {
        "model_name": MODEL,
        "dt": 0.5,
        "duration": 100.0,
        "current": 10.0,
        "sweep_param": "g_na",
        "sweep_min": 7.0,
        "sweep_max": 105.0,
        "sweep_steps": 5,
        **extra,
    }
    return client.post("/api/bifurcation", json=body)


def test_bifurcation_defaults_to_a_constant_drive_and_completes(client: TestClient) -> None:
    response = _bifurcation(client)
    assert response.status_code == 200, response.text[:400]
    payload = response.json()
    assert len(payload["param_values"]) == 5
    assert payload["protocol"] == "constant"
    assert "frequency_hz" not in payload


def test_bifurcation_runs_the_requested_protocol(client: TestClient) -> None:
    # A step drive stays inside the model's bounds; the result records it.
    response = _bifurcation(client, protocol="step")
    assert response.status_code == 200, response.text[:400]
    assert response.json()["protocol"] == "step"


def test_bifurcation_under_the_sine_drive_reports_the_model_failure(client: TestClient) -> None:
    """Report the real sine-drive fault with an authored reason and its exact clock."""
    # The drive the sweep used to impose on every request: for this model it
    # leaves the safety bounds, which is now a stated choice, not a hidden one.
    response = _bifurcation(client, protocol="sine", frequency_hz=10.0)
    assert response.status_code == 422, response.text[:400]
    assert response.json() == {
        "detail": {
            "error": "model_simulation_failed",
            "model": MODEL,
            "backend": "python",
            "step": 125,
            "time_ms": 62.5,
            "diagnostic": "model step could not produce a finite result",
        }
    }


def test_heatmap_runs_the_requested_protocol(client: TestClient) -> None:
    body = {
        "model_name": MODEL,
        "dt": 0.5,
        "duration": 50.0,
        "current": 10.0,
        "protocol": "ramp",
        "param_x": "g_na",
        "x_min": 20.0,
        "x_max": 40.0,
        "x_steps": 3,
        "param_y": "g_k",
        "y_min": 5.0,
        "y_max": 10.0,
        "y_steps": 3,
    }
    response = client.post("/api/heatmap", json=body)
    assert response.status_code == 200, response.text[:400]
    assert response.json()["protocol"] == "ramp"


def test_an_unknown_protocol_is_refused(client: TestClient) -> None:
    assert _bifurcation(client, protocol="square").status_code == 422
