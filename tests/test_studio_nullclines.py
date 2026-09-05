# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio nullcline validity tests

"""Nullclines carry validity masks; an invalid sample is never a zero field."""

from __future__ import annotations

import math

import numpy as np
import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.nullclines import NULLCLINE_SCHEMA_VERSION, nullclines_2d


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_analytic_control_harmonic_oscillator_nullclines_are_the_axes() -> None:
    """dv/dt = w vanishes on w = 0 and dw/dt = -v on v = 0; the whole grid is valid."""
    result = nullclines_2d(
        ["dv/dt = w", "dw/dt = -v"], {}, ["v", "w"], {"v": (-1.0, 1.0), "w": (-1.0, 1.0)}, 21
    )
    assert result["schema_version"] == NULLCLINE_SCHEMA_VERSION
    assert result["domain"]["status"] == "complete"
    assert result["contract"]["kind"] == "nullclines"
    assert result["contract"]["domain"] == "complete"
    assert np.asarray(result["validity_0"]).all() and np.asarray(result["validity_1"]).all()
    # Every contour cell of dv/dt = w straddles w = 0 (cell corner at w in [-0.1, 0]).
    for _x, y in result["nullcline_0"]["points"]:
        assert -0.11 < y <= 0.0
    for x, _y in result["nullcline_1"]["points"]:
        assert -0.11 < x <= 0.0
    # w = 0 is itself a grid sample, so the two rows of cells sharing that corner qualify.
    assert result["nullcline_0"]["cells"] == len(result["nullcline_0"]["points"]) == 40
    assert result["held"] == {} and result["current"] == 0.0


def test_singular_field_reports_invalid_samples_instead_of_zero_contours() -> None:
    """1/v is undefined at v = 0: that column is invalid and carries no contour."""
    result = nullclines_2d(
        ["dv/dt = 1 / v - w", "dw/dt = v"], {}, ["v", "w"], {"v": (-1.0, 1.0), "w": (-3.0, 3.0)}, 21
    )
    validity = np.asarray(result["validity_0"], dtype=bool)
    assert result["domain"]["status"] == "partial"
    assert not validity[:, 10].any()  # the v = 0 column
    assert validity[:, :10].all() and validity[:, 11:].all()
    assert np.asarray(result["validity_1"], dtype=bool).all()
    detail = result["contract"]["domain_detail"]
    assert detail["cells_valid"]["v"] == 21 * 20
    assert detail["invalid_fraction"]["v"] == pytest.approx(1 / 21)
    # No contour point of dv/dt may sit on a cell touching the invalid column
    # (cells with lower-left x index 9 or 10 touch column 10).
    x_axis = result["grid"]["x"]
    for x, _y in result["nullcline_0"]["points"]:
        index = x_axis.index(x)
        assert index not in (9, 10)
    # The genuine nullcline 1/v = w exists on both sides.
    assert result["nullcline_0"]["cells"] > 0


def test_partial_domain_from_log_and_sqrt_is_masked_per_component() -> None:
    result = nullclines_2d(
        ["dv/dt = log(w) - v", "dw/dt = sqrt(w) - 1"],
        {},
        ["v", "w"],
        {"v": (-2.0, 2.0), "w": (-1.0, 3.0)},
        20,
    )
    assert result["domain"]["status"] == "partial"
    validity_0 = np.asarray(result["validity_0"], dtype=bool)
    validity_1 = np.asarray(result["validity_1"], dtype=bool)
    y_axis = result["grid"]["y"]
    for row, w in enumerate(y_axis):
        assert validity_0[row].all() == (w > 0.0)
        assert validity_1[row].all() == (w >= 0.0)
    # sqrt(w) = 1 at w = 1 : every contour cell of component 1 straddles w = 1.
    for _x, y in result["nullcline_1"]["points"]:
        assert y <= 1.0 < y + (y_axis[1] - y_axis[0]) + 1e-12


def test_overflow_and_non_finite_samples_are_invalid_not_zero() -> None:
    result = nullclines_2d(
        ["dv/dt = exp(v) - w", "dw/dt = v * 0"],
        {},
        ["v", "w"],
        {"v": (0.0, 1000.0), "w": (-1.0, 1.0)},
        21,
    )
    validity = np.asarray(result["validity_0"], dtype=bool)
    assert result["domain"]["status"] == "partial"
    assert validity[:, 0].all() and not validity[:, -1].any()
    # dw/dt = 0 everywhere is a genuine zero field of a valid component: its contour covers every cell.
    assert result["nullcline_1"]["cells"] == 20 * 20
    assert np.asarray(result["validity_1"], dtype=bool).all()


def test_empty_domain_is_reported_when_no_sample_is_valid() -> None:
    result = nullclines_2d(
        ["dv/dt = log(w)", "dw/dt = v"], {}, ["v", "w"], {"v": (-1.0, 1.0), "w": (-3.0, -1.0)}, 20
    )
    assert result["domain"]["status"] == "empty"
    assert result["nullcline_0"]["points"] == []
    assert result["contract"]["domain"] == "empty"
    assert result["contract"]["domain_detail"]["cells_valid"]["v"] == 0


def test_held_variables_and_input_current_are_stated_and_used() -> None:
    result = nullclines_2d(
        ["dv/dt = -(v - E) + I * u", "dw/dt = a * (v - w)", "du/dt = -u"],
        {"E": -65.0, "a": 0.1},
        ["v", "w"],
        {"v": (-80.0, -40.0), "w": (-80.0, -40.0)},
        11,
        current=5.0,
        held={"u": 2.0},
    )
    assert result["held"] == {"u": 2.0}
    assert result["current"] == 5.0
    # dv/dt = 0 at v = -65 + 10 = -55 for every w.
    for x, _y in result["nullcline_0"]["points"]:
        assert x <= -55.0 < x + 4.0 + 1e-12
    default_held = nullclines_2d(
        ["dv/dt = -(v - E) + I * u", "dw/dt = a * (v - w)", "du/dt = -u"],
        {"E": -65.0, "a": 0.1},
        ["v", "w"],
        {"v": (-80.0, -40.0), "w": (-80.0, -40.0)},
        11,
    )
    assert default_held["held"] == {"u": 0.0}


def test_stochastic_drift_field_evaluates_xi_at_zero() -> None:
    result = nullclines_2d(
        ["dv/dt = w + xi", "dw/dt = -v"], {}, ["v", "w"], {"v": (-1.0, 1.0), "w": (-1.0, 1.0)}, 11
    )
    assert result["domain"]["status"] == "complete"
    assert any("xi" in item for item in result["contract"]["applicability"])


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"var_names": ["v"]}, "var_names"),
        ({"var_names": ["v", "v"]}, "var_names"),
        ({"var_names": ["v", "q"]}, "var_names"),
        ({"held": {"q": 1.0}}, "held"),
        ({"held": {"v": 1.0}}, "held"),
        ({"ranges": {"v": (1.0, -1.0)}}, "ranges"),
        ({"ranges": {"v": (math.nan, 1.0)}}, "ranges"),
        ({"equations": ["dv/dt = w + zz", "dw/dt = -v"]}, "equations"),
        ({"grid_size": 1}, "grid_size"),
    ],
)
def test_invalid_requests_name_their_field(kwargs: dict[str, object], field: str) -> None:
    request: dict[str, object] = {
        "equations": ["dv/dt = w", "dw/dt = -v"],
        "params": {},
        "var_names": ["v", "w"],
        "ranges": {"v": (-1.0, 1.0), "w": (-1.0, 1.0)},
        "grid_size": 5,
    }
    request.update(kwargs)
    with pytest.raises(ModelInputError) as info:
        nullclines_2d(**request)  # type: ignore[arg-type]
    assert info.value.field == field


def test_nullcline_route_carries_domain_in_the_manifest(client: TestClient) -> None:
    response = client.post(
        "/api/nullclines",
        json={
            "equations": ["dv/dt = log(w) - v", "dw/dt = sqrt(w) - 1"],
            "params": {},
            "var_names": ["v", "w"],
            "ranges": {"v": [-2.0, 2.0], "w": [-1.0, 3.0]},
            "grid_size": 20,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"]["status"] == "partial"
    assert payload["analysis_metadata"]["contract"] == "nullclines"
    assert payload["analysis_metadata"]["domain"] == "partial"
    assert "validity_0" in payload["analysis_metadata"]["output_keys"]

    rejected = client.post(
        "/api/nullclines",
        json={
            "equations": ["dv/dt = w", "dw/dt = -v"],
            "params": {},
            "var_names": ["v", "q"],
            "ranges": {},
            "grid_size": 20,
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["field"] == "var_names"

    held = client.post(
        "/api/nullclines",
        json={
            "equations": ["dv/dt = -(v + 65) + I * u", "dw/dt = v - w", "du/dt = -u"],
            "params": {},
            "var_names": ["v", "w"],
            "ranges": {"v": [-80.0, -40.0], "w": [-80.0, -40.0]},
            "grid_size": 20,
            "current": 5.0,
            "held": {"u": 2.0},
        },
    )
    assert held.status_code == 200
    assert held.json()["held"] == {"u": 2.0}
    assert held.json()["current"] == 5.0

    unknown_key = client.post(
        "/api/nullclines",
        json={
            "equations": ["dv/dt = w", "dw/dt = -v"],
            "params": {},
            "var_names": ["v", "w"],
            "ranges": {},
            "grid": 3,
        },
    )
    assert unknown_key.status_code == 422
