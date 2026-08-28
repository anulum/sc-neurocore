# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Independent equation checks for the source-bound McKean system."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.mckean import McKeanNeuron


def _rhs(n: McKeanNeuron, v: float, w: float, current: float) -> tuple[float, float]:
    heaviside = 1.0 if v >= n.a else 0.0
    return -n.lambda_ * v + n.mu * heaviside - w + current, n.b * v


def _oracle(n: McKeanNeuron, current: float) -> tuple[float, float, int]:
    dt = n.dt
    k1 = _rhs(n, n.v, n.w, current)
    k2 = _rhs(n, n.v + dt * k1[0] / 2.0, n.w + dt * k1[1] / 2.0, current)
    k3 = _rhs(n, n.v + dt * k2[0] / 2.0, n.w + dt * k2[1] / 2.0, current)
    k4 = _rhs(n, n.v + dt * k3[0], n.w + dt * k3[1], current)
    v = n.v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
    w = n.w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    return v, w, int(n.v < n.a <= v)


def test_source_defaults_and_right_continuous_switch_are_pinned() -> None:
    n = McKeanNeuron()
    assert (n.v, n.w, n.a, n.lambda_, n.mu, n.b, n.dt) == (
        0.0,
        0.0,
        0.25,
        1.0,
        1.0,
        0.01,
        0.1,
    )
    assert n._derivatives(n.a, 0.0, 0.0)[0] == 0.75
    assert n._derivatives(math.nextafter(n.a, -math.inf), 0.0, 0.0)[0] < 0.0


@pytest.mark.parametrize("current", [-0.2, 0.0, 0.5, 3.0])
def test_one_step_matches_independent_simultaneous_rk4(current: float) -> None:
    n = McKeanNeuron(v=0.1, w=-0.05)
    expected_v, expected_w, expected_event = _oracle(n, current)
    assert n.step(current) == expected_event
    assert n.v == pytest.approx(expected_v, abs=1.0e-15)
    assert n.w == pytest.approx(expected_w, abs=1.0e-15)


def test_event_is_upward_crossing_without_reset() -> None:
    n = McKeanNeuron()
    assert n.step(3.0) == 1
    assert n.v >= n.a
    crossed = n.v
    assert n.step(0.0) == 0
    assert n.v != 0.0 and n.v != crossed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": 0.0},
        {"lambda_": 0.0},
        {"mu": 0.25},
        {"b": 0.0},
        {"dt": 0.0},
        {"dt": 1.01},
    ],
)
def test_source_constraints_are_enforced(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        McKeanNeuron(**kwargs)


def test_failure_is_atomic() -> None:
    n = McKeanNeuron()
    before = (n.v, n.w)
    with pytest.raises(ValueError, match="current"):
        n.step(math.nan)
    assert (n.v, n.w) == before


def test_corrupted_state_validation_is_atomic() -> None:
    n = McKeanNeuron()
    n.v = 1
    n.w = "invalid"  # type: ignore[assignment]
    before = (n.v, n.w, n.a, n.lambda_, n.mu, n.b, n.dt)
    with pytest.raises(TypeError, match="w"):
        n.step(0.0)
    assert (n.v, n.w, n.a, n.lambda_, n.mu, n.b, n.dt) == before


def test_source_and_sc_identities_are_distinct() -> None:
    from sc_neurocore.neurons.models.sc_triangular_mckean import SCTriangularMcKeanNeuron

    assert not hasattr(McKeanNeuron(), "epsilon")
    assert not hasattr(SCTriangularMcKeanNeuron(), "lambda_")
