# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Publication contract for WongWangUnit

"""Scalar, stochastic-boundary, and atomic-batch tests for Wong-Wang 2006."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit
from sc_neurocore.network.population import Population


def _phi_oracle(current: float) -> float:
    """Evaluate the Appendix transfer function independently."""
    x = 270.0 * current - 108.0
    if -0.154 * x > 700.0:
        return 0.0
    if abs(x) < 1.0e-7:
        return 1.0 / 0.154
    return max(0.0, x / -math.expm1(-0.154 * x))


def test_defaults_are_the_paper_parameter_set() -> None:
    """Pin the reduced model and the paper's stated 0.1 ms step."""
    unit = WongWangUnit()
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == (0.1, 0.1, 0.0, 0.0)
    assert (unit.tau_s, unit.tau_ampa, unit.gamma) == (0.1, 0.002, 0.641)
    assert (unit.j_n, unit.j_cross, unit.i_0, unit.sigma) == (0.2609, 0.0497, 0.3255, 0.02)
    assert unit.dt == 0.0001


def test_supplied_samples_execute_one_simultaneous_euler_ou_update() -> None:
    """Discriminate the source Euler/OU recurrence from the removed RK4 path."""
    unit = WongWangUnit(s1=0.24, s2=0.11, noise1=0.01, noise2=-0.02)
    old = (unit.s1, unit.s2, unit.noise1, unit.noise2)
    stim1, stim2, xi1, xi2 = 0.17, 0.03, 0.5, -1.0
    current1 = unit.j_n * old[0] - unit.j_cross * old[1] + unit.i_0 + stim1 + old[2]
    current2 = unit.j_n * old[1] - unit.j_cross * old[0] + unit.i_0 + stim2 + old[3]
    expected_rates = (_phi_oracle(current1), _phi_oracle(current2))
    scale = math.sqrt(unit.dt / unit.tau_ampa) * unit.sigma
    expected_state = (
        old[0] + unit.dt * (-old[0] / unit.tau_s + (1.0 - old[0]) * unit.gamma * expected_rates[0]),
        old[1] + unit.dt * (-old[1] / unit.tau_s + (1.0 - old[1]) * unit.gamma * expected_rates[1]),
        old[2] - (unit.dt / unit.tau_ampa) * old[2] + scale * xi1,
        old[3] - (unit.dt / unit.tau_ampa) * old[3] + scale * xi2,
    )

    actual_rates = unit.step_with_gaussian_samples(stim1, stim2, xi1, xi2)

    assert actual_rates == pytest.approx(expected_rates, abs=1.0e-15)
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == pytest.approx(
        expected_state, abs=1.0e-15
    )


def test_step_draws_exactly_two_external_normal_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin caller-visible RNG order as xi1 then xi2."""
    samples = iter((0.25, -0.5))
    draws: list[float] = []

    def draw() -> float:
        value = next(samples)
        draws.append(value)
        return value

    monkeypatch.setattr(np.random, "randn", draw)
    stochastic = WongWangUnit()
    deterministic = WongWangUnit()
    rates = stochastic.step(0.01, -0.02)
    expected = deterministic.step_with_gaussian_samples(0.01, -0.02, 0.25, -0.5)
    assert draws == [0.25, -0.5]
    assert rates == expected
    assert (stochastic.s1, stochastic.s2, stochastic.noise1, stochastic.noise2) == (
        deterministic.s1,
        deterministic.s2,
        deterministic.noise1,
        deterministic.noise2,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("s1", -0.01),
        ("s2", 1.01),
        ("noise1", math.nan),
        ("noise2", math.inf),
        ("tau_s", 0.0),
        ("tau_ampa", 0.0),
        ("gamma", 0.0),
        ("j_n", -0.1),
        ("j_cross", -0.1),
        ("i_0", math.nan),
        ("sigma", -0.1),
        ("dt", 0.0),
    ),
)
def test_constructor_rejects_invalid_numerical_contract(field: str, value: float) -> None:
    """Reject invalid source parameters before a state exists."""
    with pytest.raises((ValueError, FloatingPointError)):
        WongWangUnit(**{field: value})


@pytest.mark.parametrize("bad_input", (math.nan, math.inf, -math.inf))
def test_invalid_inputs_leave_all_four_states_unchanged(bad_input: float) -> None:
    """Fail atomically at the explicit stochastic boundary."""
    unit = WongWangUnit(noise1=0.01, noise2=-0.02)
    before = (unit.s1, unit.s2, unit.noise1, unit.noise2)
    with pytest.raises(ValueError, match="finite"):
        unit.step_with_gaussian_samples(0.0, 0.0, bad_input, 0.0)
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == before


def test_out_of_range_candidate_is_rejected_without_silent_clipping() -> None:
    """Preserve the mathematical update by rejecting, not clipping, bad candidates."""
    unit = WongWangUnit(s1=0.99, s2=0.1, sigma=0.0, dt=0.1)
    before = (unit.s1, unit.s2, unit.noise1, unit.noise2)
    with pytest.raises(FloatingPointError, match=r"left \[0, 1\]"):
        unit.step_with_gaussian_samples(10.0, 0.0, 0.0, 0.0)
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == before


def test_phi_is_stable_at_the_removable_singularity_and_extremes() -> None:
    """Exercise the exact transfer boundary without overflow or NaN."""
    assert WongWangUnit._phi(108.0 / 270.0) == pytest.approx(1.0 / 0.154)
    assert WongWangUnit._phi(-1.0e6) == 0.0
    assert WongWangUnit._phi(0.5) == pytest.approx(_phi_oracle(0.5), abs=1.0e-15)
    with pytest.raises(ValueError, match="synaptic current"):
        WongWangUnit._phi(math.nan)


def test_reset_preserves_configuration_and_resets_all_dynamic_states() -> None:
    """Reset only the four evolving values."""
    unit = WongWangUnit(tau_s=0.12, tau_ampa=0.003, dt=0.0002)
    unit.step_with_gaussian_samples(0.2, 0.0, 1.0, -1.0)
    unit.reset()
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == (0.1, 0.1, 0.0, 0.0)
    assert (unit.tau_s, unit.tau_ampa, unit.dt) == (0.12, 0.003, 0.0002)


def test_public_python_batch_returns_complete_consistent_traces() -> None:
    """Expose all physical states, rates, and final-state receipts."""
    steps = 64
    stim1 = np.linspace(0.0, 0.03, steps)
    stim2 = np.linspace(0.01, -0.02, steps)
    xi = np.sin(np.arange(2 * steps) * 0.17)
    unit = WongWangUnit()
    result = unit.simulate(stim1, stim2, xi, backend="python")
    for key in ("s1", "s2", "noise1", "noise2", "r1", "r2"):
        trace = result[key]
        assert isinstance(trace, np.ndarray) and trace.shape == (steps,)
    s1_trace = result["s1"]
    s2_trace = result["s2"]
    noise1_trace = result["noise1"]
    noise2_trace = result["noise2"]
    assert isinstance(s1_trace, np.ndarray)
    assert isinstance(s2_trace, np.ndarray)
    assert isinstance(noise1_trace, np.ndarray)
    assert isinstance(noise2_trace, np.ndarray)
    assert unit.s1 == float(result["s1_final"]) == float(s1_trace[-1])
    assert unit.s2 == float(result["s2_final"]) == float(s2_trace[-1])
    assert unit.noise1 == float(result["noise1_final"]) == float(noise1_trace[-1])
    assert unit.noise2 == float(result["noise2_final"]) == float(noise2_trace[-1])


def test_public_batch_error_is_atomic() -> None:
    """Do not mutate the instance when dispatch or validation fails."""
    unit = WongWangUnit(noise1=0.01, noise2=-0.02)
    before = (unit.s1, unit.s2, unit.noise1, unit.noise2)
    with pytest.raises(ValueError, match="xi length"):
        unit.simulate(np.zeros(2), np.zeros(2), np.zeros(2), backend="python")
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == before


def test_empty_batch_is_a_complete_no_op() -> None:
    """Preserve every dynamic state for zero work."""
    unit = WongWangUnit(s1=0.2, s2=0.3, noise1=0.01, noise2=-0.02)
    result = unit.simulate([], [], [], backend="python")
    assert all(np.asarray(result[key]).size == 0 for key in ("s1", "s2", "noise1", "noise2"))
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == (0.2, 0.3, 0.01, -0.02)


def test_population_can_construct_rate_circuit_instances() -> None:
    """Keep catalogue construction separate from spike-network compatibility."""
    assert Population(WongWangUnit, n=5, label="ww").n == 5
