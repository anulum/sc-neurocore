# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — COBALIFNeuron public behavioural contract tests

"""Source defaults, RK4 recurrence, refractory timing, and safety contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron

_State = tuple[float, float, float, float]


def _snapshot(neuron: COBALIFNeuron) -> _State:
    """Return every mutable runtime state through public attributes."""
    return neuron.v, neuron.g_e, neuron.g_i, neuron.refractory_time


def _derivatives(
    neuron: COBALIFNeuron,
    v: float,
    g_e: float,
    g_i: float,
    current: float,
) -> tuple[float, float, float]:
    """Independently evaluate the sourced continuous right-hand sides."""
    synaptic = g_e * (v - neuron.e_e) + g_i * (v - neuron.e_i)
    return (
        (-neuron.g_l * (v - neuron.e_l) - synaptic + current) / neuron.c_m,
        -g_e / neuron.tau_e,
        -g_i / neuron.tau_i,
    )


def _rk4_reference(
    neuron: COBALIFNeuron,
    v: float,
    g_e: float,
    g_i: float,
    current: float,
) -> tuple[float, float, float]:
    """Advance the three sourced equations with independent classical RK4."""
    dt = neuron.dt
    k1v, k1e, k1i = _derivatives(neuron, v, g_e, g_i, current)
    k2v, k2e, k2i = _derivatives(
        neuron,
        v + 0.5 * dt * k1v,
        g_e + 0.5 * dt * k1e,
        g_i + 0.5 * dt * k1i,
        current,
    )
    k3v, k3e, k3i = _derivatives(
        neuron,
        v + 0.5 * dt * k2v,
        g_e + 0.5 * dt * k2e,
        g_i + 0.5 * dt * k2i,
        current,
    )
    k4v, k4e, k4i = _derivatives(
        neuron,
        v + dt * k3v,
        g_e + dt * k3e,
        g_i + dt * k3i,
        current,
    )
    return (
        v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        g_e + (dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
        g_i + (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
    )


def _decay_reference(value: float, tau: float, dt: float) -> float:
    """Advance one conductance decay with independent classical RK4."""
    k1 = -value / tau
    k2 = -(value + 0.5 * dt * k1) / tau
    k3 = -(value + 0.5 * dt * k2) / tau
    k4 = -(value + dt * k3) / tau
    return value + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def test_factory_defaults_match_brette_benchmark_one() -> None:
    """Keep the maintained factory aligned with Brette et al. Benchmark 1."""
    neuron = COBALIFNeuron()

    assert _snapshot(neuron) == (-60.0, 0.0, 0.0, 0.0)
    assert (
        neuron.c_m,
        neuron.g_l,
        neuron.e_l,
        neuron.e_e,
        neuron.e_i,
        neuron.tau_e,
        neuron.tau_i,
        neuron.v_threshold,
        neuron.v_reset,
        neuron.refractory_period,
        neuron.dt,
    ) == (200.0, 10.0, -60.0, 0.0, -80.0, 5.0, 10.0, -50.0, -60.0, 5.0, 0.1)


def test_default_step_preserves_resting_state() -> None:
    """Zero drive leaves the source resting fixed point unchanged."""
    neuron = COBALIFNeuron()

    outputs = [neuron.step(0.0) for _ in range(20)]

    assert outputs == [0] * 20
    assert _snapshot(neuron) == pytest.approx((-60.0, 0.0, 0.0, 0.0))


def test_conductance_injections_precede_the_coupled_rk4_candidate() -> None:
    """Boundary conductance events participate in every RK4 stage."""
    neuron = COBALIFNeuron()
    expected = _rk4_reference(neuron, neuron.v, 5.0, 3.0, 0.0)

    assert neuron.step(0.0, delta_ge=5.0, delta_gi=3.0) == 0

    assert _snapshot(neuron) == pytest.approx((*expected, 0.0), rel=0.0, abs=1.0e-14)


def test_excitatory_and_inhibitory_reversals_have_opposite_effects() -> None:
    """The two source reversal potentials drive membrane voltage oppositely."""
    rest = COBALIFNeuron()
    excited = COBALIFNeuron()
    inhibited = COBALIFNeuron()

    rest.step(0.0)
    excited.step(0.0, delta_ge=20.0)
    inhibited.step(0.0, delta_gi=20.0)

    assert excited.v > rest.v > inhibited.v


def test_spike_resets_voltage_and_preserves_rk4_conductance_candidate() -> None:
    """Threshold handling occurs after the complete coupled RK4 candidate."""
    neuron = COBALIFNeuron(v=-51.0)
    _, expected_ge, expected_gi = _rk4_reference(neuron, neuron.v, 5.0, 0.0, 1.0e5)

    assert neuron.step(1.0e5, delta_ge=5.0) == 1

    assert neuron.v == neuron.v_reset
    assert neuron.g_e == pytest.approx(expected_ge, rel=0.0, abs=1.0e-14)
    assert neuron.g_i == pytest.approx(expected_gi, rel=0.0, abs=1.0e-14)
    assert neuron.refractory_time == neuron.refractory_period


def test_refractory_hold_is_exactly_five_milliseconds() -> None:
    """The source refractory interval neither loses nor gains a float residue step."""
    neuron = COBALIFNeuron(v=-51.0, e_l=-65.0)
    assert neuron.step(1.0e5) == 1

    for _ in range(50):
        assert neuron.step(0.0) == 0
        assert neuron.v == neuron.v_reset

    assert neuron.refractory_time == 0.0
    assert neuron.step(0.0) == 0
    assert neuron.v < neuron.v_reset


def test_conductances_decay_with_rk4_during_refractory_hold() -> None:
    """Absolute refractoriness holds voltage but does not freeze synapses."""
    neuron = COBALIFNeuron(v=-60.0, g_e=5.0, g_i=3.0, refractory_time=0.1)
    expected_ge = _decay_reference(neuron.g_e, neuron.tau_e, neuron.dt)
    expected_gi = _decay_reference(neuron.g_i, neuron.tau_i, neuron.dt)

    assert neuron.step(500.0, delta_ge=1.0, delta_gi=2.0) == 0

    assert neuron.v == neuron.v_reset
    assert neuron.g_e == pytest.approx(
        _decay_reference(6.0, neuron.tau_e, neuron.dt), rel=0.0, abs=1.0e-14
    )
    assert neuron.g_i == pytest.approx(
        _decay_reference(5.0, neuron.tau_i, neuron.dt), rel=0.0, abs=1.0e-14
    )
    assert neuron.g_e != expected_ge
    assert neuron.g_i != expected_gi
    assert neuron.refractory_time == 0.0


def test_python_batch_matches_repeated_public_steps() -> None:
    """The Python batch path is the exact public step recurrence, not a surrogate."""
    batched = COBALIFNeuron(v=-59.0, g_e=1.0, g_i=0.5)
    stepped = COBALIFNeuron(v=-59.0, g_e=1.0, g_i=0.5)

    trace, spikes = batched.simulate(
        300,
        current=650.0,
        delta_ge=0.15,
        delta_gi=0.07,
        backend="python",
    )
    expected_trace = []
    expected_spikes = 0
    for _ in range(300):
        expected_spikes += stepped.step(650.0, 0.15, 0.07)
        expected_trace.append(stepped.v)

    np.testing.assert_array_equal(trace, np.asarray(expected_trace))
    assert spikes == expected_spikes
    assert _snapshot(batched) == _snapshot(stepped)


def test_reset_restores_leak_reversal_and_clears_all_runtime_state() -> None:
    """Reset preserves configuration while restoring the complete runtime state."""
    neuron = COBALIFNeuron(e_l=-62.0, dt=0.05)
    neuron.step(100.0, delta_ge=5.0, delta_gi=2.0)

    neuron.reset()

    assert _snapshot(neuron) == (-62.0, 0.0, 0.0, 0.0)
    assert neuron.dt == 0.05


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("v", -250.0),
        ("g_e", -1.0),
        ("g_i", float("inf")),
        ("g_e", 1.1e9),
        ("refractory_time", 5.1),
        ("c_m", 0.0),
        ("g_l", -1.0),
        ("tau_e", 0.0),
        ("tau_i", float("nan")),
        ("e_l", float("inf")),
        ("e_e", float("nan")),
        ("e_i", float("inf")),
        ("v_threshold", float("nan")),
        ("v_reset", -250.0),
        ("refractory_period", 0.0),
        ("dt", 0.0),
    ],
)
def test_invalid_runtime_state_or_parameters_do_not_mutate(field: str, value: float) -> None:
    """Malformed stored contracts fail before any runtime state mutation."""
    neuron = COBALIFNeuron(v=-60.0, g_e=1.0, g_i=2.0)
    setattr(neuron, field, value)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(0.0)

    assert _snapshot(neuron) == before


@pytest.mark.parametrize(
    ("current", "delta_ge", "delta_gi"),
    [
        (float("nan"), 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, float("inf")),
        (0.0, float("nan"), 0.0),
    ],
)
def test_invalid_step_inputs_do_not_mutate(
    current: float,
    delta_ge: float,
    delta_gi: float,
) -> None:
    """Malformed boundary inputs fail before any runtime state mutation."""
    neuron = COBALIFNeuron(v=-60.0, g_e=1.0, g_i=2.0)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(current, delta_ge=delta_ge, delta_gi=delta_gi)

    assert _snapshot(neuron) == before


def test_voltage_candidate_outside_safety_envelope_does_not_mutate() -> None:
    """A raw suprathreshold RK4 candidate is validated before voltage reset."""
    neuron = COBALIFNeuron(v=90.0, g_e=0.0, g_i=0.0)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0e8)

    assert _snapshot(neuron) == before


def test_conductance_candidate_outside_safety_envelope_does_not_mutate() -> None:
    """Oversized boundary conductance events fail before RK4 or mutation."""
    neuron = COBALIFNeuron(g_e=1.0, g_i=2.0)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(0.0, delta_ge=1.1e9)

    assert _snapshot(neuron) == before
