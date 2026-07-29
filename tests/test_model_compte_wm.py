# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source Compte cell behavioral contracts

"""Focused source, safety, reset, and pipeline contracts for CompteWMNeuron."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron


def _snapshot(neuron: CompteWMNeuron) -> tuple[float, ...]:
    return tuple(neuron.get_state().values())


def test_source_control_set_defaults_are_explicit() -> None:
    """Pin the source pyramidal, channel, and integration defaults."""
    neuron = CompteWMNeuron()
    assert (neuron.g_l, neuron.g_ampa, neuron.g_nmda, neuron.g_gaba) == (
        0.025,
        0.0031,
        0.000381,
        0.001336,
    )
    assert (neuron.v_reset, neuron.tau_ref, neuron.tau_gaba, neuron.dt) == (
        -60.0,
        2.0,
        10.0,
        0.02,
    )


def test_three_presynaptic_event_pathways_are_separate() -> None:
    """Reject the inherited AMPA/NMDA conflation and self-GABA recurrence."""
    recurrent = CompteWMNeuron()
    external = CompteWMNeuron()
    inhibitory = CompteWMNeuron()
    recurrent.step(spike_in=True)
    external.step(external_spike=True)
    inhibitory.step(inhibitory_spike=True)
    assert recurrent.s_ampa == 0.0 and recurrent.s_nmda > 0.0 and recurrent.x_nmda > 0.0
    assert external.s_ampa > 0.0 and external.s_nmda == 0.0 and external.x_nmda == 0.0
    assert inhibitory.s_gaba > 0.0
    assert recurrent.s_gaba == external.s_gaba == 0.0


def test_output_spike_resets_and_starts_source_refractory_without_autapse() -> None:
    """Keep postsynaptic reset independent of incoming inhibitory state."""
    neuron = CompteWMNeuron(v=-50.01)
    assert neuron.step(1.0) == 1
    assert neuron.v == -60.0
    assert neuron._ref_remaining == 2.0
    assert neuron.s_gaba == 0.0
    assert neuron.step(100.0) == 0
    assert neuron.v == -60.0
    assert neuron._ref_remaining == pytest.approx(1.98)


def test_synaptic_states_continue_during_refractory() -> None:
    """Evolve incoming channel kinetics while the membrane remains clamped."""
    neuron = CompteWMNeuron(v=-50.01)
    neuron.step(1.0, external_spike=True)
    before = neuron.s_ampa
    neuron.step(0.0)
    assert neuron.v == neuron.v_reset
    assert 0.0 < neuron.s_ampa < before


def test_mg_block_is_voltage_dependent_and_bounded() -> None:
    neuron = CompteWMNeuron()
    low = neuron._mg_block(-80.0)
    high = neuron._mg_block(0.0)
    assert 0.0 <= low < high <= 1.0


def test_midpoint_rk2_matches_independent_one_step_arithmetic() -> None:
    """Exercise the coupled nonlinear NMDA and membrane midpoint."""
    neuron = CompteWMNeuron(v=-63.0, s_ampa=0.2, s_nmda=0.1, x_nmda=0.3, s_gaba=0.4)
    initial = (-63.0, 0.2, 0.1, 0.3, 0.4)
    k1 = neuron._derivatives(*initial, 0.7, membrane_active=True)
    midpoint = tuple(value + 0.5 * neuron.dt * slope for value, slope in zip(initial, k1))
    k2 = neuron._derivatives(
        midpoint[0], midpoint[1], midpoint[2], midpoint[3], midpoint[4], 0.7, membrane_active=True
    )
    expected = tuple(value + neuron.dt * slope for value, slope in zip(initial, k2))
    assert neuron.step(0.7) == 0
    assert _snapshot(neuron)[:5] == pytest.approx(expected, abs=0.0)


def test_reset_preserves_configuration() -> None:
    neuron = CompteWMNeuron(e_l=-68.0, dt=0.01, tau_gaba=9.0)
    neuron.step(2.0, True, external_spike=True, inhibitory_spike=True)
    neuron.reset()
    assert _snapshot(neuron) == (-68.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert (neuron.dt, neuron.tau_gaba) == (0.01, 9.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", math.nan),
        ("s_ampa", -1.0),
        ("s_nmda", 1.1),
        ("x_nmda", math.inf),
        ("s_gaba", -1.0),
        ("g_l", -1.0),
        ("g_ampa", math.nan),
        ("g_nmda", -1.0),
        ("g_gaba", -1.0),
        ("c_m", 0.0),
        ("mg", -1.0),
        ("tau_ampa", 0.0),
        ("tau_nmda", math.nan),
        ("tau_x", 0.0),
        ("tau_gaba", 0.0),
        ("alpha_nmda", -1.0),
        ("v_threshold", math.nan),
        ("v_reset", -250.0),
        ("tau_ref", 0.0),
        ("dt", 0.0),
        ("_ref_remaining", -1.0),
    ],
)
def test_invalid_mutated_state_is_atomic(field: str, value: float) -> None:
    neuron = CompteWMNeuron(v=-63.0, s_ampa=0.2, s_nmda=0.1, x_nmda=0.3, s_gaba=0.4)
    setattr(neuron, field, value)
    before = _snapshot(neuron)
    with pytest.raises(ValueError):
        neuron.step(0.0)
    assert _snapshot(neuron) == before


def test_invalid_current_and_gate_overflow_are_atomic() -> None:
    neuron = CompteWMNeuron(v=-63.0, s_ampa=0.2, s_nmda=0.1, x_nmda=0.3, s_gaba=0.4)
    before = _snapshot(neuron)
    with pytest.raises(ValueError):
        neuron.step(math.nan)
    assert _snapshot(neuron) == before
    neuron.x_nmda = neuron._GATE_MAX
    before = _snapshot(neuron)
    with pytest.raises(ValueError):
        neuron.step(0.0, True)
    assert _snapshot(neuron) == before


def test_complete_python_batch_matches_scalar_steps() -> None:
    """Exercise all public inputs, traces, events, and final-state mutation."""
    steps = 800
    index = np.arange(steps)
    currents = 1.0 + 0.2 * np.sin(index * 0.03)
    recurrent = (index % 17 == 0).astype(np.int64)
    external = (index % 11 == 0).astype(np.int64)
    inhibitory = (index % 23 == 0).astype(np.int64)
    scalar = CompteWMNeuron()
    expected = []
    for values in zip(currents, recurrent, external, inhibitory, strict=True):
        event = scalar.step(
            float(values[0]),
            bool(values[1]),
            external_spike=bool(values[2]),
            inhibitory_spike=bool(values[3]),
        )
        expected.append((*scalar.get_state().values(), event))
    batched = CompteWMNeuron()
    result = batched.simulate(currents, recurrent, external, inhibitory, backend="python")
    arrays = tuple(
        cast(npt.NDArray[np.float64], result[key])
        for key in ("voltages", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "refractory")
    )
    actual: npt.NDArray[np.float64] = np.column_stack(
        tuple((*arrays, cast(npt.NDArray[np.int64], result["events"])))
    )
    np.testing.assert_array_equal(actual, np.asarray(expected))
    assert batched.get_state() == scalar.get_state()
