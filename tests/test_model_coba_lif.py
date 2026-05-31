# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — COBALIFNeuron behavioural contract tests

"""Module-specific conductance-based LIF dynamics contracts."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.coba_lif import COBALIFNeuron


def _snapshot(neuron: COBALIFNeuron) -> tuple[float, float, float]:
    return neuron.v, neuron.g_e, neuron.g_i


def test_default_step_preserves_finite_conductance_state() -> None:
    neuron = COBALIFNeuron()

    outputs = [neuron.step(0.0) for _ in range(20)]

    assert set(outputs) <= {0, 1}
    assert neuron.v == pytest.approx(neuron.e_l)
    assert neuron.g_e == pytest.approx(0.0)
    assert neuron.g_i == pytest.approx(0.0)


def test_conductance_injections_are_applied_before_exponential_decay() -> None:
    neuron = COBALIFNeuron()

    assert neuron.step(0.0, delta_ge=5.0, delta_gi=3.0) == 0

    assert neuron.g_e == pytest.approx(5.0 * math.exp(-neuron.dt / neuron.tau_e))
    assert neuron.g_i == pytest.approx(3.0 * math.exp(-neuron.dt / neuron.tau_i))


def test_excitatory_conductance_depolarizes_relative_to_rest() -> None:
    rest = COBALIFNeuron()
    excited = COBALIFNeuron()

    rest.step(0.0)
    excited.step(0.0, delta_ge=20.0)

    assert excited.v > rest.v


def test_inhibitory_conductance_hyperpolarizes_relative_to_rest() -> None:
    rest = COBALIFNeuron(v=-60.0)
    inhibited = COBALIFNeuron(v=-60.0)

    rest.step(0.0)
    inhibited.step(0.0, delta_gi=20.0)

    assert inhibited.v < rest.v


def test_suprathreshold_drive_resets_voltage_but_preserves_decayed_conductance() -> None:
    neuron = COBALIFNeuron(v=-51.0)

    assert neuron.step(1.0e5, delta_ge=5.0) == 1

    assert neuron.v == neuron.v_reset
    assert neuron.g_e > 0.0
    assert neuron.g_i == 0.0


def test_reset_restores_resting_voltage_and_clears_conductances_only() -> None:
    neuron = COBALIFNeuron(e_l=-62.0, dt=0.05)
    neuron.step(100.0, delta_ge=5.0, delta_gi=2.0)

    neuron.reset()

    assert _snapshot(neuron) == (-62.0, 0.0, 0.0)
    assert neuron.dt == 0.05


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("v", -250.0),
        ("g_e", -1.0),
        ("g_i", float("inf")),
        ("g_e", 1.1e9),
        ("c_m", 0.0),
        ("g_l", -1.0),
        ("tau_e", 0.0),
        ("tau_i", float("nan")),
        ("e_l", float("inf")),
        ("e_e", float("nan")),
        ("e_i", float("inf")),
        ("v_threshold", float("nan")),
        ("v_reset", -250.0),
        ("dt", 0.0),
    ],
)
def test_invalid_runtime_state_or_parameters_do_not_mutate(field: str, value: float) -> None:
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
    current: float, delta_ge: float, delta_gi: float
) -> None:
    neuron = COBALIFNeuron(v=-60.0, g_e=1.0, g_i=2.0)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(current, delta_ge=delta_ge, delta_gi=delta_gi)

    assert _snapshot(neuron) == before


def test_voltage_candidate_outside_safety_envelope_does_not_mutate() -> None:
    neuron = COBALIFNeuron(v=90.0, g_e=0.0, g_i=0.0)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0e8)

    assert _snapshot(neuron) == before


def test_conductance_candidate_outside_safety_envelope_does_not_mutate() -> None:
    neuron = COBALIFNeuron(g_e=1.0, g_i=2.0)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(0.0, delta_ge=1.1e9)

    assert _snapshot(neuron) == before
