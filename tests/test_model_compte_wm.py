# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CompteWMNeuron behavioural contract tests

"""Module-specific Compte working-memory NMDA/Mg-block dynamics contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron


def _snapshot(n: CompteWMNeuron) -> tuple[float, float, float, float, float]:
    return n.v, n.s_ampa, n.s_nmda, n.x_nmda, n.s_gaba


def test_spike_input_activates_ampa_and_nmda_precursor_with_decay() -> None:
    n = CompteWMNeuron()

    assert n.step(0.0, spike_in=True) == 0

    assert n.s_ampa > 0.0
    assert n.x_nmda > 0.0
    assert n.s_nmda > 0.0


def test_mg_block_is_voltage_dependent_and_bounded() -> None:
    n = CompteWMNeuron()

    low = n._mg_block(-80.0)
    high = n._mg_block(0.0)

    assert 0.0 <= low < high <= 1.0


def test_excitation_depolarizes_relative_to_rest() -> None:
    rest = CompteWMNeuron()
    excited = CompteWMNeuron()

    rest.step(0.0)
    excited.step(0.0, spike_in=True)

    assert excited.v > rest.v


def test_suprathreshold_drive_resets_voltage_and_adds_gaba_feedback() -> None:
    n = CompteWMNeuron(v=-51.0)

    assert n.step(100.0) == 1

    assert n.v == n.v_reset
    assert n.s_gaba > 0.0


def test_reset_restores_resting_voltage_and_clears_synaptic_gates() -> None:
    n = CompteWMNeuron(e_l=-68.0)
    n.step(3.0, spike_in=True)

    n.reset()

    assert _snapshot(n) == (-68.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("v", -250.0),
        ("s_ampa", -1.0),
        ("s_nmda", 1.1),
        ("x_nmda", float("inf")),
        ("s_gaba", -1.0),
        ("g_l", -1.0),
        ("g_ampa", -1.0),
        ("g_nmda", float("nan")),
        ("g_gaba", -1.0),
        ("e_l", float("inf")),
        ("e_exc", float("nan")),
        ("e_inh", float("inf")),
        ("c_m", 0.0),
        ("mg", -1.0),
        ("tau_ampa", 0.0),
        ("tau_nmda", float("nan")),
        ("tau_x", 0.0),
        ("alpha_nmda", -1.0),
        ("v_threshold", float("nan")),
        ("v_reset", -250.0),
        ("dt", 0.0),
    ],
)
def test_invalid_runtime_state_or_parameters_do_not_mutate(field: str, value: float) -> None:
    n = CompteWMNeuron(v=-60.0, s_ampa=0.2, s_nmda=0.1, x_nmda=0.3, s_gaba=0.4)
    setattr(n, field, value)
    before = _snapshot(n)

    with pytest.raises(ValueError):
        n.step(0.0)

    assert _snapshot(n) == before


def test_non_finite_current_does_not_mutate_state() -> None:
    n = CompteWMNeuron(v=-60.0, s_ampa=0.2, s_nmda=0.1, x_nmda=0.3, s_gaba=0.4)
    before = _snapshot(n)

    with pytest.raises(ValueError):
        n.step(float("nan"))

    assert _snapshot(n) == before


def test_nmda_candidate_above_unit_bound_does_not_mutate() -> None:
    n = CompteWMNeuron(s_nmda=0.99, x_nmda=1.0e6, dt=0.1)
    before = _snapshot(n)

    with pytest.raises(ValueError):
        n.step(0.0)

    assert _snapshot(n) == before


def test_voltage_candidate_outside_safety_envelope_does_not_mutate() -> None:
    n = CompteWMNeuron(v=90.0)
    before = _snapshot(n)

    with pytest.raises(ValueError):
        n.step(1.0e6)

    assert _snapshot(n) == before
