# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ComplementaryLIFNeuron behavioural contract tests

"""Module-specific complementary LIF dual-path dynamics contracts."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.clif import ComplementaryLIFNeuron


def _snapshot(neuron: ComplementaryLIFNeuron) -> tuple[float, float, float]:
    return neuron.v_pos, neuron.v_neg, neuron.alpha


def test_alpha_matches_leaky_membrane_decay_constant() -> None:
    neuron = ComplementaryLIFNeuron(tau=20.0, dt=0.5)

    assert neuron.alpha == pytest.approx(math.exp(-0.5 / 20.0))
    assert 0.0 <= neuron.alpha < 1.0


def test_positive_and_negative_inputs_charge_separate_paths() -> None:
    positive = ComplementaryLIFNeuron(v_threshold=10.0)
    negative = ComplementaryLIFNeuron(v_threshold=10.0)

    assert positive.step(0.4) == 0
    assert negative.step(-0.4) == 0

    assert positive.v_pos == pytest.approx(0.4)
    assert positive.v_neg == 0.0
    assert negative.v_pos == 0.0
    assert negative.v_neg == pytest.approx(0.4)


def test_zero_input_decays_both_paths_by_runtime_alpha() -> None:
    neuron = ComplementaryLIFNeuron(v_pos=2.0, v_neg=1.0, tau=10.0, dt=1.0, v_threshold=10.0)

    assert neuron.step(0.0) == 0

    assert neuron.v_pos == pytest.approx(2.0 * neuron.alpha)
    assert neuron.v_neg == pytest.approx(1.0 * neuron.alpha)


def test_positive_and_negative_threshold_crossings_reset_both_paths() -> None:
    positive = ComplementaryLIFNeuron()
    negative = ComplementaryLIFNeuron()

    assert positive.step(1.5) == 1
    assert negative.step(-1.5) == -1

    assert (positive.v_pos, positive.v_neg) == (0.0, 0.0)
    assert (negative.v_pos, negative.v_neg) == (0.0, 0.0)


def test_balanced_alternating_drive_remains_bounded_and_mostly_silent() -> None:
    neuron = ComplementaryLIFNeuron(v_threshold=1.0)

    outputs = [neuron.step(0.4 if step % 2 == 0 else -0.4) for step in range(200)]

    assert outputs.count(1) + outputs.count(-1) < 10
    assert abs(neuron.v_pos - neuron.v_neg) < 0.5


def test_runtime_tau_or_dt_mutation_recomputes_alpha_before_update() -> None:
    neuron = ComplementaryLIFNeuron(v_pos=1.0, v_threshold=10.0)
    neuron.tau = 100.0
    neuron.dt = 2.0

    assert neuron.step(0.0) == 0

    expected_alpha = math.exp(-2.0 / 100.0)
    assert neuron.alpha == pytest.approx(expected_alpha)
    assert neuron.v_pos == pytest.approx(expected_alpha)


def test_reset_only_clears_dynamic_paths() -> None:
    neuron = ComplementaryLIFNeuron(tau=20.0, dt=0.5, v_threshold=3.0)
    neuron.step(1.0)

    neuron.reset()

    assert (neuron.v_pos, neuron.v_neg) == (0.0, 0.0)
    assert neuron.tau == 20.0
    assert neuron.dt == 0.5
    assert neuron.v_threshold == 3.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v_pos", float("nan")),
        ("v_neg", float("inf")),
        ("v_pos", 1.0e13),
        ("v_neg", -1.0e13),
        ("tau", 0.0),
        ("tau", float("nan")),
        ("dt", 0.0),
        ("dt", float("inf")),
        ("v_threshold", 0.0),
        ("v_threshold", float("nan")),
    ],
)
def test_invalid_runtime_state_or_parameters_do_not_mutate(field: str, value: float) -> None:
    neuron = ComplementaryLIFNeuron(v_pos=0.25, v_neg=0.5)
    setattr(neuron, field, value)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(0.1)

    assert _snapshot(neuron) == before


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = ComplementaryLIFNeuron(v_pos=0.25, v_neg=0.5)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(float("nan"))

    assert _snapshot(neuron) == before


def test_non_finite_candidate_does_not_mutate_state() -> None:
    neuron = ComplementaryLIFNeuron(v_pos=0.25, v_neg=0.5)
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0e309)

    assert _snapshot(neuron) == before
