# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sliding PSN publication fidelity and atomicity

"""Paper-exact sliding PSN dynamics, validation, and atomic rejection."""

from __future__ import annotations

import math
from typing import cast

import pytest

from sc_neurocore.neurons.models import ParallelSpikingNeuron as PublicPSN
from sc_neurocore.neurons.models.psn import ParallelSpikingNeuron

_DRIVE = tuple(0.4 + 0.3 * math.sin(index * 0.17) for index in range(64))


def _oracle_hidden(weights: tuple[float, ...], drive: tuple[float, ...], t: int) -> float:
    """Recompute paper Eq. 14 directly from the drive with zero padding."""
    k = len(weights)
    hidden = 0.0
    for i, weight in enumerate(weights):
        j = t - k + 1 + i
        hidden += weight * (drive[j] if j >= 0 else 0.0)
    return hidden


def test_registry_exports_the_canonical_class() -> None:
    assert PublicPSN is ParallelSpikingNeuron


def test_matches_paper_equation_oracle_bit_exactly() -> None:
    weights = (0.1, -0.2, 0.35, 0.75)
    neuron = ParallelSpikingNeuron(kernel_size=4, v_threshold=0.4, weights=weights)
    for t, current in enumerate(_DRIVE):
        spike = neuron.step(current)
        hidden = _oracle_hidden(weights, _DRIVE, t)
        assert neuron.hidden == hidden
        assert spike == (1 if hidden >= 0.4 else 0)


def test_uniform_default_weights_are_one_over_k() -> None:
    neuron = ParallelSpikingNeuron(kernel_size=5)
    assert neuron.weights == tuple(1.0 / 5 for _ in range(5))
    assert neuron.v_threshold == 1.0


def test_warm_up_matches_zero_padded_pre_history() -> None:
    neuron = ParallelSpikingNeuron()
    neuron.step(0.5)
    assert neuron.hidden == 0.5 / 8
    neuron.step(0.25)
    assert neuron.hidden == (0.5 + 0.25) / 8


def test_theta_is_right_continuous_at_threshold() -> None:
    neuron = ParallelSpikingNeuron(kernel_size=1, v_threshold=1.0)
    assert neuron.step(1.0) == 1


def test_firing_never_clears_the_retained_inputs() -> None:
    neuron = ParallelSpikingNeuron(kernel_size=4, v_threshold=0.5)
    spikes = [neuron.step(1.0) for _ in range(8)]
    assert sum(spikes) >= 5
    assert neuron._history == [1.0, 1.0, 1.0, 1.0]


def test_reset_clears_history_and_hidden_only() -> None:
    neuron = ParallelSpikingNeuron(kernel_size=4, v_threshold=0.5)
    neuron.step(1.0)
    neuron.reset()
    assert neuron._history == [0.0, 0.0, 0.0, 0.0]
    assert neuron.hidden == 0.0
    assert neuron.weights == (0.25, 0.25, 0.25, 0.25)


@pytest.mark.parametrize("bad", (math.nan, math.inf, -math.inf))
def test_non_finite_input_is_rejected_atomically(bad: float) -> None:
    neuron = ParallelSpikingNeuron()
    neuron.step(0.7)
    before = (list(neuron._history), neuron.hidden)
    with pytest.raises(ValueError, match="current"):
        neuron.step(bad)
    assert (list(neuron._history), neuron.hidden) == before


def test_overflowing_hidden_state_is_rejected_atomically() -> None:
    huge = 1e308
    neuron = ParallelSpikingNeuron(kernel_size=2, weights=(huge, huge))
    neuron._history = [huge, huge]
    with pytest.raises(ValueError, match="non-finite"):
        neuron.step(huge)
    assert neuron._history == [huge, huge]


@pytest.mark.parametrize("kernel_size", (0, -3))
def test_non_positive_kernel_size_is_rejected(kernel_size: int) -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        ParallelSpikingNeuron(kernel_size=kernel_size)


def test_non_integer_kernel_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        ParallelSpikingNeuron(kernel_size=cast(int, 4.0))


def test_non_integer_kernel_size_with_weights_is_rejected() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        ParallelSpikingNeuron(kernel_size=cast(int, 2.0), weights=(0.5, 0.5))


def test_weights_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="weights"):
        ParallelSpikingNeuron(kernel_size=4, weights=(0.5, 0.5))


def test_non_finite_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="weights"):
        ParallelSpikingNeuron(kernel_size=2, weights=(0.5, math.nan))


def test_non_finite_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="v_threshold"):
        ParallelSpikingNeuron(v_threshold=math.inf)


def test_mutated_kernel_size_is_rejected_before_stepping() -> None:
    neuron = ParallelSpikingNeuron(kernel_size=2)
    neuron.kernel_size = 0
    with pytest.raises(ValueError, match="kernel_size"):
        neuron.step(0.1)


def test_corrupted_history_is_rejected_before_stepping() -> None:
    neuron = ParallelSpikingNeuron(kernel_size=2)
    neuron._history = [0.0, math.nan]
    with pytest.raises(ValueError, match="retained inputs"):
        neuron.step(0.1)


def test_shortened_history_is_rejected_before_stepping() -> None:
    neuron = ParallelSpikingNeuron(kernel_size=2)
    neuron._history = [0.0]
    with pytest.raises(ValueError, match="retained inputs"):
        neuron.step(0.1)
