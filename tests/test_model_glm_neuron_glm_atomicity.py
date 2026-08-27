# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLM invalid-input atomicity and determinism contracts

"""Fail-closed and determinism contracts for the point-process GLM."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from sc_neurocore.neurons.models.glm_neuron import GLMNeuron


def _buffers(neuron: GLMNeuron) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return neuron._stim_buf.copy(), neuron._spike_buf.copy()


@pytest.mark.parametrize("stimulus", (math.nan, math.inf, -math.inf))
def test_non_finite_stimulus_is_rejected_atomically(stimulus: float) -> None:
    neuron = GLMNeuron(seed=1)
    stim_before, spike_before = _buffers(neuron)
    with pytest.raises(ValueError, match="stimulus"):
        neuron.step(stimulus)
    assert np.array_equal(neuron._stim_buf, stim_before)
    assert np.array_equal(neuron._spike_buf, spike_before)


def test_nan_stimulus_no_longer_poisons_the_history_buffer() -> None:
    """Reject NaN before it can silently poison the stimulus history."""

    neuron = GLMNeuron(seed=1)
    with pytest.raises(ValueError, match="stimulus"):
        neuron.step(math.nan)
    assert np.all(np.isfinite(neuron._stim_buf))
    assert neuron.step(1.0) in (0, 1)


@pytest.mark.parametrize("uniform", (1.0, -0.1, math.nan, math.inf))
def test_out_of_domain_uniform_is_rejected_atomically(uniform: float) -> None:
    neuron = GLMNeuron(seed=1)
    stim_before, spike_before = _buffers(neuron)
    with pytest.raises(ValueError, match="uniform"):
        neuron.step(1.0, uniform=uniform)
    assert np.array_equal(neuron._stim_buf, stim_before)
    assert np.array_equal(neuron._spike_buf, spike_before)


def test_corrupted_configuration_is_rejected_atomically() -> None:
    neuron = GLMNeuron(seed=1)
    neuron.mu = math.nan
    stim_before, spike_before = _buffers(neuron)
    with pytest.raises(ValueError, match="mu"):
        neuron.step(1.0)
    assert np.array_equal(neuron._stim_buf, stim_before)
    assert np.array_equal(neuron._spike_buf, spike_before)


def test_corrupted_history_buffer_is_rejected() -> None:
    neuron = GLMNeuron(seed=1)
    neuron._stim_buf[3] = math.inf
    with pytest.raises(ValueError, match="buffers"):
        neuron.step(1.0)


def test_resized_history_buffer_is_rejected() -> None:
    neuron = GLMNeuron(seed=1)
    neuron._spike_buf = np.zeros(3)
    with pytest.raises(ValueError, match="buffers"):
        neuron.step(1.0)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("n_k", 0, ValueError),
        ("n_k", 1.5, TypeError),
        ("n_h", True, TypeError),
        ("mu", math.nan, ValueError),
        ("dt_ms", 0.0, ValueError),
        ("dt_ms", math.inf, ValueError),
    ],
)
def test_constructor_rejects_invalid_configuration(
    field: str, value: float | bool, error: type[Exception]
) -> None:
    constructor: Callable[..., GLMNeuron] = GLMNeuron
    with pytest.raises(error):
        constructor(**{field: value})


def test_constructor_rejects_mismatched_filters() -> None:
    with pytest.raises(ValueError, match="lengths"):
        GLMNeuron(n_k=10, k=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        GLMNeuron(h=np.full(20, math.nan))


def test_seeded_generator_is_reproducible_and_none_is_entropy() -> None:
    first = GLMNeuron(seed=42)
    second = GLMNeuron(seed=42)
    train_first = [first.step(5.0) for _ in range(500)]
    train_second = [second.step(5.0) for _ in range(500)]
    assert train_first == train_second
    assert GLMNeuron().seed is None


def test_explicit_uniform_is_deterministic_and_bypasses_the_generator() -> None:
    left = GLMNeuron(seed=1)
    right = GLMNeuron(seed=2)
    samples = [((index * 37 + 11) % 97) / 97.0 for index in range(200)]
    train_left = [left.step(5.0, uniform=sample) for sample in samples]
    train_right = [right.step(5.0, uniform=sample) for sample in samples]
    assert train_left == train_right
    assert np.array_equal(left._stim_buf, right._stim_buf)
    assert np.array_equal(left._spike_buf, right._spike_buf)


def test_forced_spike_and_forced_silence() -> None:
    neuron = GLMNeuron(seed=1)
    assert neuron.step(20.0, uniform=0.0) == 1
    assert neuron._spike_buf[0] == 1.0
    silent = GLMNeuron(seed=1)
    assert silent.step(0.0, uniform=0.999999) == 0
    assert silent._spike_buf[0] == 0.0


def test_reset_clears_history_and_preserves_filters() -> None:
    neuron = GLMNeuron(seed=3)
    for _ in range(50):
        neuron.step(5.0)
    k_before = neuron.k.copy()
    neuron.reset()
    assert np.all(neuron._stim_buf == 0.0)
    assert np.all(neuron._spike_buf == 0.0)
    assert np.array_equal(neuron.k, k_before)
