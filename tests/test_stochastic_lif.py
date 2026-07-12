# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic LIF neuron

import numpy as np
import pytest

from sc_neurocore.neurons.stochastic_lif import LIF_NOISE_STD, StochasticLIFNeuron


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resistance": -1.0},
        {"refractory_period": True},
        {"refractory_period": 1.5},
    ],
)
def test_invalid_stochastic_lif_physics_fail_closed(kwargs):
    with pytest.raises(ValueError):
        StochasticLIFNeuron(**kwargs)


@pytest.mark.parametrize("input_scale", [-0.1, float("nan"), float("inf")])
def test_invalid_bitstream_input_scale_fails_closed(input_scale):
    neuron = StochasticLIFNeuron(noise_std=0.0)
    with pytest.raises(ValueError, match="input_scale"):
        neuron.process_bitstream(np.array([0, 1], dtype=np.uint8), input_scale=input_scale)


class NonFiniteEntropySource:
    def sample_normal(self, mean: float, std: float) -> float:
        return float("nan")


def test_non_finite_entropy_sample_fails_closed_before_state_mutation():
    neuron = StochasticLIFNeuron(noise_std=0.1, entropy_source=NonFiniteEntropySource())
    old_v = neuron.v

    with pytest.raises(ValueError, match="noise"):
        neuron.step(0.0)

    assert neuron.v == old_v


@pytest.mark.parametrize("voltage", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_runtime_voltage_fails_closed_before_recurrence(voltage):
    neuron = StochasticLIFNeuron(noise_std=0.0)
    neuron.v = voltage

    with pytest.raises(ValueError, match="v"):
        neuron.step(0.0)

    if np.isnan(voltage):
        assert np.isnan(neuron.v)
    else:
        assert neuron.v == voltage


@pytest.mark.parametrize("counter", [-1, 1.5, True])
def test_invalid_refractory_counter_fails_closed_before_recurrence(counter):
    neuron = StochasticLIFNeuron(noise_std=0.0)
    neuron.refractory_counter = counter
    old_v = neuron.v

    with pytest.raises(ValueError, match="refractory_counter"):
        neuron.step(0.0)

    assert neuron.v == old_v


def test_refractory_period_holds_voltage_at_rest_for_integer_duration():
    neuron = StochasticLIFNeuron(
        v_threshold=0.5,
        refractory_period=2,
        noise_std=0.0,
        resistance=1.0,
        dt=1.0,
        tau_mem=1e9,
    )

    assert neuron.step(1.0) == 1
    assert neuron.step(1.0) == 0
    assert neuron.v == pytest.approx(neuron.v_rest)
    assert neuron.step(1.0) == 0
    assert neuron.v == pytest.approx(neuron.v_rest)
    assert neuron.step(1.0) == 1


class _ConstantEntropySource:
    """Deterministic entropy stub returning a fixed noise sample every step."""

    def __init__(self, value: float) -> None:
        self._value = value

    def sample_normal(self, mean: float, std: float) -> float:
        """Return the fixed sample, ignoring the requested mean and std."""
        return self._value


def test_default_is_deterministic_but_positive_noise_exercises_the_stochastic_path() -> None:
    """Separate measured-default behaviour from the stochastic capability.

    The measured behaviour facet probes the default constructor, whose canonical
    ``noise_std`` is ``LIF_NOISE_STD == 0.0``: the recurrence is deterministic, so
    the descriptor correctly records ``[excitable, rate-coded, tonic]`` without a
    ``stochastic`` tag. A positive ``noise_std`` still takes the entropy branch and
    changes the spike train, so the Euler-Maruyama stochastic path stays real and
    exercised — capability is kept distinct from the default behaviour tags.
    """
    assert LIF_NOISE_STD == 0.0
    bits = np.ones(64, dtype=np.uint8)

    baseline = StochasticLIFNeuron(noise_std=0.0).process_bitstream(bits, input_scale=0.08)
    repeat = StochasticLIFNeuron(noise_std=0.0).process_bitstream(bits, input_scale=0.08)
    assert np.array_equal(baseline, repeat)

    noisy = StochasticLIFNeuron(
        noise_std=0.2, entropy_source=_ConstantEntropySource(0.2)
    ).process_bitstream(bits, input_scale=0.08)
    assert not np.array_equal(noisy, baseline)
    assert int(noisy.sum()) > int(baseline.sum())
