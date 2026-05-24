# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic LIF neuron

import numpy as np
import pytest

from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron


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
