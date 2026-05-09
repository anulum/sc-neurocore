# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Neurons

import numpy as np
import pytest

from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron


def test_lif_initialization():
    neuron = StochasticLIFNeuron()
    assert neuron.v == 0.0
    assert neuron.v_threshold == 1.0


def test_lif_integration():
    # Setup neuron with no leak for simple integration testing
    neuron = StochasticLIFNeuron(tau_mem=1e9, dt=1.0, noise_std=0.0)

    # Step 1: Input 0.5. v should be 0 + 0.5 = 0.5
    spike = neuron.step(0.5)
    assert spike == 0
    assert np.isclose(neuron.v, 0.5)

    # Step 2: Input 0.6. v should be 0.5 + 0.6 = 1.1 -> Spike! -> Reset to 0.0
    spike = neuron.step(0.6)
    assert spike == 1
    assert neuron.v == 0.0


def test_lif_leak():
    # Test leak logic
    dt = 1.0
    tau = 10.0
    neuron = StochasticLIFNeuron(tau_mem=tau, dt=dt, noise_std=0.0, v_rest=0.0)

    neuron.v = 1.0  # Set initial potential
    neuron.step(0.0)  # No input

    # Expected decay: dv = -(v - v_rest) * (dt / tau)
    # dv = -(1.0 - 0.0) * (0.1) = -0.1
    # v_new = 0.9
    assert np.isclose(neuron.v, 0.9)


def test_lif_noise():
    # Test that noise is doing something
    neuron = StochasticLIFNeuron(noise_std=0.5, seed=42)
    neuron.step(0.0)
    assert neuron.v != 0.0  # Should have moved due to noise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v_rest": np.nan},
        {"v_reset": np.inf},
        {"v_threshold": np.nan},
        {"tau_mem": 0.0},
        {"tau_mem": np.inf},
        {"dt": 0.0},
        {"dt": np.inf},
        {"noise_std": -0.1},
        {"noise_std": np.nan},
        {"resistance": np.inf},
        {"refractory_period": -1},
    ],
)
def test_lif_invalid_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        StochasticLIFNeuron(**kwargs)


def test_lif_rejects_non_finite_current():
    neuron = StochasticLIFNeuron()
    with pytest.raises(ValueError, match="input_current"):
        neuron.step(np.nan)


@pytest.mark.parametrize(
    ("bits", "message"),
    [
        (np.array([[0, 1]], dtype=np.uint8), "one-dimensional"),
        (np.array([0, 2], dtype=np.uint8), "binary"),
        (np.array([0.0, np.nan]), "finite"),
    ],
)
def test_lif_process_bitstream_rejects_invalid_bits(bits, message):
    neuron = StochasticLIFNeuron()
    with pytest.raises(ValueError, match=message):
        neuron.process_bitstream(bits)


def test_lif_process_bitstream_rejects_non_finite_scale():
    neuron = StochasticLIFNeuron()
    with pytest.raises(ValueError, match="input_scale"):
        neuron.process_bitstream(np.array([0, 1], dtype=np.uint8), input_scale=np.inf)


def test_neurons_lazy_load_model():
    """Accessing a model neuron through sc_neurocore.neurons triggers __getattr__."""
    import sc_neurocore.neurons as neurons_mod

    HH = neurons_mod.HodgkinHuxleyNeuron
    assert HH is not None
    n = HH()
    assert hasattr(n, "step")
    # Second model access triggers the _load_rust_map early-return path
    FHN = neurons_mod.FitzHughNagumoNeuron
    assert FHN is not None


def test_neurons_getattr_invalid():
    """Invalid attribute on neurons package raises AttributeError."""
    import pytest
    import sc_neurocore.neurons as neurons_mod

    with pytest.raises(AttributeError, match="no_such_neuron"):
        neurons_mod.no_such_neuron
