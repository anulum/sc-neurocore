
import pytest
import numpy as np
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
    
    neuron.v = 1.0 # Set initial potential
    neuron.step(0.0) # No input
    
    # Expected decay: dv = -(v - v_rest) * (dt / tau)
    # dv = -(1.0 - 0.0) * (0.1) = -0.1
    # v_new = 0.9
    assert np.isclose(neuron.v, 0.9)

def test_lif_noise():
    # Test that noise is doing something
    neuron = StochasticLIFNeuron(noise_std=0.5, seed=42)
    neuron.step(0.0)
    assert neuron.v != 0.0 # Should have moved due to noise
