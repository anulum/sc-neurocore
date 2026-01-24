
import sys
import os
import numpy as np
import pytest

# Adjust path to find sc_neurocore
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from sc_neurocore.scpn.layers.l1_quantum import L1_QuantumLayer, L1_StochasticParameters

def test_l1_initialization():
    layer = L1_QuantumLayer()
    assert layer.params.n_qubits == 1000
    assert layer.get_global_metric() > 0.9  # Should start high

def test_l1_dynamics():
    params = L1_StochasticParameters(
        n_qubits=100, 
        bitstream_length=256,
        decoherence_rate=0.1
    )
    layer = L1_QuantumLayer(params)
    
    initial_metric = layer.get_global_metric()
    print(f"Initial Metric: {initial_metric}")
    
    # Run for 10 steps
    for i in range(10):
        layer.step(dt=0.1)
        metric = layer.get_global_metric()
        print(f"Step {i}: Metric = {metric}")
        
    final_metric = layer.get_global_metric()
    
    # Coherence should decay (but not to zero instantly)
    assert final_metric < initial_metric
    assert final_metric > 0.0

if __name__ == "__main__":
    test_l1_dynamics()
