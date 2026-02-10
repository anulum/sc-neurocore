
import pytest
import numpy as np
from sc_neurocore.sources.bitstream_current_source import BitstreamCurrentSource
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
from sc_neurocore.recorders.spike_recorder import BitstreamSpikeRecorder

def test_full_chain_integration():
    """
    Test Source -> Neuron -> Recorder chain.
    """
    # 1. Setup Source
    # 3 inputs, weights all 1.0. 
    # Inputs: [0.8, 0.8, 0.8] -> High current
    x_inputs = [0.8, 0.8, 0.8]
    weights = [1.0, 1.0, 1.0]
    
    # Map [0,1] input to [0,1] probability.
    # Map dot product output probability to current range [0, 2.0]
    # Max probability sum = 3 (if not normalized). 
    # BitstreamCurrentSource.step() calculates 'prob = n_ones / n_inputs'.
    # So max prob is 1.0.
    
    source = BitstreamCurrentSource(
        x_inputs=x_inputs,
        x_min=0.0,
        x_max=1.0,
        weight_values=weights,
        w_min=0.0,
        w_max=1.0,
        length=1000,
        y_min=0.0,
        y_max=2.0, # High enough to drive neuron
        seed=42
    )
    
    # 2. Setup Neuron
    # Threshold 1.0, No leak (infinite tau) -> Simple integrator
    neuron = StochasticLIFNeuron(
        v_threshold=1.0,
        tau_mem=1e9, 
        dt=1.0, 
        resistance=1.0,
        seed=42
    )
    
    # 3. Setup Recorder
    recorder = BitstreamSpikeRecorder(dt_ms=1.0)
    
    # 4. Run loop
    for _ in range(1000):
        # Get current I(t) from source (decoding the bitstream at this step)
        i_in = source.step()
        
        # Step neuron
        spike = neuron.step(i_in)
        
        # Record
        recorder.record(spike)
        
    # 5. Verify
    # With inputs 0.8 and weights 1.0, AND logic -> p_out approx 0.8.
    # BitstreamCurrentSource.step averages the bits from the 3 channels.
    # So effective prob ~ 0.8.
    # Current mapped to [0, 2.0]. I ~ 0.0 + 0.8 * (2.0 - 0.0) = 1.6.
    # Neuron integrates 1.6 per step. Threshold 1.0.
    # Should spike every step (reset to 0)?
    # Step 1: v=1.6 -> Spike -> v=0.
    # Step 2: v=1.6 -> Spike...
    # So firing rate should be ~100% or 1000 Hz if dt=1ms.
    
    rate = recorder.firing_rate_hz()
    # 1000 Hz = 1 spike per ms
    # Allow some stochastic variation from bitstream generation
    print(f"Firing Rate: {rate} Hz")
    assert rate > 800.0 # Expect high firing rate

def test_low_input_response():
    """
    Test that low input results in low/no firing.
    """
    x_inputs = [0.1, 0.1, 0.1]
    weights = [1.0, 1.0, 1.0]
    
    source = BitstreamCurrentSource(
        x_inputs=x_inputs,
        x_min=0.0,
        x_max=1.0,
        weight_values=weights,
        w_min=0.0,
        w_max=1.0,
        length=1000,
        y_min=0.0,
        y_max=2.0,
        seed=42
    )
    
    # Neuron with leak this time
    neuron = StochasticLIFNeuron(
        v_threshold=1.0,
        tau_mem=10.0, 
        dt=1.0, 
        resistance=1.0
    )
    
    recorder = BitstreamSpikeRecorder()
    
    for _ in range(1000):
        i_in = source.step()
        spike = neuron.step(i_in)
        recorder.record(spike)
        
    rate = recorder.firing_rate_hz()
    print(f"Low Input Firing Rate: {rate} Hz")
    assert rate < 200.0 # Significantly lower than previous test
