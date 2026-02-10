
import pytest
import numpy as np
from sc_neurocore.synapses.sc_synapse import BitstreamSynapse
from sc_neurocore.utils.bitstreams import bitstream_to_probability

def test_synapse_encoding():
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=1000, w=0.5, seed=42)
    assert len(syn.weight_bits) == 1000
    p_eff = syn.effective_weight_probability()
    assert np.isclose(p_eff, 0.5, atol=0.05)

def test_synapse_multiplication():
    # P(A AND B) = P(A) * P(B) if independent
    length = 10000
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=length, w=0.5, seed=42)
    
    # Create input with p=0.8
    # We use a different seed implicitly or manually to ensure independence
    # BitstreamSynapse uses an internal RNG for encoding.
    # Let's generate an input stream.
    from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream
    input_bits = generate_bernoulli_bitstream(0.8, length)
    
    output_bits = syn.apply(input_bits)
    
    p_out = bitstream_to_probability(output_bits)
    expected = 0.5 * 0.8 # 0.4
    
    assert np.isclose(p_out, expected, atol=0.05)

def test_synapse_update():
    syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=100, w=0.2)
    initial_p = syn.effective_weight_probability()
    
    syn.update_weight(0.8)
    new_p = syn.effective_weight_probability()
    
    assert not np.isclose(initial_p, new_p, atol=0.1)
    assert np.isclose(new_p, 0.8, atol=0.1)
