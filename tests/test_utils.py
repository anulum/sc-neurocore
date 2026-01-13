
import pytest
import numpy as np
from sc_neurocore.utils.bitstreams import (
    generate_bernoulli_bitstream,
    bitstream_to_probability,
    value_to_unipolar_prob,
    unipolar_prob_to_value,
    BitstreamEncoder,
    BitstreamAverager
)

def test_generate_bernoulli_bitstream():
    length = 1000
    p = 0.7
    bs = generate_bernoulli_bitstream(p, length)
    assert len(bs) == length
    assert set(np.unique(bs)).issubset({0, 1})
    p_hat = bitstream_to_probability(bs)
    assert np.isclose(p_hat, p, atol=0.05) # Statistical check

def test_conversions():
    x_min = 0.0
    x_max = 10.0
    x = 5.0
    
    p = value_to_unipolar_prob(x, x_min, x_max)
    assert p == 0.5
    
    x_rec = unipolar_prob_to_value(p, x_min, x_max)
    assert x_rec == 5.0
    
    # Test clipping
    p_clip_high = value_to_unipolar_prob(15.0, x_min, x_max)
    assert p_clip_high == 1.0
    
    p_clip_low = value_to_unipolar_prob(-5.0, x_min, x_max)
    assert p_clip_low == 0.0

def test_encoder_class():
    encoder = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1000, seed=42)
    bs = encoder.encode(0.3)
    x_rec = encoder.decode(bs)
    assert np.isclose(x_rec, 0.3, atol=0.05)

def test_averager():
    avg = BitstreamAverager(window=10)
    
    # Push 5 ones
    for _ in range(5):
        avg.push(1)
    
    assert avg.estimate() == 1.0 # 5/5
    
    # Push 5 zeros
    for _ in range(5):
        avg.push(0)
        
    assert avg.estimate() == 0.5 # 5/10
    
    # Push 1 one (overwrite first one)
    avg.push(1) 
    # Buffer was [1,1,1,1,1, 0,0,0,0,0]
    # Now [1,1,1,1,1, 0,0,0,0,0] -> index 0 overwritten? 
    # index was at 0 (after 10 pushes mod 10 = 0).
    # so buffer[0] becomes 1. Array is [1,1,1,1,1, 0,0,0,0,0]. No change in mean.
    
    assert avg.estimate() == 0.5
    
    # Let's verify FIFO.
    avg.reset()
    for _ in range(10):
        avg.push(1)
    assert avg.estimate() == 1.0
    
    avg.push(0) # Replaces oldest 1
    # Buffer sum = 9
    assert avg.estimate() == 0.9
