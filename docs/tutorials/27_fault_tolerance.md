# Tutorial 27: Fault Tolerance — SC's Fundamental Advantage

Stochastic computing has inherent fault tolerance: a single bit-flip
in a bitstream changes the encoded probability by only 1/L. In contrast,
a bit-flip in a fixed-point register can corrupt the MSB and cause
catastrophic error.

## SC vs Fixed-Point Under Bit Flips

```python
import numpy as np
from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream, bitstream_to_probability
from sc_neurocore.utils.fault_injection import FaultInjector

p_true = 0.7
error_rate = 0.1  # 10% bit-flip rate

# SC: flip 10% of a 1024-bit stream
bs = generate_bernoulli_bitstream(p_true, 1024)
corrupted = FaultInjector.inject_bit_flips(bs, error_rate)
sc_error = abs(bitstream_to_probability(corrupted) - p_true)

# Fixed-point Q8.8: flip 10% of 16 bits
q_val = int(round(p_true * 256))
bits = q_val
for pos in range(16):
    if np.random.random() < error_rate:
        bits ^= (1 << pos)
if bits >= 32768:
    bits -= 65536
fp_error = abs(bits / 256.0 - p_true)

print(f"SC error:          {sc_error:.4f}")
print(f"Fixed-point error: {fp_error:.4f}")
# SC error is typically 10-100x smaller
```

## Hardware-Aware Training

Train around memristive defects (stuck-at faults):

```python
from sc_neurocore.layers.hardware_aware import HardwareAwareSCLayer

layer = HardwareAwareSCLayer(
    n_inputs=8, n_neurons=4, length=256,
    stuck_rate=0.1,  # 10% of synapses are stuck
    seed=42,
)

# Stuck synapses receive zero gradient — network learns around defects
out = layer.forward([0.5] * 8)
print(f"Output with 10% stuck synapses: {out}")
print(f"Stuck synapse count: {layer.n_stuck}")
```

## Adaptive Bitstream Length

Compute minimum length for target precision:

```python
from sc_neurocore.utils.bitstreams import adaptive_length

L = adaptive_length(p=0.5, epsilon=0.01, confidence=0.95, method="hoeffding")
print(f"Need {L} bits for 1% precision at 95% confidence")
```
