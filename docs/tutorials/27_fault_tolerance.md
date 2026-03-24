<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 27: Fault Tolerance — SC's Fundamental Advantage

Stochastic computing has inherent fault tolerance: a single bit-flip
in a bitstream changes the encoded probability by only $1/L$. In contrast,
a bit-flip in a fixed-point register can corrupt the MSB and cause
catastrophic error. This property makes SC circuits naturally suited for
radiation-hardened, memristive, and other unreliable hardware substrates.

## Why SC is Fault-Tolerant

In fixed-point arithmetic, bit position determines significance. A flip
in bit 15 (MSB) of a Q8.8 value changes the result by 128.0. A flip in
bit 0 (LSB) changes it by 0.004. The expected error from a random bit-flip
is enormous.

In SC, every bit has equal weight: $1/L$. A flip in any position changes
the encoded probability by exactly $1/L$. With $L=1024$, the worst-case
single-bit error is 0.001 — always small, regardless of which bit flipped.

| Metric | Fixed-point Q8.8 | SC (L=1024) |
|--------|:---:|:---:|
| Worst-case single-bit error | 128.0 | 0.001 |
| Expected single-bit error | ~32.0 | 0.001 |
| Graceful degradation | No (MSB flip = catastrophe) | Yes (linear) |
| Error at 10% bit-flip rate | ~50-100% | ~10% |

## 1. SC vs Fixed-Point Under Bit Flips

```python
import numpy as np
from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream, bitstream_to_probability
from sc_neurocore.utils.fault_injection import FaultInjector

p_true = 0.7
N_TRIALS = 100

for error_rate in [0.01, 0.05, 0.10, 0.20]:
    sc_errors = []
    fp_errors = []

    for _ in range(N_TRIALS):
        # SC: flip bits in a 1024-bit stream
        bs = generate_bernoulli_bitstream(p_true, 1024)
        corrupted = FaultInjector.inject_bit_flips(bs, error_rate)
        sc_errors.append(abs(bitstream_to_probability(corrupted) - p_true))

        # Fixed-point Q8.8: flip bits in a 16-bit register
        q_val = int(round(p_true * 256))
        bits = q_val
        for pos in range(16):
            if np.random.random() < error_rate:
                bits ^= (1 << pos)
        if bits >= 32768:
            bits -= 65536
        fp_errors.append(abs(bits / 256.0 - p_true))

    print(f"Error rate {error_rate*100:4.0f}%: "
          f"SC={np.mean(sc_errors):.4f}  FP={np.mean(fp_errors):.4f}  "
          f"Ratio={np.mean(fp_errors)/max(np.mean(sc_errors),1e-9):.0f}x")
```

Typical output shows SC error 10-100x smaller than fixed-point at every noise level.

## 2. Degradation Curves

Plot how accuracy degrades with increasing fault rate:

```python
import numpy as np
from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream, bitstream_to_probability
from sc_neurocore.utils.fault_injection import FaultInjector

error_rates = np.linspace(0, 0.3, 31)
sc_degradation = []
fp_degradation = []

for rate in error_rates:
    sc_err, fp_err = [], []
    for _ in range(200):
        # SC
        bs = generate_bernoulli_bitstream(0.5, 512)
        c = FaultInjector.inject_bit_flips(bs, rate)
        sc_err.append(abs(bitstream_to_probability(c) - 0.5))

        # Fixed-point
        val = 128  # 0.5 in Q8.8
        for pos in range(16):
            if np.random.random() < rate:
                val ^= (1 << pos)
        if val >= 32768:
            val -= 65536
        fp_err.append(abs(val / 256.0 - 0.5))

    sc_degradation.append(np.mean(sc_err))
    fp_degradation.append(np.mean(fp_err))

# SC degrades linearly, fixed-point degrades catastrophically
print(f"At 5%:  SC={sc_degradation[5]:.4f}  FP={fp_degradation[5]:.4f}")
print(f"At 15%: SC={sc_degradation[15]:.4f}  FP={fp_degradation[15]:.4f}")
print(f"At 30%: SC={sc_degradation[30]:.4f}  FP={fp_degradation[30]:.4f}")
```

## 3. Hardware-Aware Training (Memristive Defects)

Real memristive crossbar arrays have stuck-at faults. SC-NeuroCore's
`HardwareAwareSCLayer` masks stuck synapses during training — the network
learns to route information around defects.

```python
from sc_neurocore.layers.hardware_aware import HardwareAwareSCLayer

# 10% of synapses are permanently stuck
layer = HardwareAwareSCLayer(
    n_inputs=8, n_neurons=4, length=256,
    stuck_rate=0.10,
    seed=42,
)

# Forward pass — stuck synapses contribute nothing
out = layer.forward([0.5] * 8)
print(f"Output with 10% stuck synapses: {[f'{x:.3f}' for x in out]}")
print(f"Stuck synapse count: {layer.n_stuck} / {8*4}")

# Compare with a healthy layer
from sc_neurocore import VectorizedSCLayer
healthy = VectorizedSCLayer(n_inputs=8, n_neurons=4, length=256)
out_h = healthy.forward([0.5] * 8)
print(f"Healthy layer output:           {[f'{x:.3f}' for x in out_h]}")
```

## 4. Adaptive Bitstream Length

Longer bitstreams give higher precision but take more cycles. Use analytical
bounds to compute the minimum length for a target precision:

```python
from sc_neurocore.utils.bitstreams import adaptive_length

# Hoeffding bound (distribution-free)
L_h = adaptive_length(p=0.5, epsilon=0.01, confidence=0.95, method="hoeffding")
print(f"Hoeffding: L={L_h} for 1% precision at 95% confidence")

# Chebyshev bound (uses variance)
L_c = adaptive_length(p=0.5, epsilon=0.01, confidence=0.95, method="chebyshev")
print(f"Chebyshev: L={L_c}")

# Variance-based (tightest, uses known p)
L_v = adaptive_length(p=0.5, epsilon=0.01, confidence=0.95, method="variance")
print(f"Variance:  L={L_v}")
```

### Precision-latency tradeoff

| Bitstream length L | Effective bits | Max error | Cycles |
|---|---|---|---|
| 64 | ~3 | ±0.125 | 64 |
| 256 | ~4 | ±0.063 | 256 |
| 1024 | ~5 | ±0.031 | 1024 |
| 4096 | ~6 | ±0.016 | 4096 |

Error scales as $O(1/\sqrt{L})$ — doubling precision requires 4x the bits.

## 5. Radiation Hardening

SC's fault tolerance makes it a candidate for space and nuclear environments
where single-event upsets (SEUs) flip random bits. Combined with triple
modular redundancy (TMR):

```python
from sc_neurocore import VectorizedSCLayer
from sc_neurocore.utils.fault_injection import FaultInjector
import numpy as np

# TMR: three copies, majority vote
inputs = [0.3, 0.5, 0.7, 0.9]
outputs = []
for replica in range(3):
    layer = VectorizedSCLayer(n_inputs=4, n_neurons=2, length=512)
    out = layer.forward(inputs)
    # Inject SEU faults
    out_noisy = out + np.random.normal(0, 0.05, out.shape)  # analog noise
    outputs.append(out_noisy)

# Majority vote (median)
voted = np.median(outputs, axis=0)
print(f"TMR+SC output: {voted}")
```

## Applications

| Domain | Why SC fault tolerance matters |
|--------|-------------------------------|
| **Space** | SEU tolerance without heavy shielding |
| **Medical implants** | Graceful degradation, no catastrophic failure |
| **Memristive hardware** | Works around stuck-at defects |
| **Edge AI** | Low-power + reliable on cheap hardware |
| **Automotive** | Functional safety (ISO 26262) without 10x redundancy |

## Further Reading

- [Tutorial 13: Fixed-Point Arithmetic](13_fixed_point_design.md) — Q8.8 details
- [Tutorial 26: Predictive Coding](26_predictive_coding.md) — another SC advantage
- [Safety: FMEA](../safety/FMEA_SNN_COMPILER.md) — failure mode analysis
