<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 44: SNN Model Compression

Trained SNN models are often too large for small FPGAs or edge
neuromorphic chips. Model compression reduces resource usage through
pruning, quantisation, and automatic target-aware optimisation — while
preserving accuracy.

SC-NeuroCore provides a compression pipeline that goes from a trained
PyTorch SNN directly to a size-optimised FPGA deployment.

## Why Compress?

| Target | LUTs Available | Typical SNN Requirement | Fits? |
|--------|---------------|------------------------|-------|
| iCE40 UP5K | 5,280 | 2,000–50,000 | Maybe |
| ECP5 25K | 24,576 | 2,000–50,000 | Usually |
| Artix-7 35T | 20,800 | 2,000–50,000 | Usually |

A 784→256→128→10 MNIST SNN has ~230K weights. At Q8.8 (16 bits each),
that's 3.7 Mbit — too large for iCE40's 30 BRAMs (240 Kbit). After
pruning (90% sparsity) and quantisation (4-bit), it fits in 37 Kbit.

## 1. Weight Pruning

Remove near-zero synaptic weights. SNN weights are naturally sparse —
many connections have negligible effect on firing patterns.

```python
import numpy as np
from sc_neurocore.compression import prune_weights

# Simulate trained weights (3 layers)
weights = [
    np.random.randn(784, 256) * 0.05,  # input → hidden1
    np.random.randn(256, 128) * 0.1,   # hidden1 → hidden2
    np.random.randn(128, 10) * 0.2,    # hidden2 → output
]

# Prune weights below threshold
pruned, report = prune_weights(weights, threshold=0.05)
print(f"Sparsity: {report.sparsity:.1%}")
print(f"Parameters: {report.original_params:,} → {report.remaining_params:,}")
print(f"Memory reduction: {report.compression_ratio:.1f}×")
```

### Magnitude vs Activity-Based Pruning

Magnitude pruning removes small weights. Activity-based pruning removes
weights connected to low-activity neurons — more targeted for SNNs:

```python
from sc_neurocore.compression import prune_weights

# Activity-based: monitor spike rates during validation,
# prune weights connected to neurons with <5% firing rate
pruned, report = prune_weights(
    weights,
    method="activity",
    activity_threshold=0.05,
    spike_rates=monitored_rates,  # from SpikeMonitor
)
```

## 2. Structural Pruning

Remove entire neurons (not just individual weights). This reduces
network dimensions, directly cutting LUT and FF usage:

```python
from sc_neurocore.compression import prune_neurons

# Remove neurons with <5% average firing rate
pruned_weights, report = prune_neurons(weights, activity_threshold=0.05)
print(f"Removed {report.pruned_neurons} neurons")
print(f"New dimensions: {report.new_dimensions}")
# e.g., [784, 196, 112, 10] — hidden layers shrunk
```

## 3. Weight Quantisation

Reduce weight precision from float32 to lower bit-widths:

```python
from sc_neurocore.compression.quantization import quantize_weights

# Q8.8 (16-bit fixed-point) — SC-NeuroCore's default hardware format
q16 = quantize_weights(weights, bits=16, scheme="fixed_point")

# Q4.4 (8-bit) — aggressive, 2× compression vs Q8.8
q8 = quantize_weights(weights, bits=8, scheme="fixed_point")

# Binary weights (1-bit) — extreme compression, XOR-based compute
q1 = quantize_weights(weights, bits=1, scheme="binary")

# Ternary weights {-1, 0, +1} — 2-bit, no multipliers needed
q2 = quantize_weights(weights, bits=2, scheme="ternary")
```

### Quantisation Impact on Accuracy

Measured on MNIST SNN (784→128→10, 25 timesteps):

| Precision | Bits/Weight | Accuracy | Memory | LUTs (est.) |
|-----------|-------------|----------|--------|-------------|
| float32 | 32 | 97.2% | 400 KB | — |
| Q8.8 | 16 | 97.0% | 200 KB | baseline |
| Q4.4 | 8 | 96.5% | 100 KB | 0.6× |
| Ternary | 2 | 94.1% | 25 KB | 0.2× |
| Binary | 1 | 91.8% | 12.5 KB | 0.1× |

These numbers are from our training benchmarks (see
`benchmarks/results/`), not estimates.

## 4. Delay Quantisation

Learnable delays (from `DelayLinear`) are float-valued during training.
For hardware, they must be integer clock cycles:

```python
from sc_neurocore.compression.quantization import quantize_delays

delays = np.array([1.3, 2.7, 4.1, 0.8])
quantised = quantize_delays(delays, resolution=2)
# [2, 4, 4, 2] — rounded to nearest multiple of resolution
```

## 5. Auto-Optimise for Target FPGA

Automatically select pruning threshold and quantisation level to fit
a specific FPGA target:

```python
from sc_neurocore.optimizer import fit_to_target

result = fit_to_target(
    layer_shapes=[(784, 128), (128, 10)],
    weights=weights,
    target="ice40",  # or "ecp5", "artix7"
)

print(result.summary())
# Target: ice40 (5280 LUTs, 30 BRAMs)
# Selected: Q4.4, 85% pruning, structural (128→96 hidden)
# Estimated: 2400 LUTs, 12 BRAMs — FITS
# Accuracy impact: -0.7% (from validation set)
```

## 6. End-to-End: Train → Compress → Deploy

```python
from sc_neurocore.training import SpikingNet, train_epoch
from sc_neurocore.compression import prune_weights
from sc_neurocore.compression.quantization import quantize_weights

# 1. Train
model = SpikingNet(n_input=784, n_hidden=128, n_output=10)
# ... training loop ...

# 2. Export SC weights
sc_weights = model.to_sc_weights()

# 3. Prune
pruned, _ = prune_weights(
    [w["weight"] for w in sc_weights],
    threshold=0.05,
)

# 4. Quantise
quantised = quantize_weights(pruned, bits=8)

# 5. In the Studio:
#    - Load the quantised weights
#    - Click Pipeline → ice40
#    - View resource utilisation with compression applied
```

## Comparison with Other Frameworks

| Feature | SC-NeuroCore | snnTorch | Norse | Lava |
|---------|:-----------:|:-------:|:-----:|:----:|
| Weight pruning | Yes | Manual | No | No |
| Structural pruning | Yes | No | No | No |
| Quantisation (multi-level) | 1/2/4/8/16-bit | No | No | 8-bit |
| FPGA-aware auto-optimise | Yes | No | No | Loihi-aware |
| Delay quantisation | Yes | No | No | No |

## References

- Han et al. (2016). "Deep Compression: Compressing Deep Neural
  Networks with Pruning, Trained Quantization and Huffman Coding." ICLR.
- Kundu et al. (2021). "Spike-Thrift: Towards Energy-Efficient Deep
  Spiking Neural Networks by Limiting Spiking Activity via Attention-
  Guided Compression." WACV 2021.
- Kim et al. (2022). "Exploring Lottery Ticket Hypothesis in Spiking
  Neural Networks." ECCV 2022.
