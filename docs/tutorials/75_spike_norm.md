# Tutorial 75: Spike Normalisation

Five SNN-specific batch normalisation variants that handle binary
activations and temporal dynamics correctly. Standard BN fails in SNNs
because spike activations are binary and statistics shift across
timesteps (temporal covariate shift). These normalizers fix that.

No other SNN library ships all five as reusable modules.

## The Problem

Standard batch normalisation assumes:
1. Continuous activations → violated (spikes are binary)
2. Stable statistics → violated (distribution changes every timestep)

Naively applying BN to SNNs degrades CIFAR-10 accuracy by 5-15%
(Zheng et al. 2021). The five normalizers below are purpose-built for
spiking networks.

## Available Normalizers

| Normalizer | Key Idea | Inference Cost | Reference |
|-----------|----------|:--------------:|-----------|
| `ThresholdDependentBN` | Incorporates threshold into statistics | 1 multiply/add | Zheng 2021 (AAAI) |
| `PerTimestepBN` | Separate statistics per timestep | 1 multiply/add | Kim & Panda 2021 |
| `TemporalEffectiveBN` | Per-timestep learned scaling factor | 1 multiply/add | Duan 2022 (NeurIPS) |
| `MembranePotentialBN` | BN on membrane → folds into threshold | **Zero** | Guo 2023 (ICCV) |
| `TemporalAccumulatedBN` | Normalizes accumulated membrane | 1 multiply/add | Jiang 2024 (ICLR) |

**Recommendation for FPGA: `MembranePotentialBN`** — folds into
threshold at inference, zero overhead.

## ThresholdDependentBN

Standard BN ignores the spike threshold when computing statistics.
tdBN incorporates it — the running mean and variance are conditioned
on threshold-relative membrane potential:

```python
import numpy as np
from sc_neurocore.spike_norm import ThresholdDependentBN

rng = np.random.RandomState(42)

tdbn = ThresholdDependentBN(n_features=64, threshold=1.0)

# Training: update running statistics
for step in range(100):
    x = rng.randn(32, 64).astype(np.float32)  # batch=32, features=64
    x_norm = tdbn.forward(x, training=True)

# Inference: use frozen statistics
x_test = rng.randn(8, 64).astype(np.float32)
x_norm = tdbn.forward(x_test, training=False)
print(f"Output range: [{x_norm.min():.3f}, {x_norm.max():.3f}]")
```

## PerTimestepBN (BNTT)

Maintains separate running statistics for each timestep. This handles
temporal covariate shift — the distribution at t=0 (no spikes yet) is
very different from t=25 (steady-state activity):

```python
from sc_neurocore.spike_norm import PerTimestepBN

bntt = PerTimestepBN(n_features=64, T=10)

# Each timestep gets its own BN statistics
for t in range(10):
    x_t = rng.randn(32, 64).astype(np.float32)
    out_t = bntt.forward(x_t, t=t, training=True)
    print(f"t={t}: mean={out_t.mean():.4f}, std={out_t.std():.4f}")
```

## TemporalEffectiveBN

Adds a learned per-timestep scaling factor on top of standard BN. The
network learns how much to trust each timestep's contribution:

```python
from sc_neurocore.spike_norm import TemporalEffectiveBN

tebn = TemporalEffectiveBN(n_features=64, T=10)

for t in range(10):
    x_t = rng.randn(32, 64).astype(np.float32)
    out_t = tebn.forward(x_t, t=t, training=True)
```

## MembranePotentialBN (Recommended for FPGA)

The key insight: normalise the membrane potential *before* the spike
function, then at inference, fold BN parameters into a per-neuron
threshold. **Zero compute overhead at inference.**

```python
from sc_neurocore.spike_norm import MembranePotentialBN

mpbn = MembranePotentialBN(n_features=64, threshold=1.0)

# Training: standard BN on membrane potential
for step in range(200):
    x = rng.randn(32, 64).astype(np.float32)
    x_norm = mpbn.forward(x, training=True)

# At inference: fold BN into threshold
hw_thresholds = mpbn.fused_threshold()
print(f"Hardware thresholds: shape={hw_thresholds.shape}")
print(f"Range: [{hw_thresholds.min():.3f}, {hw_thresholds.max():.3f}]")
# Each neuron gets its own threshold — no BN computation at runtime
```

### How Threshold Fusion Works

BN computes: `y = gamma * (x - mean) / sqrt(var + eps) + beta`

The spike decision is: `spike = (y > V_th)`

Combining:
```
spike = (gamma * (x - mean) / sqrt(var + eps) + beta > V_th)
spike = (x > (V_th - beta) * sqrt(var + eps) / gamma + mean)
spike = (x > new_threshold)
```

The fused threshold absorbs all BN parameters. On FPGA, this is just a
different threshold value per neuron — zero additional LUTs.

## TemporalAccumulatedBN

Normalizes the accumulated membrane potential across time, not
instantaneous values. Better for temporal coding where spike timing
(not just rate) carries information:

```python
from sc_neurocore.spike_norm import TemporalAccumulatedBN

tabn = TemporalAccumulatedBN(n_features=64, T=10)

accumulated = np.zeros((32, 64), dtype=np.float32)
for t in range(10):
    x_t = rng.randn(32, 64).astype(np.float32)
    accumulated += x_t
    out_t = tabn.forward(accumulated, t=t, training=True)
```

## Comparison on CIFAR-10

Published results (not our measurements):

| Normalizer | CIFAR-10 Accuracy | Inference Overhead |
|-----------|:-----------------:|:------------------:|
| No normalisation | 89.2% | 0 |
| Standard BN | 84.1% (-5.1%) | 1 multiply/add |
| ThresholdDependentBN | 93.1% (+3.9%) | 1 multiply/add |
| PerTimestepBN | 92.8% (+3.6%) | 1 multiply/add |
| TemporalEffectiveBN | 93.4% (+4.2%) | 1 multiply/add |
| MembranePotentialBN | 93.0% (+3.8%) | **Zero** |

## References

- Zheng et al. (2021). "Going Deeper with Directly-Trained Larger
  Spiking Neural Networks." AAAI 2021.
- Kim & Panda (2021). "Revisiting Batch Normalization for Training
  Low-Latency Deep Spiking Neural Networks from Scratch." Front.
  Neurosci. 15:773954.
- Duan et al. (2022). "Temporal Effective Batch Normalization in
  Spiking Neural Networks." NeurIPS 2022.
- Guo et al. (2023). "Membrane Potential Batch Normalization for
  Spiking Neural Networks." ICCV 2023.
- Jiang et al. (2024). "Towards Accurate and Efficient SNNs via
  Temporal Accumulated Batch Normalization." ICLR 2024.
