# Tutorial 75: Spike Normalization

5 SNN-specific batch normalization variants. Standard BN fails in SNNs because
spike activations are binary and statistics shift across timesteps. These
normalizers handle temporal dynamics, threshold interaction, and inference
re-parameterization (zero-overhead deployment).

No other SNN library ships these as reusable modules.

## The Problem

Standard batch normalization assumes continuous activations with stable statistics.
SNNs violate both assumptions: activations are binary spikes, and the distribution
changes at every timestep (temporal covariate shift). Naively applying BN to SNNs
degrades accuracy by 5-15% on CIFAR-10 (Zheng 2021).

## Available Normalizers

| Normalizer | Key Idea | Reference |
|-----------|----------|-----------|
| `ThresholdDependentBN` | Incorporates firing threshold into normalization | Zheng 2021 |
| `PerTimestepBN` | Separate BN statistics per timestep | Kim & Panda 2021 |
| `TemporalEffectiveBN` | Per-timestep scaling factor on top of BN | Duan 2022 (NeurIPS) |
| `MembranePotentialBN` | BN on membrane, folds into threshold at inference | Guo 2023 (ICCV) |
| `TemporalAccumulatedBN` | Normalizes accumulated membrane across time | Jiang 2024 (ICLR) |

## Quick Start

```python
import numpy as np
from sc_neurocore.spike_norm import (
    ThresholdDependentBN,
    PerTimestepBN,
    TemporalEffectiveBN,
    MembranePotentialBN,
    TemporalAccumulatedBN,
)

# Simulated batch of presynaptic currents: (batch=32, features=64)
rng = np.random.RandomState(42)
x = rng.randn(32, 64)

# tdBN: threshold-aware normalization
tdbn = ThresholdDependentBN(n_features=64, threshold=1.0)
x_norm = tdbn.forward(x, training=True)

# BNTT: different statistics per timestep
bntt = PerTimestepBN(n_features=64, T=10)
for t in range(10):
    x_t = rng.randn(32, 64)
    out_t = bntt.forward(x_t, t=t, training=True)

# MPBN: fuse into threshold at inference (zero overhead)
mpbn = MembranePotentialBN(n_features=64, threshold=1.0)
for _ in range(100):
    mpbn.forward(rng.randn(32, 64), training=True)
hw_thresholds = mpbn.fused_threshold()  # shape (64,)
# No BN computation at inference — threshold absorbs it
```

## MPBN: Zero-Overhead Inference

`MembranePotentialBN` is recommended for hardware deployment. At inference, BN
parameters fold into a per-neuron threshold:

```
new_threshold[i] = (V_th - beta[i]) * sqrt(var[i] + eps) / gamma[i] + mean[i]
```

Identical behavior to training BN with zero compute overhead.

## API Reference

::: sc_neurocore.spike_norm.normalizers
    options:
      show_root_heading: true
