# Tutorial 64: Spike Encoding Zoo + Auto-Optimizer

7 spike encoding schemes convert continuous signals into spike trains.
Each has different trade-offs: rate coding is robust but energy-hungry,
latency coding is efficient but fragile. The auto-optimizer recommends
the best encoding for your specific data and task.

## The Encoding Decision

Every SNN starts with encoding: how do you convert a continuous input
(pixel value, sensor reading, voltage) into binary spikes? The wrong
encoding wastes energy or loses information.

| Encoding | Spikes/Value | Energy | Precision | Best For |
|----------|:-----------:|:------:|:---------:|----------|
| Rate | T (many) | High | High | General, robust |
| Latency | 1 | Very low | Medium | Classification, speed |
| Delta | ~0.1T | Low | High (changes) | Temporal signals |
| Phase | 1 | Low | Medium | Periodic signals |
| Burst | 1-T | Medium | High | Intensity-critical |
| Rank-order | 1 | Very low | Low | High-dimensional input |
| Sigma-delta | ~0.1T | Low | Very high | ADC replacement |

## All 7 Encoders

```python
from sc_neurocore.encoding import (
    rate_encode, latency_encode, delta_encode,
    phase_encode, burst_encode, rank_order_encode, sigma_delta_encode,
)
import numpy as np

values = np.array([0.2, 0.5, 0.8, 0.1], dtype=np.float32)
T = 20  # timesteps

# Rate coding: probability of spike per timestep = value
rate_spikes = rate_encode(values, T, seed=42)
print(f"Rate: {rate_spikes.sum(axis=0)} spikes per channel")

# Latency coding: time-to-first-spike inversely proportional to value
latency_spikes = latency_encode(values, T)
first_spikes = [np.argmax(latency_spikes[:, i]) for i in range(4)]
print(f"Latency: first spikes at {first_spikes}")

# Phase coding: spike position within oscillation cycle
phase_spikes = phase_encode(values, T)

# Burst coding: number of spikes in burst proportional to value
burst_spikes = burst_encode(values, T)

# Rank-order coding: only the first spike matters (order encodes info)
rank_spikes = rank_order_encode(values, T)

# For temporal signals (not static values):
signal = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float32)

# Delta coding: spike on change exceeding threshold
delta_spikes = delta_encode(signal, threshold=0.2)
print(f"Delta: {delta_spikes.sum()} spikes for 100-sample signal")

# Sigma-delta coding: integrating delta modulator (ADC-like)
sd_spikes = sigma_delta_encode(signal, threshold=0.1)
print(f"Sigma-delta: {sd_spikes.sum()} spikes")
```

## Auto-Optimizer

Don't guess — let the optimizer analyse your data and recommend the
best encoding:

```python
from sc_neurocore.encoding import EncodingOptimizer

rng = np.random.default_rng(42)
my_data = rng.standard_normal((100, 32)).astype(np.float32)

opt = EncodingOptimizer(T=32)
recs = opt.recommend(my_data)

print("Encoding recommendations (best first):")
for r in recs:
    print(f"  {r.encoding:15s}: score={r.score:.3f}, "
          f"sparsity={r.sparsity:.2f}, info_retained={r.info_retained:.2f}")
    print(f"    {r.reason}")
```

### How the Optimizer Decides

The optimizer evaluates each encoding on three criteria:

1. **Information retention:** mutual information between encoded spikes
   and original signal
2. **Sparsity:** average spike rate (lower = more energy-efficient)
3. **Reconstruction error:** how well the signal can be decoded back

The combined score balances these. For classification tasks, information
retention dominates. For FPGA deployment, sparsity (energy) dominates.

## FPGA Deployment

Encoders map to simple FPGA circuits:

| Encoder | FPGA Implementation | LUTs |
|---------|-------------------|:-----:|
| Rate | LFSR + comparator | ~10 |
| Latency | Counter + comparator | ~8 |
| Delta | Subtractor + threshold | ~12 |
| Sigma-delta | Integrator + threshold | ~16 |

## References

- Guo et al. (2021). "Neural Coding in Spiking Neural Networks: A
  Comparative Study for Robust Neuromorphic Systems." Front. Neurosci.
- Kim et al. (2020). "Spiking-YOLO: Spiking Neural Network for
  Energy-Efficient Object Detection." AAAI 2020.
