# Tutorial 64: Spike Encoding Zoo + Auto-Optimizer

7 encoding schemes and automatic selection.

## All 7 Encoders

```python
from sc_neurocore.encoding import (
    rate_encode, latency_encode, delta_encode,
    phase_encode, burst_encode, rank_order_encode, sigma_delta_encode,
)
import numpy as np

values = np.array([0.2, 0.5, 0.8, 0.1])
T = 20

rate_spikes = rate_encode(values, T)           # Poisson rate coding
latency_spikes = latency_encode(values, T)     # Time-to-first-spike
phase_spikes = phase_encode(values, T)         # Phase within oscillation
burst_spikes = burst_encode(values, T)         # Burst length
rank_spikes = rank_order_encode(values, T)     # Firing order

signal = np.sin(np.linspace(0, 4*np.pi, 100))
delta_spikes = delta_encode(signal, threshold=0.2)
sd_spikes = sigma_delta_encode(signal, threshold=0.1)
```

## Auto-Optimizer

```python
from sc_neurocore.encoding import EncodingOptimizer

opt = EncodingOptimizer(T=32)
recs = opt.recommend(my_data)

for r in recs:
    print(f"{r.encoding}: score={r.score:.2f}, sparsity={r.sparsity:.2f}")
    print(f"  {r.reason}")
```

## When to Use Each

| Encoding | Best For | Energy |
|----------|---------|--------|
| Rate | General purpose, robust | High (many spikes) |
| Latency | Classification, low-latency | Very low (1 spike) |
| Delta | Temporal change detection | Low (sparse) |
| Phase | Periodic signals, oscillations | Medium |
| Burst | Intensity-critical tasks | Medium |
| Rank-order | High-dimensional input | Very low |
| Sigma-delta | Continuous signals, ADC-like | Low |
