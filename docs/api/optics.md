# Optics — Photonic Stochastic Computing

Simulated photonic SC layer using laser interference for bitstream generation. Photonic SC uses physical randomness (laser phase noise) instead of LFSR pseudo-randomness, giving truly uncorrelated bitstreams.

## Theory

Two coherent laser beams with phase noise φ produce interference intensity:

`I = I₁ + I₂ + 2√(I₁I₂) cos(φ)`

Normalized: `I_norm = 0.5 + 0.5 * cos(φ)` where φ ~ Uniform(0, 2π).

Bitstream generation: a bit is 1 if the interference intensity falls below the input probability. This maps naturally to photonic hardware where a photodetector and comparator produce the output bit.

## Components

- **`PhotonicBitstreamLayer`** — Multi-channel photonic SC layer.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_channels` | (required) | Number of independent optical channels |
| `laser_power` | 1.0 | SNR control (reserved for future noise model) |

**Methods:**

- `simulate_interference(length)` — Generate intensity patterns of shape (n_channels, length)
- `forward(input_probs, length)` — Generate bitstreams where P(bit=1) ≈ input_prob per channel

## Usage

```python
from sc_neurocore.optics.photonic_layer import PhotonicBitstreamLayer
import numpy as np

layer = PhotonicBitstreamLayer(n_channels=4)

# Generate photonic bitstreams
probs = np.array([0.2, 0.4, 0.6, 0.8])
bitstreams = layer.forward(probs, length=10000)
print(f"Measured rates: {bitstreams.mean(axis=1)}")
# ≈ [0.20, 0.40, 0.60, 0.80]

# Raw interference pattern
intensity = layer.simulate_interference(length=1000)
print(f"Intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
```

::: sc_neurocore.optics.photonic_layer
    options:
      show_root_heading: true
