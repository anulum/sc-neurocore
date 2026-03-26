# Digital Twin — FPGA Mismatch Simulation

Simulate FPGA hardware imperfections during SNN training. Training through these imperfections produces networks that tolerate hardware mismatch at deployment time.

## Hardware Imperfection Model

Real FPGA implementations suffer from:

| Imperfection | Source | Typical Magnitude |
|-------------|--------|-------------------|
| Q8.8 quantization | Fixed-point arithmetic | Step = 1/256 |
| Weight perturbation | Process variation in LUT/BRAM | CV ≈ 2% |
| Threshold mismatch | Per-neuron comparator variation | CV ≈ 5% |
| Clock jitter | PLL/oscillator noise | ±1% period |
| Routing delay | Path-dependent timing skew | Clipped to ±10% |

Calibrated against published data: ~20% coefficient of variation for analog mixed-signal neuromorphic processors. Digital FPGAs have lower variation (~1-5%) but Q8.8 quantization is the dominant error source.

## Components

- **`FPGAMismatchModel`** — Wraps weight matrices and neuron parameters with hardware imperfections.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `quantization_bits` | 16 | Fixed-point width (16 = Q8.8) |
| `weight_cv` | 0.02 | Weight perturbation coefficient of variation |
| `threshold_cv` | 0.05 | Per-neuron threshold variation |
| `clock_jitter_pct` | 0.01 | Clock period variation |
| `seed` | 42 | RNG seed for reproducibility |

**Methods:**

- `quantize(values)` — Apply Q-format quantization
- `perturb_weights(weights)` — Add process variation + quantize
- `perturb_thresholds(thresholds)` — Add per-neuron mismatch + quantize
- `jitter_timing(n_steps)` — Generate per-step timing variation (clipped to [0.9, 1.1])
- `apply_to_network_weights(weights)` — Apply all imperfections to a list of weight matrices
- `mismatch_report(weights)` — Report expected error statistics

## Usage

```python
from sc_neurocore.digital_twin import FPGAMismatchModel
import numpy as np

model = FPGAMismatchModel(quantization_bits=16, weight_cv=0.02)

# Apply to trained weights
weights = [np.random.randn(64, 32) * 0.1, np.random.randn(10, 64) * 0.1]
faulted = model.apply_to_network_weights(weights)

# Get error report
report = model.mismatch_report(weights)
print(f"MAE: {report['mean_absolute_error']:.6f}")
print(f"Max error: {report['max_absolute_error']:.6f}")
```

See [Tutorial 48: Digital Twin](../tutorials/48_digital_twin.md).

::: sc_neurocore.digital_twin
    options:
      show_root_heading: true
