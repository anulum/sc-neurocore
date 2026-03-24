<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 48: Hardware Digital Twin

Train through simulated FPGA imperfections so your SNN tolerates
hardware mismatch at deployment time.

## 1. Create Mismatch Model

```python
from sc_neurocore.digital_twin import FPGAMismatchModel

twin = FPGAMismatchModel(
    weight_cv=0.03,         # 3% weight variation
    threshold_cv=0.05,      # 5% threshold mismatch
    clock_jitter_pct=0.01,  # 1% clock jitter
)
```

## 2. Apply to Weights

```python
import numpy as np
weights = [np.random.randn(8, 4) * 0.5]
perturbed = twin.apply_to_network_weights(weights)
report = twin.mismatch_report(weights)
print(f"MAE: {report['mean_absolute_error']:.6f}")
```

## 3. Q8.8 Quantization

```python
values = np.array([0.123456789, 0.5, 1.0])
quantized = twin.quantize(values)
# [0.125, 0.5, 1.0] -- Q8.8 rounds to nearest 1/256
```

## 4. Train Through Mismatch

Apply mismatch each training iteration. The network learns to be
robust to hardware variation:

```python
for epoch in range(100):
    noisy_w = twin.apply_to_network_weights(trained_weights)
    # Forward with noisy, backward updates clean weights
```

## Further Reading

- [Tutorial 27: Fault Tolerance](27_fault_tolerance.md)
- [API: Digital Twin](../api/digital_twin.md)
