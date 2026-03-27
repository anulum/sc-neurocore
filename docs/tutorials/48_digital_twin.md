<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 48: Hardware Digital Twin

Train your SNN through simulated FPGA imperfections so it tolerates
hardware mismatch at deployment time. The digital twin models weight
variation, threshold mismatch, clock jitter, and Q8.8 quantisation
noise — all during training, not after.

## Why a Digital Twin

FPGA hardware introduces errors that don't exist in simulation:

| Error Source | Magnitude | Effect |
|-------------|-----------|--------|
| Weight variation (process) | 1-5% CV | Shifted synaptic strengths |
| Threshold mismatch | 3-10% CV | Different firing rates per neuron |
| Clock jitter | 0.1-1% | Temporal noise in spike timing |
| Q8.8 quantisation | ±1/256 per value | Rounding noise everywhere |

Training without accounting for these produces models that work
in simulation but fail on silicon. The digital twin injects these
errors *during training* so the network learns to be robust.

## 1. Create Mismatch Model

```python
from sc_neurocore.digital_twin import FPGAMismatchModel
import numpy as np

twin = FPGAMismatchModel(
    weight_cv=0.03,         # 3% weight coefficient of variation
    threshold_cv=0.05,      # 5% threshold mismatch
    clock_jitter_pct=0.01,  # 1% clock jitter
)
```

## 2. Apply to Weights

```python
rng = np.random.default_rng(42)
weights = [rng.standard_normal((64, 32)).astype(np.float32) * 0.5]

# Apply hardware-realistic perturbation
perturbed = twin.apply_to_network_weights(weights)

report = twin.mismatch_report(weights)
print(f"Weight MAE: {report['mean_absolute_error']:.6f}")
print(f"Max perturbation: {report['max_perturbation']:.6f}")
print(f"Affected weights: {report['affected_fraction']:.1%}")
```

## 3. Q8.8 Quantisation

SC-NeuroCore's hardware uses Q8.8 fixed-point (8 integer bits, 8
fraction bits, resolution 1/256 ≈ 0.0039):

```python
values = np.array([0.123456789, 0.5, 1.0, -0.333])
quantised = twin.quantize(values)
print(f"Original:  {values}")
print(f"Quantised: {quantised}")
# [0.12109375, 0.5, 1.0, -0.33203125]

error = np.abs(values - quantised)
print(f"Quantisation error: max={error.max():.6f}, mean={error.mean():.6f}")
```

## 4. Train Through Mismatch

The key technique: apply mismatch in the forward pass, but update
clean weights in the backward pass. The network learns weight values
that are robust to perturbation:

```python
# Training loop with mismatch injection
for epoch in range(100):
    for x_batch, y_batch in train_loader:
        # Forward with noisy weights (simulates hardware)
        noisy_weights = twin.apply_to_network_weights(trained_weights)
        output = forward(x_batch, noisy_weights)
        loss = compute_loss(output, y_batch)

        # Backward updates CLEAN weights (not noisy)
        gradients = backward(loss)
        trained_weights = update(trained_weights, gradients)
```

After mismatch-aware training, deployment accuracy typically drops
<1% from simulation, compared to 5-15% without.

## 5. Full Pipeline

```python
# 1. Train normally first (get baseline accuracy)
# 2. Fine-tune with mismatch injection (5-10 more epochs)
# 3. Quantise to Q8.8
# 4. Apply mismatch one final time → verify accuracy
# 5. Deploy to FPGA with confidence
```

## Comparison

| Approach | Accuracy Drop on Hardware |
|----------|:------------------------:|
| No twin (deploy directly) | 5-15% |
| Post-training quantisation only | 3-8% |
| Digital twin (this tutorial) | **<1%** |
| Digital twin + QAT (Tutorial 77) | **<0.5%** |

## References

- Neftci et al. (2011). "Systematic study of the impact of device
  variability on spiking neural networks." Neural Computation 23(6).
- He et al. (2020). "Noise-Aware Training of DNN in FPGA." FCCM 2020.
