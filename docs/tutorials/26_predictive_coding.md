<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 26: Zero-Multiplication Predictive Coding (Conjecture C9)

SC-NeuroCore implements predictive coding with ZERO multiplications:

1. **Prediction error = XOR(predicted, actual)** — one gate per bit
2. **Error magnitude = popcount(XOR result) / L** — Hamming distance
3. **Precision update = STDP** — push weights toward actual input

This maps entirely to XOR gates + a popcount tree on FPGA. No DSP blocks needed.

## What is Predictive Coding?

Predictive coding is a theory of neural computation where each layer predicts
the activity of the layer below. The difference between prediction and reality
(the prediction error) is all that propagates upward. When predictions are
accurate, error signals are small — the brain only processes what's surprising.

In conventional digital hardware, computing prediction error requires subtraction
and multiplication. SC-NeuroCore eliminates both by exploiting stochastic
bitstream properties.

## The SC Insight

Two stochastic bitstreams $a$ and $b$ encoding probabilities $p_a$ and $p_b$:

- $\text{XOR}(a, b)$ produces a bitstream encoding $|p_a - p_b|$ (approximately)
- $\text{popcount}(\text{XOR}(a, b)) / L$ gives the Hamming distance $\approx |p_a - p_b|$

XOR IS subtraction in the stochastic domain. One gate per bit, no carry chain,
no DSP block, no multiplier.

## 1. Basic Usage

```python
from sc_neurocore.layers.predictive_coding import PredictiveCodingSCLayer

layer = PredictiveCodingSCLayer(
    n_inputs=4,
    n_neurons=2,
    length=512,
    lr=0.1,
    seed=42,
)

# Train on a repeated pattern
errors = []
for epoch in range(30):
    result = layer.forward([0.3, 0.7, 0.5, 0.9])
    errors.append(result['prediction_error'])
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}: error = {result['prediction_error']:.4f}")

# Error decreases as the layer learns to predict the input
print(f"Initial error: {errors[0]:.4f}")
print(f"Final error:   {errors[-1]:.4f}")
print(f"Reduction:     {(1 - errors[-1]/errors[0])*100:.0f}%")
```

## 2. Novelty Detection

After learning a pattern, novel inputs produce high prediction error:

```python
layer = PredictiveCodingSCLayer(n_inputs=3, n_neurons=2, length=512, lr=0.2, seed=42)

# Learn a familiar pattern
for _ in range(50):
    layer.forward([0.8, 0.2, 0.5])

# Test familiar vs novel
familiar = layer.forward([0.8, 0.2, 0.5])["prediction_error"]
slightly_novel = layer.forward([0.7, 0.3, 0.5])["prediction_error"]
very_novel = layer.forward([0.2, 0.8, 0.5])["prediction_error"]

print(f"Familiar:       {familiar:.4f}")
print(f"Slightly novel: {slightly_novel:.4f}")
print(f"Very novel:     {very_novel:.4f}")
# Prediction error scales with input novelty
```

This is the computational basis for surprise detection — the same mechanism
used by ArcaneNeuron's predictor compartment (Tutorial 34).

## 3. Multi-Pattern Learning

```python
import numpy as np

layer = PredictiveCodingSCLayer(n_inputs=4, n_neurons=4, length=1024, lr=0.05, seed=42)

# Two alternating patterns
pattern_a = [0.9, 0.1, 0.8, 0.2]
pattern_b = [0.1, 0.9, 0.2, 0.8]

for epoch in range(100):
    pattern = pattern_a if epoch % 2 == 0 else pattern_b
    result = layer.forward(pattern)

# After training, both patterns should have low error
err_a = layer.forward(pattern_a)["prediction_error"]
err_b = layer.forward(pattern_b)["prediction_error"]
err_new = layer.forward([0.5, 0.5, 0.5, 0.5])["prediction_error"]

print(f"Pattern A error: {err_a:.4f}")
print(f"Pattern B error: {err_b:.4f}")
print(f"Novel pattern:   {err_new:.4f}")  # higher than A or B
```

## 4. FPGA Resource Comparison

| Operation | Conventional | SC Predictive Coding |
|-----------|-------------|---------------------|
| Subtraction | 16-bit ripple subtract (16 LUTs) | XOR gate (1 LUT per bit) |
| Multiplication | Array multiplier (256 LUTs) | Not needed |
| Error magnitude | Absolute value circuit | Popcount tree (log₂(L) LUTs) |
| Weight update | MAC unit + accumulator | STDP counter |
| **Total per synapse** | **~300 LUTs** | **~10 LUTs** |

For a 100-input, 10-output predictive coding layer:
- Conventional: ~300K LUTs (won't fit on most FPGAs)
- SC: ~10K LUTs (fits easily on iCE40 HX8K)

## 5. How the Learning Rule Works

The STDP-based update rule adjusts predictions toward actual inputs:

1. Compute prediction error: `error = XOR(predicted_bits, actual_bits)`
2. For each input-output pair with high error: increase weight (LTP)
3. For each input-output pair with low error: decrease weight slightly (LTD)
4. The asymmetry drives weights toward values that minimize prediction error

This is biologically plausible — real cortical circuits adjust synaptic
strength based on prediction error signals from higher areas.

## 6. Hierarchical Predictive Coding

Stack multiple layers for hierarchical prediction:

```python
# Layer 1: predicts raw input
l1 = PredictiveCodingSCLayer(n_inputs=8, n_neurons=4, length=512, lr=0.1, seed=1)

# Layer 2: predicts L1's residual errors
l2 = PredictiveCodingSCLayer(n_inputs=4, n_neurons=2, length=512, lr=0.05, seed=2)

inputs = [0.3, 0.7, 0.5, 0.9, 0.2, 0.8, 0.4, 0.6]

for epoch in range(50):
    r1 = l1.forward(inputs)
    # Feed L1's prediction error as L2's input
    r2 = l2.forward(list(r1['neuron_errors'][:4]))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: L1 error={r1['prediction_error']:.4f}, "
              f"L2 error={r2['prediction_error']:.4f}")
```

Each layer compresses the residual from the layer below. This is analogous
to hierarchical processing in visual cortex: V1 predicts retinal input,
V2 predicts V1 residuals, V4 predicts V2 residuals.

## Further Reading

- [Tutorial 34: ArcaneNeuron](34_arcane_neuron.md) — uses predictive coding for self-modeling
- [Tutorial 27: Fault Tolerance](27_fault_tolerance.md) — SC's inherent noise robustness
- [API: Layers](../api/layers.md) — PredictiveCodingSCLayer API docs
