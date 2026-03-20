# Tutorial 26: Zero-Multiplication Predictive Coding (Conjecture C9)

SC-NeuroCore implements predictive coding with ZERO multiplications:

1. **Prediction error = XOR(predicted, actual)** — one gate per bit
2. **Error magnitude = popcount(XOR result) / L** — Hamming distance
3. **Precision update = STDP** — push weights toward actual input

This maps entirely to XOR gates + a popcount tree on FPGA. No DSP blocks.

## Basic Usage

```python
from sc_neurocore.layers.predictive_coding import PredictiveCodingSCLayer

layer = PredictiveCodingSCLayer(n_inputs=4, n_neurons=2, length=512, lr=0.1, seed=42)

# Train on a repeated pattern
for epoch in range(20):
    result = layer.forward([0.3, 0.7, 0.5, 0.9])
    print(f"Epoch {epoch}: error = {result['prediction_error']:.4f}")
# Error decreases as predictions improve
```

## Novelty Detection

```python
# After learning one pattern, present a novel input
layer = PredictiveCodingSCLayer(n_inputs=3, n_neurons=2, length=512, lr=0.2, seed=42)

for _ in range(30):
    layer.forward([0.8, 0.2, 0.5])

familiar = layer.forward([0.8, 0.2, 0.5])["prediction_error"]
novel = layer.forward([0.2, 0.8, 0.5])["prediction_error"]
print(f"Familiar: {familiar:.3f}, Novel: {novel:.3f}")
# Novel input produces higher prediction error (surprise)
```

## Why This Matters

Traditional predictive coding requires subtraction (analog) or multiplication
(digital). In SC, XOR IS subtraction — the Hamming distance between two
Bernoulli(p) streams approximates |p1 - p2|. This is the most hardware-efficient
predictive coding architecture possible.
