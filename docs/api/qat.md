# Quantization-Aware Training — STE for Hardware Deployment

Train SNNs through quantization using straight-through estimators (STE). The missing link between training and FPGA deployment: weights are quantized in the forward pass but maintain full precision in the backward pass.

## How STE Works

Standard quantization is non-differentiable (rounding has zero gradient almost everywhere). The straight-through estimator passes the gradient through quantization as if it weren't there:

- **Forward:** `W_q = round(W / scale) * scale` (quantized)
- **Backward:** `∂L/∂W = ∂L/∂W_q` (identity, as if no quantization)

This trains weights to be robust to their own quantization noise. At export time, weights are already at target precision.

## Components

- **`QuantizedSNNLayer`** — SNN layer with quantization-aware forward pass.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_inputs` | (required) | Input dimension |
| `n_neurons` | (required) | Output dimension |
| `weight_bits` | 8 | Target weight precision (2, 4, 8, 16) |
| `threshold` | 1.0 | LIF spike threshold |
| `tau_mem` | 20.0 | Membrane time constant |

- **`TernaryWeights`** — Ternary quantization: {-1, 0, +1}. 94% memory reduction. Weights with `|w| < threshold_ratio * mean(|w|)` become 0.
- **`quantize_aware_train_step`** — One QAT training step with STE gradient flow. Returns `{'output', 'loss'}`.
- **`_ste_quantize`** — Core quantization function. Supports symmetric and asymmetric modes.

## Usage

```python
from sc_neurocore.qat import QuantizedSNNLayer, quantize_aware_train_step, TernaryWeights
import numpy as np

# Create QAT layer
layer = QuantizedSNNLayer(n_inputs=784, n_neurons=128, weight_bits=8)

# Training loop with STE
for epoch in range(100):
    result = quantize_aware_train_step(layer, x_train, y_target, lr=0.01)
    print(f"Loss: {result['loss']:.4f}")

# Export hardware-ready weights (already quantized to 8-bit)
hw_weights = layer.export_weights()

# Ternary quantization for extreme compression
tw = TernaryWeights(threshold_ratio=0.7)
ternary = tw.quantize(layer.W)
print(f"Sparsity: {tw.sparsity(layer.W):.1%}")  # ~50-70% zeros
```

**References:** QP-SNN (ICLR 2025), SpikeFit (EurIPS 2025).

See [Tutorial 77: QAT](../tutorials/77_qat.md).

::: sc_neurocore.qat.quantize
    options:
      show_root_heading: true
      members:
        - QuantizedSNNLayer
        - TernaryWeights
        - quantize_aware_train_step
