# Tutorial 77: Quantization-Aware Training

Train SNNs through quantization using straight-through estimators (STE).
Closes the gap between full-precision training and fixed-point hardware.
Includes ternary weight quantization (94% memory reduction).

## The Problem

SNNs trained in float64 lose accuracy when deployed to fixed-point hardware.
Post-training quantization drops 3-8% accuracy. QAT simulates quantization
during training so the model learns to compensate.

## Quantized SNN Layer

```python
import numpy as np
from sc_neurocore.qat import QuantizedSNNLayer, quantize_aware_train_step

layer = QuantizedSNNLayer(
    n_inputs=784, n_neurons=128,
    weight_bits=8, threshold=1.0, tau_mem=20.0,
)

x = np.random.randn(784)
target = np.zeros(128); target[42] = 1.0
result = quantize_aware_train_step(layer, x, target, lr=0.01)
print(f"Loss: {result['loss']:.4f}")

hw_weights = layer.export_weights()  # already at 8-bit precision
```

## Ternary Weights

Each weight is {-1, 0, +1}. 94% memory reduction:

```python
from sc_neurocore.qat import TernaryWeights

ternary = TernaryWeights(threshold_ratio=0.7)
weights = np.random.randn(128, 784) * 0.1
t_weights = ternary.quantize(weights)
print(f"Sparsity: {ternary.sparsity(weights):.1%}")
```

| Bits | Memory vs Float32 | Use Case |
|------|-------------------|----------|
| 2 (ternary) | 16x reduction | Extreme edge |
| 4 | 8x reduction | FPGA LUT-based |
| 8 | 4x reduction | Standard ASIC/FPGA |

## API Reference

::: sc_neurocore.qat.quantize
    options:
      show_root_heading: true
