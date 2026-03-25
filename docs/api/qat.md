# Quantization-Aware Training

Train SNNs through quantization using straight-through estimators (STE).

- `QuantizedSNNLayer` — SNN layer with quantized forward pass (STE). Weights quantized in forward, full-precision in backward. Export hardware-ready weights directly.
- `TernaryWeights` — {-1, 0, +1} quantization. 94% memory reduction. Threshold-based zeroing.
- `quantize_aware_train_step` — One QAT training step with STE gradient flow.

Supported precisions: 2-bit (ternary), 4-bit, 8-bit, 16-bit.

```python
from sc_neurocore.qat import QuantizedSNNLayer, TernaryWeights, quantize_aware_train_step
```

See [Tutorial 77: QAT](../tutorials/77_qat.md) for usage examples.

::: sc_neurocore.qat.quantize
    options:
      show_root_heading: true
      members:
        - QuantizedSNNLayer
        - TernaryWeights
        - quantize_aware_train_step
