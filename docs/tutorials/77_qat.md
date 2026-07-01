# Tutorial 77: Quantisation-Aware Training

Standard SNN training uses float32/64 weights. Deploying to FPGA
requires fixed-point (Q8.8, Q4.4, or even ternary). Post-training
quantisation drops 3-8% accuracy because the model never learned to
operate with quantised weights.

Quantisation-Aware Training (QAT) simulates quantisation during the
forward pass using Straight-Through Estimators (STE) for gradients.
The model learns to compensate for quantisation noise — closing the
accuracy gap between training and deployment.

## The Problem

```
Train (float32) → Quantise (Q8.8) → Deploy (FPGA)
                  ↑ accuracy drops here
```

With QAT:
```
Train (float32 + simulated Q8.8) → Deploy (Q8.8 FPGA)
                                   ↑ no surprise accuracy drop
```

## Quantised SNN Layer

```python
import numpy as np
from sc_neurocore.qat import QuantizedSNNLayer, quantize_aware_train_step

layer = QuantizedSNNLayer(
    n_inputs=784,
    n_neurons=128,
    weight_bits=8,     # Q4.4 fixed-point during forward pass
    threshold=1.0,
    tau_mem=20.0,
)

# Forward pass quantises weights to 8-bit, but gradients flow through
# via STE (straight-through estimator)
x = np.random.randn(784).astype(np.float32)
target = np.zeros(128, dtype=np.float32)
target[42] = 1.0

result = quantize_aware_train_step(layer, x, target, lr=0.01)
print(f"Loss: {result['loss']:.4f}")
print(f"Gradient norm: {result['grad_norm']:.4f}")

# Export weights — already at target precision
hw_weights = layer.export_weights()
print(f"Weight range: [{hw_weights.min():.4f}, {hw_weights.max():.4f}]")
print(f"Unique values: {len(np.unique(hw_weights))}")  # 256 for 8-bit
```

### How STE Works

During the forward pass, weights are quantised:
```
w_q = round(w * 2^fraction_bits) / 2^fraction_bits
```

During the backward pass, the quantisation step is ignored — gradients
pass through as if quantisation didn't happen:
```
dL/dw = dL/dw_q   (straight-through)
```

This works because the quantisation error is small relative to the
gradient signal. The model learns weight values that are close to
quantisation grid points.

## Precision Levels

| Bits | Format | Unique Values | Memory vs Float32 | Accuracy Gap |
|------|--------|--------------|-------------------|--------------|
| 16 | Q8.8 | 65,536 | 2× reduction | <0.1% |
| 8 | Q4.4 | 256 | 4× reduction | 0.5–1.0% |
| 4 | Q2.2 | 16 | 8× reduction | 1–3% |
| 2 | Ternary | 3 ({-1,0,+1}) | 16× reduction | 3–5% |
| 1 | Binary | 2 ({-1,+1}) | 32× reduction | 5–8% |

Gap measured on MNIST SNN (784→128→10) with QAT vs without QAT.
Without QAT, post-training quantisation adds another 2-5% on top.

## Ternary Weights

Each weight is constrained to {-1, 0, +1}. No multipliers needed in
hardware — multiply becomes conditional negate or zero.

```python
from sc_neurocore.qat import TernaryWeights

ternary = TernaryWeights(threshold_ratio=0.7)

# Quantise trained weights
weights = np.random.randn(128, 784) * 0.1
t_weights = ternary.quantize(weights)

print(f"Sparsity: {ternary.sparsity(weights):.1%}")
print(f"Unique values: {np.unique(t_weights)}")  # [-1, 0, 1]
print(f"Memory: {weights.nbytes:,} → {t_weights.nbytes // 16:,} bytes (ternary packed)")
```

### Ternary FPGA Implementation

On FPGA, ternary weights require only 2 bits of storage and zero
multiplier LUTs:

```
if weight == +1: output += input
if weight == -1: output -= input
if weight ==  0: (skip)
```

A 784×128 layer with ternary weights uses ~1600 LUTs on iCE40 vs
~8000 LUTs with Q8.8 multipliers. That's a **5× resource reduction**.

## Training Loop with QAT

```python
from sc_neurocore.qat import QuantizedSNNLayer, quantize_aware_train_step

# Build a multi-layer QAT network
layers = [
    QuantizedSNNLayer(784, 256, weight_bits=8),
    QuantizedSNNLayer(256, 128, weight_bits=8),
    QuantizedSNNLayer(128, 10, weight_bits=8),
]

# Train for multiple epochs
for epoch in range(10):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        for layer in layers:
            result = quantize_aware_train_step(layer, x_batch, y_batch, lr=0.001)
            total_loss += result["loss"]
    print(f"Epoch {epoch}: loss={total_loss:.4f}")

# Export — no post-training quantisation needed
for i, layer in enumerate(layers):
    hw = layer.export_weights()
    print(f"Layer {i}: {hw.shape}, {len(np.unique(hw))} unique values")
```

## Learned Quantisers: LSQ, PACT, and Per-Channel Observers

STE with a fixed per-tensor scale leaves accuracy on the table at low bit
widths. Three PyTorch quantisers close most of the remaining gap and compose
with the surrogate-gradient SNN modules.

### LSQ — learned step size

`LSQLinear` learns the quantiser step size jointly with the weights instead of
deriving it from the running weight range (Esser et al. 2020). Per-output-neuron
(per-channel) steps are the default.

```python
import torch
from sc_neurocore.qat import LSQLinear

layer = LSQLinear(784, 128, n_bits=4, per_channel=True)
out = layer(torch.randn(32, 784))
out.sum().backward()                 # gradients reach both weights and the step
export = layer.export_quantized()    # int32 codes + per-channel step for FPGA
print(export["weight_int"].shape, export["step"].shape)  # (128, 784) (128,)
```

### PACT — parameterised activation clipping

`PACTActivation` learns the activation clipping bound `alpha` (Choi et al. 2018),
so low-bit activation quantisation no longer needs a hand-tuned range. Useful for
the continuous input current feeding the first layer.

```python
from sc_neurocore.qat import PACTActivation

act = PACTActivation(n_bits=4, alpha_init=6.0)
y = act(torch.randn(32, 784) * 3)    # clipped to [0, alpha] then quantised
codes, scale = act.quantize(torch.randn(32, 784))
```

### Per-channel observers

Observers derive quantiser scales from calibration statistics. A per-channel
observer gives each output neuron its own scale, which recovers the accuracy a
single per-tensor scale loses when channels have very different magnitudes.

```python
from sc_neurocore.qat import PerChannelMinMaxObserver

obs = PerChannelMinMaxObserver(n_bits=4, ch_axis=0, symmetric=True)
for batch in calibration_weights:
    obs.observe(batch)
scale, zero_point = obs.calculate_qparams()   # one scale per output channel
w_fake_quant = obs.quantize(weight)
```

### End-to-end: `LSQPACTLIFNet`

`LSQPACTLIFNet` wires LSQ per-channel weight quantisation and a PACT-quantised
analogue input into a feedforward LIF SNN:

```python
from sc_neurocore.qat import LSQPACTLIFNet

net = LSQPACTLIFNet(784, 128, 10, weight_bits=4, act_bits=4)
spikes, mem = net(torch.randn(25, 32, 784))   # (T, batch, features)
export = net.export_quantized()                # per-layer LSQ codes + input scale
```

## Integration with Studio

In the Visual SNN Studio:

1. Train your network in the Training Monitor (float32)
2. Switch to the FPGA tab
3. Select target bit-width (8, 4, 2, or 1)
4. The Studio applies QAT-style quantisation and shows the accuracy
   impact before synthesis
5. Click Synthesise to see resource usage with quantised weights

## Comparison

| Feature | SC-NeuroCore QAT | snnTorch | Norse | Brevitas |
|---------|:----------------:|:--------:|:-----:|:--------:|
| STE training | Yes | No | No | Yes (ANN) |
| Ternary weights | Yes | No | No | Yes (ANN) |
| Binary weights | Yes | No | No | Yes (ANN) |
| LSQ (learned step) | Yes | No | No | Yes (ANN) |
| PACT (learned clip) | Yes | No | No | Yes (ANN) |
| Per-channel observers | Yes | No | No | Yes (ANN) |
| FPGA-aware | Yes | No | No | Partial |
| SNN-specific | Yes | — | — | No (ANN only) |

SC-NeuroCore is the only framework providing quantisation-aware training
specifically for spiking neural networks with direct FPGA deployment.

## References

- Hubara et al. (2016). "Binarized Neural Networks." NeurIPS 2016.
- Li & Liu (2016). "Ternary Weight Networks." arXiv:1605.04711.
- Jacob et al. (2018). "Quantization and Training of Neural Networks
  for Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
- Deng et al. (2021). "Comprehensive SNN Compressed Accelerator on
  FPGA." IEEE TCAS-I 68(7):2889-2901.
- Esser et al. (2020). "Learned Step Size Quantization." ICLR 2020.
- Choi et al. (2018). "PACT: Parameterized Clipping Activation for
  Quantized Neural Networks." arXiv:1805.06085.
