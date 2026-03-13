<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Stochastic Computing for ML Engineers

A guide for deep learning practitioners familiar with PyTorch/JAX who
want to understand how SC-NeuroCore relates to conventional neural
networks and where stochastic computing offers unique advantages.

## SC vs conventional deep learning

| Aspect | Conventional DNN | SC-NeuroCore |
|--------|-----------------|-------------|
| Representation | float32/float16 tensors | Bitstream probabilities ∈ [0, 1] |
| Multiply | FMA instruction | AND gate |
| Activation | ReLU, sigmoid, etc. | LIF spike (threshold) |
| Training | Backprop (autograd) | Surrogate gradient or rate-level pseudo-gradient |
| Inference | GPU batch | CPU/FPGA bitstream |
| Power (inference) | 50-300 W (GPU) | 0.01-1 W (FPGA) |
| Precision | 32/16/8 bit | ~log₂(L) effective bits |

## Mapping DNN concepts to SC

### Linear layer → VectorizedSCLayer

```python
# PyTorch
layer = torch.nn.Linear(784, 128)
output = layer(input)  # matrix multiply + bias

# SC-NeuroCore
from sc_neurocore import VectorizedSCLayer
layer = VectorizedSCLayer(n_inputs=784, n_neurons=128, length=256)
output = layer.forward(input)  # bitwise AND + MUX + LIF
```

Weights in SC are probabilities [0, 1] — analogous to sigmoid-constrained
weights. There is no bias term; the threshold of the LIF neuron serves
a similar role.

### Activation → LIF neuron

The LIF neuron is the SC equivalent of an activation function. It
integrates input over the bitstream length and fires when a threshold
is crossed. The output firing rate is a non-linear function of the
input — similar to a soft threshold or sigmoid.

```
Input probability p → LIF → Output firing rate f(p)

For typical parameters:
  f(p) ≈ 0           for p < 0.1
  f(p) ≈ 2.5·(p-0.1) for 0.1 < p < 0.5
  f(p) ≈ 1           for p > 0.5
```

This is roughly a clipped-ReLU. The exact shape depends on leak,
gain, and threshold parameters.

### Dropout → SC noise

SC has built-in stochasticity. A bitstream with probability p will
randomly produce 0s and 1s — this acts like multiplicative noise
(similar to dropout). Shorter bitstreams (smaller L) increase the
noise variance, equivalent to stronger regularisation.

```python
# Effective dropout rate from SC noise
# Standard deviation of bitstream estimate = sqrt(p(1-p)/L)
# For p=0.5, L=256: std = 0.031 ≈ 3% noise
# For p=0.5, L=64:  std = 0.063 ≈ 6% noise
```

### BatchNorm → not needed

SC values are inherently bounded to [0, 1] and the stochastic encoding
provides normalisation. Internal covariate shift is less of an issue
because the bitstream representation is already normalised.

### Attention → StochasticAttention

```python
# PyTorch
attn = torch.nn.MultiheadAttention(embed_dim=64, num_heads=4)
output, weights = attn(Q, K, V)

# SC-NeuroCore
from sc_neurocore import StochasticAttention
attn = StochasticAttention(dim_k=64, temperature=1.0)
output = attn.forward_softmax(Q, K, V)  # proper softmax
# or
output = attn.forward(Q, K, V)  # row-sum normalised (SC-native, cheaper)
```

## Training SC networks

### Approach 1: Rate-level pseudo-gradient (recommended)

Treat the network as a continuous function of its weights, ignoring
the stochastic bitstream during backpropagation. Compute gradients
at the rate (probability) level, update weights in float, then
re-encode weights as bitstreams.

```python
# Forward: SC bitstream inference (stochastic)
h1 = layer1.forward(x)
h1 = np.clip(h1, 0.01, 0.99)
scores = layer2.forward(h1)

# Backward: float gradient (deterministic)
grad_out = cross_entropy_grad(scores, label)
dW2 = np.outer(grad_out, h1)
layer2.weights -= lr * dW2
layer2.weights = np.clip(layer2.weights, 0.01, 0.99)
layer2._refresh_packed_weights()  # re-encode bitstreams
```

This is similar to straight-through estimator (STE) used in
quantisation-aware training.

### Approach 2: Surrogate gradient (snnTorch-style)

Use a differentiable surrogate for the LIF spike function:

```python
# SC-NeuroCore provides surrogate gradient utilities
from sc_neurocore.learning import SurrogateGradientTrainer

trainer = SurrogateGradientTrainer(
    network=[layer1, layer2],
    surrogate="fast_sigmoid",  # or "triangular", "arctangent"
    lr=0.001,
)

for epoch in range(10):
    for x, y in dataloader:
        loss = trainer.train_step(x, y)
```

See Tutorial 03 for the full surrogate gradient flow.

### Approach 3: STDP (unsupervised)

For feature extraction without labels:

```python
from sc_neurocore import StochasticSTDPSynapse
# STDP operates on individual bitstream steps
# No gradient computation needed — fully local
```

See Tutorial 08 for STDP details.

## Performance comparison

### MNIST (50 PCA features, 128 hidden, 10 output)

| Method | Test accuracy | Training time | Inference power |
|--------|-------------|---------------|----------------|
| PyTorch MLP (float32, GPU) | 97.5% | 5 s | ~100 W |
| SC-NeuroCore (L=512, CPU) | ~88% | ~60 s | ~50 W |
| SC-NeuroCore (L=512, Rust) | ~88% | ~6 s | ~15 W |
| SC-NeuroCore (L=256, FPGA) | ~85% | — (fixed weights) | ~0.05 W |

SC trades ~10% accuracy for 2000× lower inference power. For
always-on edge applications (keyword spotting, gesture recognition,
anomaly detection), this trade-off is often worthwhile.

### Where SC wins

1. **Ultra-low-power inference**: 50 mW vs 100 W (FPGA vs GPU)
2. **Fault tolerance**: Random bit flips in SC cause graceful degradation,
   not catastrophic failure. A stuck-at-0 fault in one AND gate
   slightly reduces one weight — the network still works.
3. **Hardware cost**: SC multiplier = 1 gate. An edge device can
   fit thousands of SC neurons in the LUT budget of a single
   conventional multiplier.
4. **Noise robustness**: SC inherently operates on noisy signals.
   Sensor noise is just more randomness in the bitstream.

### Where SC loses

1. **Precision**: ~8 effective bits at L=256 vs 16-32 bits float
2. **Training speed**: Bitstream simulation is slow in software
3. **Correlation sensitivity**: Reusing random numbers breaks SC
   arithmetic (must use independent LFSRs)
4. **Sequential nature**: L cycles per operation (mitigated by
   massive parallelism on FPGA)

## Converting a PyTorch model

```python
import torch
import numpy as np
from sc_neurocore import VectorizedSCLayer

# Trained PyTorch model
pytorch_model = torch.nn.Sequential(
    torch.nn.Linear(50, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)
# ... training ...

# Extract weights, normalise to [0, 1]
w1 = pytorch_model[0].weight.detach().numpy()
w2 = pytorch_model[2].weight.detach().numpy()

def normalise_weights(w):
    """Map float weights to SC probability range [0.01, 0.99]."""
    w_min, w_max = w.min(), w.max()
    return 0.01 + 0.98 * (w - w_min) / (w_max - w_min)

# Create SC layers with converted weights
sc_layer1 = VectorizedSCLayer(n_inputs=50, n_neurons=128, length=512)
sc_layer1.weights = normalise_weights(w1)
sc_layer1._refresh_packed_weights()

sc_layer2 = VectorizedSCLayer(n_inputs=128, n_neurons=10, length=512)
sc_layer2.weights = normalise_weights(w2)
sc_layer2._refresh_packed_weights()

# The SC network will have slightly lower accuracy due to quantisation
# but can deploy on FPGA at milliwatt power
```

## SC-native architectures

Instead of converting from conventional DNNs, design architectures
that exploit SC properties:

1. **Wide-and-shallow**: SC noise favours wider layers (more averaging)
   over deeper networks (noise compounds)
2. **Ensemble averaging**: Run the same network multiple times with
   different LFSR seeds and average the outputs — reduces noise by √N
3. **Progressive precision**: Start with L=64 for early layers (feature
   detection is noise-tolerant), increase to L=512 for final layers
   (classification needs precision)
4. **Reservoir + readout**: Fixed random SC reservoir (no training noise)
   + linear readout (no SC noise in training)

## Further reading

- Tutorial 03: Surrogate Gradient Training
- Tutorial 07: MNIST Classification
- Tutorial 10: Reservoir Computing
- Guide: Performance Tuning
- Research: Foundational Whitepaper
