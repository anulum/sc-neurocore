<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 37: JAX JIT Training

SC-NeuroCore supports JAX for JIT-compiled, GPU-accelerated SNN training.
JAX's functional transformation model (grad, jit, vmap) enables efficient
surrogate gradient computation on stochastic computing layers.

## Prerequisites

```bash
pip install sc-neurocore[jax]
# or: pip install jax jaxlib
```

## 1. JAX Dense Layer

The `JaxSCDenseLayer` provides a JAX-compatible SC dense layer:

```python
import jax
import jax.numpy as jnp
from sc_neurocore.layers.jax_dense_layer import JaxSCDenseLayer

layer = JaxSCDenseLayer(n_inputs=8, n_neurons=4, length=256, seed=42)

# Forward pass — JIT-compiled
inputs = jnp.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.1, 0.6, 0.4])
output = layer.forward(inputs)
print(f"Output: {output}")
```

## 2. Surrogate Gradient Training

JAX's `grad` computes gradients through the non-differentiable spike function
using a straight-through estimator:

```python
import jax
import jax.numpy as jnp
from sc_neurocore.accel.jax_backend import to_jax

def lif_step(v, current, threshold=1.0, tau=20.0, dt=1.0):
    """Single LIF step with straight-through surrogate gradient."""
    dv = (-v + current) * dt / tau
    v_new = v + dv
    # Spike: hard threshold forward, surrogate backward
    spike = (v_new >= threshold).astype(jnp.float32)
    # Straight-through: gradient of spike ≈ gradient of sigmoid
    spike_surrogate = jax.nn.sigmoid(10.0 * (v_new - threshold))
    spike = spike - jax.lax.stop_gradient(spike - spike_surrogate)
    v_reset = v_new * (1.0 - spike)
    return v_reset, spike

# Differentiable through the spike function
grad_fn = jax.grad(lambda v, I: lif_step(v, I)[1].sum())
gradient = grad_fn(jnp.float32(0.8), jnp.float32(1.5))
print(f"Gradient of spikes w.r.t. voltage: {gradient:.4f}")
```

## 3. Training Loop on Synthetic Data

```python
import jax
import jax.numpy as jnp
import numpy as np

# Parameters
n_inputs, n_hidden, n_outputs = 10, 32, 3
n_steps = 50
lr = 0.01

# Initialize weights
key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)
w1 = jax.random.normal(k1, (n_inputs, n_hidden)) * 0.1
w2 = jax.random.normal(k2, (n_hidden, n_outputs)) * 0.1

def forward(w1, w2, x_seq):
    """Forward pass: n_steps of LIF dynamics."""
    v1 = jnp.zeros(n_hidden)
    v2 = jnp.zeros(n_outputs)
    out_spikes = jnp.zeros(n_outputs)

    for t in range(n_steps):
        # Hidden layer
        I1 = x_seq[t] @ w1
        dv1 = (-v1 + I1) / 20.0
        v1 = v1 + dv1
        s1 = jax.nn.sigmoid(10.0 * (v1 - 1.0))
        v1 = v1 * (1.0 - s1)

        # Output layer
        I2 = s1 @ w2
        dv2 = (-v2 + I2) / 20.0
        v2 = v2 + dv2
        s2 = jax.nn.sigmoid(10.0 * (v2 - 1.0))
        v2 = v2 * (1.0 - s2)
        out_spikes = out_spikes + s2

    return out_spikes / n_steps

def loss_fn(w1, w2, x_seq, target):
    pred = forward(w1, w2, x_seq)
    return jnp.mean((pred - target) ** 2)

# JIT-compile the gradient function
grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))

# Training
for epoch in range(100):
    # Random input sequence and target
    x_seq = jax.random.uniform(jax.random.PRNGKey(epoch), (n_steps, n_inputs))
    target = jnp.array([1.0, 0.0, 0.0])

    g1, g2 = grad_fn(w1, w2, x_seq, target)
    w1 = w1 - lr * g1
    w2 = w2 - lr * g2

    if epoch % 20 == 0:
        l = loss_fn(w1, w2, x_seq, target)
        print(f"Epoch {epoch}: loss = {l:.4f}")
```

## 4. GPU Acceleration

JAX auto-detects GPU. Training runs on CUDA/ROCm without code changes:

```python
# Check if GPU is available
print(jax.devices())  # [GpuDevice(id=0)] or [CpuDevice(id=0)]

# vmap for batch training
batched_loss = jax.vmap(loss_fn, in_axes=(None, None, 0, 0))
# Process entire batch in parallel on GPU
```

## 5. Export Trained Weights to SC

After training, convert JAX weights to SC-NeuroCore format:

```python
import numpy as np
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer

# Convert JAX arrays to NumPy
w1_np = np.array(w1)

# Normalize weights to [0, 1] for SC bitstream encoding
w_min, w_max = w1_np.min(), w1_np.max()
w_normalized = (w1_np - w_min) / (w_max - w_min)

# Create SC layer with trained weights
sc_layer = SCDenseLayer(n_inputs=n_inputs, n_neurons=n_hidden, length=512)
sc_layer.set_weights(w_normalized)

# Now run with stochastic bitstreams
output = sc_layer.forward([0.5] * n_inputs)
print(f"SC output: {output}")
```

## 6. Holonomic Adapters with JAX

The SCPN holonomic adapter ecosystem uses JAX for differentiable layer transforms:

```python
from sc_neurocore.adapters.holonomic.l1_quantum import L1QuantumAdapter

adapter = L1QuantumAdapter()
# adapt() uses JAX JIT under the hood
state = adapter.adapt(initial_state)
```

## JAX vs PyTorch vs NumPy

| Feature | JAX | PyTorch | NumPy |
|---------|-----|---------|-------|
| Autodiff | `jax.grad` (functional) | `loss.backward()` (imperative) | Manual |
| JIT | `jax.jit` (trace-based) | `torch.compile` | N/A |
| GPU | Automatic | `.to("cuda")` | N/A (use CuPy) |
| Batch | `jax.vmap` | `DataLoader` | Manual |
| SC integration | `JaxSCDenseLayer` | `SpikingNet` → `to_sc_weights()` | `VectorizedSCLayer` |
| Best for | Research, custom surrogates | Production training | Inference, co-sim |

## Runnable Demo

```bash
PYTHONPATH=src python examples/jax_training_demo.py
```

## Further Reading

- [Tutorial 03: Surrogate Gradient Training](03_surrogate_gradient_training.md) — PyTorch path
- [Tutorial 20: Adapter Ecosystem](20_adapter_ecosystem.md) — JAX holonomic adapters
- [API: Acceleration](../api/accel.md) — JAX/CuPy/MPI backend API
