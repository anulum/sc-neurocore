# Tutorial 78: Residual Blocks for Deep SNNs

Build 400+ layer deep spiking networks with two residual architectures:
MS-ResNet (membrane shortcut) and SEW-ResNet (activation-before-addition).

## The Problem

Deep SNNs (>10 layers) suffer from vanishing spikes. Standard residual
connections fail because `spike(f(x) + x)` clips the identity mapping
through the binary activation. MS-ResNet adds the shortcut to membrane
potential instead, preserving gradient flow.

## Membrane Shortcut Block (MS-ResNet)

```python
import numpy as np
from sc_neurocore.residual import MembraneShortcutBlock

block = MembraneShortcutBlock(n_features=64, threshold=1.0, tau_mem=10.0)
x = (np.random.rand(64) > 0.5).astype(float)
spikes = block.forward(x)
```

## SEW Block

`spike(W@x) + x` instead of `spike(W@x + x)`:

```python
from sc_neurocore.residual import SEWBlock

block = SEWBlock(n_features=64, threshold=1.0)
spikes = block.forward(x)
```

## Deep SNN Stack

```python
from sc_neurocore.residual import DeepSNNStack

model = DeepSNNStack(n_features=64, n_blocks=20, block_type="ms")
print(f"Depth: {model.depth} layers")  # 40
output = model.forward(x)
```

| Block | Residual Path | Reference |
|-------|--------------|-----------|
| `MembraneShortcutBlock` | Input → membrane potential | Hu 2024 (TNNLS) |
| `SEWBlock` | spike(W@x) + x | Fang 2021 (NeurIPS) |

MS-ResNet: 482-layer SNN on CIFAR-10 — the deepest SNN published.

## API Reference

::: sc_neurocore.residual.blocks
    options:
      show_root_heading: true
