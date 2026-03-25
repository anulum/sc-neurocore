# Residual Blocks

SNN residual blocks for building 400+ layer deep spiking networks.

- `MembraneShortcutBlock` — MS-ResNet: residual adds to membrane potential, not spikes. Block dynamical isometry. (Hu 2024, TNNLS — 482-layer SNN on CIFAR-10)
- `SEWBlock` — Activation-before-addition: `spike(W@x) + x`. (Fang 2021, NeurIPS)
- `DeepSNNStack` — Stack of residual blocks. `block_type='ms'` or `'sew'`. `.depth` reports effective weight layers.

```python
from sc_neurocore.residual import MembraneShortcutBlock, SEWBlock, DeepSNNStack
```

See [Tutorial 78: Residual Blocks](../tutorials/78_residual.md) for usage examples.

::: sc_neurocore.residual.blocks
    options:
      show_root_heading: true
      members:
        - MembraneShortcutBlock
        - SEWBlock
        - DeepSNNStack
