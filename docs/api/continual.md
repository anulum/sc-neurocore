# Continual Learning

Train, deploy, adapt without catastrophic forgetting.

- `EWCRegularizer` — Elastic Weight Consolidation: penalize changes to weights important for previous tasks. Fisher information matrix tracks parameter importance.
- `PackNetPruner` — Progressive pruning: each task gets a dedicated subset of weights, frozen after learning.

SNNs have a natural advantage for continual learning: sparse binary activations mean less catastrophic interference than dense float activations.

```python
from sc_neurocore.continual import EWCRegularizer, PackNetPruner
```

See [Tutorial 58: Continual Learning](../tutorials/58_continual_learning.md).

::: sc_neurocore.continual
    options:
      show_root_heading: true
