# SNN Optimizer

LLVM-style optimization passes for SNN computation graphs: dead neuron elimination, weight pruning, layer fusion, redundant connection removal.

```python
from sc_neurocore.snn_optimizer import SNNOptimizer

opt = SNNOptimizer()
optimized = opt.optimize(model, passes=["prune", "fuse", "eliminate_dead"])
```

See [Tutorial 66: SNN Optimizer](../tutorials/66_snn_optimizer.md).

::: sc_neurocore.snn_optimizer
    options:
      show_root_heading: true
