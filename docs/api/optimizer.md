# Resource Optimizer

Compress an SNN to fit a target FPGA (LUT/BRAM/DSP constraints).

```python
from sc_neurocore.optimizer import ResourceOptimizer

opt = ResourceOptimizer(target_luts=10000, target_bram=36)
compressed = opt.optimize(model)
```

::: sc_neurocore.optimizer
    options:
      show_root_heading: true
