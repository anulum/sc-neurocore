# Knowledge Distillation

SNN-to-SNN and ANN-to-SNN knowledge transfer with temporal spike alignment.

- `TemporalDistillationLoss` — Per-timestep KL divergence with entropy regularization. Matches softened output distributions from teacher to student. (CVPR 2025)
- `SelfDistiller` — Use the same model at extended T timesteps as implicit teacher. No separate teacher model needed.

```python
from sc_neurocore.distillation import TemporalDistillationLoss, SelfDistiller
```

See [Tutorial 76: Knowledge Distillation](../tutorials/76_distillation.md) for usage examples.

::: sc_neurocore.distillation.distill
    options:
      show_root_heading: true
      members:
        - TemporalDistillationLoss
        - SelfDistiller
