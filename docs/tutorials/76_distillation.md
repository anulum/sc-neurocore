# Tutorial 76: Knowledge Distillation

Transfer knowledge from a large/slow teacher SNN to a small/fast student SNN.
Temporal-aware distillation matches per-timestep output distributions.
Self-distillation uses extended timesteps as an implicit teacher.

## Why Distillation for SNNs

SNN accuracy scales with timesteps T: more timesteps = more spikes = better
integration = higher accuracy. But hardware deployment needs small T for
latency. Distillation transfers the accuracy of T=32 into a T=4 model.

## Temporal Distillation Loss

```python
import numpy as np
from sc_neurocore.distillation import TemporalDistillationLoss

loss_fn = TemporalDistillationLoss(
    temperature=3.0,
    alpha=0.5,
    entropy_weight=0.1,
)

teacher_logits = np.random.randn(32, 10)  # T=32, 10 classes
student_logits = np.random.randn(4, 10)   # T=4, 10 classes
targets = np.zeros(10); targets[3] = 1.0

result = loss_fn.compute(student_logits, teacher_logits, targets)
print(f"Total: {result['total_loss']:.3f}, "
      f"Distill: {result['distill_loss']:.3f}, "
      f"Task: {result['task_loss']:.3f}")
```

## Self-Distillation

No separate teacher model needed. Run the same model at more timesteps
to generate soft targets:

```python
from sc_neurocore.distillation import SelfDistiller

distiller = SelfDistiller(T_teacher=32, T_student=8, temperature=3.0)

def run_model(inputs, T):
    return np.random.randn(10)  # your SNN forward pass

soft_targets = distiller.generate_targets(run_model, inputs=np.zeros(784))
```

## API Reference

::: sc_neurocore.distillation.distill
    options:
      show_root_heading: true
