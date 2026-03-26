# Knowledge Distillation — Temporal Spike Transfer

SNN-to-SNN and ANN-to-SNN knowledge transfer with temporal spike alignment. No SNN library ships distillation as a reusable module.

## TemporalDistillationLoss

Matches per-timestep output distributions from teacher to student, with entropy regularization to prevent learning erroneous knowledge.

`L_total = α * L_distill + (1-α) * L_task + L_entropy`

Where:

- `L_distill = T² * KL(teacher_soft || student_soft)` — temperature-softened KL divergence
- `L_task = CrossEntropy(student, targets)` — standard task loss (optional)
- `L_entropy = -β * H(student_soft)` — entropy regularization (prevents collapse)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `temperature` | 3.0 | Softmax temperature for logit matching |
| `alpha` | 0.5 | Balance: 0=task only, 1=distill only |
| `entropy_weight` | 0.1 | Entropy regularization strength |

Returns dict: `{'total_loss', 'distill_loss', 'task_loss', 'entropy_loss'}`.

## SelfDistiller — Implicit Teacher

Uses the same model at extended timesteps as implicit teacher. Run model at T_teacher steps (more accurate, slower) to generate soft targets for training at T_student steps (faster, less accurate).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `T_teacher` | 32 | Timesteps for teacher pass |
| `T_student` | 8 | Timesteps for student pass |
| `temperature` | 3.0 | Softmax temperature |

## Usage

```python
from sc_neurocore.distillation import TemporalDistillationLoss, SelfDistiller
import numpy as np

# Teacher-student distillation
loss_fn = TemporalDistillationLoss(temperature=3.0, alpha=0.7)
result = loss_fn.compute(
    student_logits=student_output,
    teacher_logits=teacher_output,
    targets=one_hot_labels,  # optional
)
print(f"Total: {result['total_loss']:.4f}, Distill: {result['distill_loss']:.4f}")

# Self-distillation
sd = SelfDistiller(T_teacher=32, T_student=8)
soft_targets = sd.generate_targets(run_fn=model.forward, inputs=x)
```

**Reference:** CVPR 2025 — temporal separation + entropy regularization.

See [Tutorial 76: Knowledge Distillation](../tutorials/76_distillation.md).

::: sc_neurocore.distillation.distill
    options:
      show_root_heading: true
      members:
        - TemporalDistillationLoss
        - SelfDistiller
