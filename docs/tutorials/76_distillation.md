# Tutorial 76: Knowledge Distillation for SNNs

Transfer knowledge from a large, slow teacher SNN to a small, fast
student SNN. SNN-specific distillation matches temporal spike patterns,
not just output logits — preserving the timing information that makes
SNNs useful.

## Why Distillation for SNNs

SNN accuracy scales with timesteps T: more timesteps = more spikes =
better temporal integration = higher accuracy. But hardware deployment
needs small T for low latency.

| Timesteps | Accuracy (MNIST) | Latency | Hardware Cost |
|-----------|-----------------|---------|---------------|
| T=32 | 97.2% | 3.2 ms | 32× compute |
| T=8 | 95.1% | 0.8 ms | 8× compute |
| T=4 | 92.8% | 0.4 ms | 4× compute |
| T=4 + distillation | **95.8%** | 0.4 ms | 4× compute |

Distillation recovers ~3% accuracy at the small T — getting T=32
quality at T=4 latency.

## Temporal Distillation Loss

Standard KD matches output distributions. Temporal distillation also
matches per-timestep spike rate trajectories — the teacher's temporal
dynamics guide the student:

```python
import numpy as np
from sc_neurocore.distillation import TemporalDistillationLoss

loss_fn = TemporalDistillationLoss(
    temperature=3.0,      # softens teacher output distribution
    alpha=0.5,            # balance: 0=task only, 1=distillation only
    entropy_weight=0.1,   # spike entropy regularisation
)

# Teacher output: 32 timesteps, 10 classes
teacher_logits = np.random.randn(32, 10).astype(np.float32)

# Student output: 4 timesteps, 10 classes
student_logits = np.random.randn(4, 10).astype(np.float32)

# Ground truth
targets = np.zeros(10, dtype=np.float32)
targets[3] = 1.0

result = loss_fn.compute(student_logits, teacher_logits, targets)
print(f"Total loss:       {result['total_loss']:.4f}")
print(f"Distillation loss: {result['distill_loss']:.4f}")
print(f"Task loss:         {result['task_loss']:.4f}")
print(f"Entropy loss:      {result['entropy_loss']:.4f}")
```

### Loss Components

1. **Task loss:** Cross-entropy between student output and ground truth
2. **Distillation loss:** KL divergence between softened teacher and
   student output distributions (per-timestep, then aggregated)
3. **Entropy loss:** Encourages spike diversity — prevents the student
   from collapsing to always-fire or never-fire

The `alpha` parameter controls the balance. Start at 0.5 (equal weight),
increase to 0.7-0.9 when the teacher is significantly better.

## Self-Distillation

No separate teacher model needed. The same model runs at extended
timesteps (T=32) to generate soft targets, then trains at reduced
timesteps (T=8):

```python
from sc_neurocore.distillation import SelfDistiller

distiller = SelfDistiller(
    T_teacher=32,      # extended timesteps for soft targets
    T_student=8,       # deployment timesteps
    temperature=3.0,
)

# Your SNN forward pass
def run_model(inputs, T):
    # Run model for T timesteps, return spike counts per class
    return np.random.randn(10)

# Generate soft targets from the same model at T=32
inputs = np.random.randn(784).astype(np.float32)
soft_targets = distiller.generate_targets(run_model, inputs=inputs)
print(f"Soft target shape: {soft_targets.shape}")  # (10,)
print(f"Soft target entropy: {-np.sum(soft_targets * np.log(soft_targets + 1e-8)):.3f}")
```

Self-distillation is cheaper than training a separate teacher and works
well when the model architecture is fixed — you just want fewer
timesteps at inference.

## Training Recipe

```python
# Full distillation training loop
from sc_neurocore.distillation import TemporalDistillationLoss

loss_fn = TemporalDistillationLoss(temperature=3.0, alpha=0.7)

# Phase 1: Train teacher at T=32 (standard training)
# ... train_epoch(teacher_model, ..., n_timesteps=32)

# Phase 2: Distil to student at T=4
for epoch in range(20):
    for x, y in train_loader:
        # Teacher forward (no gradients needed)
        with torch.no_grad():
            teacher_out = teacher_model(x.unsqueeze(0).expand(32, *x.shape))

        # Student forward
        student_out = student_model(x.unsqueeze(0).expand(4, *x.shape))

        # Distillation loss
        loss = loss_fn.compute(student_out, teacher_out, y)
        loss["total_loss"].backward()
        optimizer.step()
```

## When to Use

| Scenario | Method |
|----------|--------|
| Large teacher exists, need small student | Standard distillation |
| Same architecture, need fewer timesteps | Self-distillation |
| No teacher budget, need fast training | Self-distillation |
| Cross-architecture (CNN teacher → SNN student) | ANN-to-SNN distillation |

## Comparison

| Feature | SC-NeuroCore | snnTorch | Norse |
|---------|:-----------:|:--------:|:-----:|
| Temporal distillation | Yes | No | No |
| Self-distillation | Yes | No | No |
| Per-timestep matching | Yes | — | — |
| Entropy regularisation | Yes | — | — |
| FPGA-aware (target T) | Yes | — | — |

## References

- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."
  arXiv:1503.02531.
- Kushawaha et al. (2021). "Distilling Spikes: Knowledge Distillation
  in Spiking Neural Networks." ICPR 2021.
- Xu et al. (2023). "Temporal Knowledge Distillation for Efficient SNNs."
  AAAI 2023.
