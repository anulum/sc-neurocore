# Tutorial 58: Continual Learning — Train, Deploy, Adapt

Build SNNs that keep learning after deployment.

## The Pipeline

1. Train with backprop (initial model)
2. Compute Fisher Information for EWC protection
3. Train on new task with EWC penalty
4. Extract plasticity parameters for on-chip deployment
5. Deploy with active local learning rules

## Quick Start

```python
from sc_neurocore.continual import ContinualLearner
import numpy as np

weights = [np.random.randn(64, 32) * 0.3, np.random.randn(10, 64) * 0.3]
cl = ContinualLearner(weights, layer_names=["hidden", "output"])

# After training task 1: compute Fisher + register
gradients = [[np.random.randn(64, 32), np.random.randn(10, 64)] for _ in range(100)]
cl.compute_fisher(gradients)
cl.register_task(accuracy=0.95)

# Train task 2 with EWC protection
# In your training loop: loss = task_loss + cl.ewc_penalty()
penalty = cl.ewc_penalty()

# Extract plasticity configs for on-chip deployment
configs = cl.extract_plasticity_configs()
for c in configs:
    print(f"{c.layer_name}: rule={c.rule}, A+={c.lr_potentiation:.4f}")
```

## Report

```python
report = cl.report()
print(report.summary())
```
