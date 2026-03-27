# Tutorial 58: Continual Learning — Train, Deploy, Adapt

Build SNNs that keep learning after deployment without forgetting
previous tasks. Combines Elastic Weight Consolidation (EWC) for
catastrophic forgetting prevention with on-chip local learning rules
for post-deployment adaptation.

## The Problem

Standard neural networks forget previous tasks when trained on new
data (catastrophic forgetting). For deployed SNNs — robots, BCI
systems, always-on sensors — the model must adapt to changing
conditions without losing its original capabilities.

## Pipeline Overview

```
1. Train on Task A (standard backprop)
2. Compute Fisher Information → identify critical weights
3. Train on Task B with EWC penalty → learn B without forgetting A
4. Extract plasticity parameters → on-chip learning rules
5. Deploy with active local learning → adapts in the field
```

## Quick Start

```python
from sc_neurocore.continual import ContinualLearner
import numpy as np

# Initial trained weights (2 layers)
rng = np.random.default_rng(42)
weights = [
    rng.standard_normal((64, 32)).astype(np.float32) * 0.3,
    rng.standard_normal((10, 64)).astype(np.float32) * 0.3,
]

cl = ContinualLearner(weights, layer_names=["hidden", "output"])
```

### Step 1: Register Task A

After training on Task A, compute the Fisher Information Matrix to
identify which weights are critical:

```python
# Gradients collected during Task A validation
gradients = [
    [rng.standard_normal((64, 32)).astype(np.float32) * 0.1,
     rng.standard_normal((10, 64)).astype(np.float32) * 0.1]
    for _ in range(100)
]

cl.compute_fisher(gradients)
cl.register_task(accuracy=0.95)
print(f"Task A registered: {cl.n_tasks} tasks, accuracy 0.95")
```

### Step 2: Train Task B with EWC Protection

The EWC penalty prevents large changes to weights that the Fisher
Information identifies as critical for Task A:

```python
# In your training loop for Task B:
penalty = cl.ewc_penalty()
# total_loss = task_b_loss + lambda * penalty
print(f"EWC penalty: {penalty:.4f}")
```

The penalty is:
```
L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta_A_i)^2
```

Where F_i is the Fisher Information for weight i, theta_A_i is the
weight value after Task A, and theta_i is the current weight. Critical
weights (high F_i) resist change; unimportant weights (low F_i) are
free to adapt.

### Step 3: Extract On-Chip Plasticity

For FPGA deployment, convert the EWC-protected weights to local
learning rules that can run on-chip:

```python
configs = cl.extract_plasticity_configs()
for c in configs:
    print(f"{c.layer_name}: rule={c.rule}, "
          f"A+={c.lr_potentiation:.4f}, A-={c.lr_depression:.4f}, "
          f"protected={c.n_protected_weights}")
```

On-chip, each synapse has a plasticity flag: protected synapses use
reduced learning rates, unprotected synapses adapt freely.

### Step 4: Report

```python
report = cl.report()
print(report.summary())
# Tasks: 2
# Task A accuracy: 0.95 (protected)
# Task B accuracy: 0.92
# Weight overlap: 34% (shared critical weights)
# Forgetting: 0.02 (Task A accuracy drop after Task B training)
```

## Multi-Task Scaling

EWC scales to many tasks. Each new task adds its own Fisher diagonal
(accumulated, not stored separately):

```python
for task_id in range(10):
    # Train on task
    # ...
    cl.compute_fisher(task_gradients)
    cl.register_task(accuracy=task_accuracy)

print(f"Tasks: {cl.n_tasks}")
print(f"Protected weight fraction: {cl.protected_fraction:.1%}")
# After 10 tasks, ~60-80% of weights are protected
# Remaining 20-40% are free for new learning
```

## Comparison with Other Approaches

| Method | Memory | Accuracy | Hardware |
|--------|--------|----------|----------|
| EWC (this tutorial) | O(N) per task | Good | FPGA/CPU |
| PackNet (pruning) | O(1) per task | Good | FPGA/CPU |
| Progressive nets | O(N) per task (new columns) | Best | GPU only |
| Replay buffer | O(data) | Good | GPU only |
| On-chip STDP | O(1) | Limited | Neuromorphic |

SC-NeuroCore combines EWC (training phase) with on-chip STDP
(deployment phase) for the best of both worlds.

## FPGA Deployment

```python
# Export for FPGA with plasticity metadata
export = cl.export_for_fpga(target="ice40")
# Generates: weights + Fisher masks + STDP parameters
# Protected synapses have reduced STDP learning rate
```

## References

- Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in
  neural networks." PNAS 114(13):3521-3526.
- Zenke et al. (2017). "Continual Learning Through Synaptic
  Intelligence." ICML 2017.
- Kim et al. (2022). "Continual Learning in Spiking Neural Networks
  via Online Weight Consolidation." IEEE TNNLS.
