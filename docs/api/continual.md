# Continual Learning — EWC + On-Chip Plasticity

Train with backprop, deploy with local plasticity. Combines Elastic Weight Consolidation (EWC) for catastrophic forgetting protection with STDP-based local learning rules that can run on-chip.

## Pipeline

1. **Train task A** with standard backprop
2. **Compute Fisher diagonal** from per-sample gradients → identifies important parameters
3. **Train task B** with EWC penalty: `L_ewc = (λ/2) * Σ F_i * (θ_i - θ*_i)²`
4. **Extract plasticity configs** for on-chip deployment (STDP parameters derived from weight statistics)
5. **Deploy** with active on-chip plasticity rules

No framework provides the integrated pipeline from "trained model" to "deployed model with active on-chip plasticity."

## Components

- **`ContinualLearner`** — Main engine managing EWC and plasticity extraction.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `weights` | (required) | List of weight matrices per layer |
| `layer_names` | auto | Names for each layer |
| `ewc_lambda` | 1000.0 | EWC regularization strength |
| `plasticity_rule` | "stdp" | Default on-chip plasticity rule |

**Key methods:**

- `compute_fisher(gradients_per_sample)` — Compute Fisher Information diagonal from per-sample gradients
- `ewc_penalty()` → float — Current EWC regularization penalty
- `register_task(accuracy)` — Register task completion
- `update_weights(new_weights)` — Update weights after training
- `extract_plasticity_configs()` → list of `PlasticityConfig` — Derive on-chip deployment parameters
- `report()` → `ContinualReport` — Full report with accuracy history

- **`PlasticityConfig`** — Per-layer on-chip plasticity configuration (rule, tau_pre/post, LR, weight bounds, homeostatic target).
- **`ContinualReport`** — Report dataclass with `summary()` method.

## Usage

```python
from sc_neurocore.continual import ContinualLearner
import numpy as np

weights = [np.random.randn(64, 32) * 0.3, np.random.randn(10, 64) * 0.3]
learner = ContinualLearner(weights, layer_names=["hidden", "output"])

# After training task A: compute Fisher
gradients = [[np.random.randn(64, 32), np.random.randn(10, 64)] for _ in range(100)]
learner.compute_fisher(gradients)
learner.register_task(accuracy=0.95)

# Training task B: EWC penalty prevents forgetting
print(f"EWC penalty: {learner.ewc_penalty():.4f}")

# Deploy with on-chip plasticity
configs = learner.extract_plasticity_configs()
for c in configs:
    print(f"{c.layer_name}: rule={c.rule}, lr+={c.lr_potentiation:.4f}")
```

**Reference:** Kirkpatrick et al. 2017 — "Overcoming catastrophic forgetting in neural networks" (EWC).

See [Tutorial 58: Continual Learning](../tutorials/58_continual_learning.md).

::: sc_neurocore.continual
    options:
      show_root_heading: true
