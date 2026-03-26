# Contrastive Self-Supervised Learning — InfoNCE + CSDP

Self-supervised learning for SNNs without labeled data. Two complementary approaches: global InfoNCE loss for batch training and local CSDP rule for biologically plausible on-chip learning.

## SpikeContrastiveLoss — InfoNCE for Spikes

Adapted InfoNCE contrastive loss for spike-rate representations. Given two augmented views of the same batch, positive pairs = same input (different augmentation), negative pairs = different inputs. The loss encourages representations of the same input to be similar, and different inputs to be dissimilar.

`loss = -mean(log(exp(sim(a_i, b_i)/τ) / Σ_j exp(sim(a_i, b_j)/τ)))`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `temperature` | 0.5 | Contrastive temperature scaling |

Returns 0.0 for batch size < 2 (no negatives possible).

## CSDPRule — Contrastive Signal-Dependent Plasticity

Biologically plausible local learning rule. Generalizes the Forward-Forward algorithm to spiking circuits:

- **Positive phase:** Present real data → Hebbian update: `dW = lr * (post ⊗ pre) - decay * W`
- **Negative phase:** Present corrupted data → anti-Hebbian update: `dW = -lr * (post ⊗ pre)`
- **Goodness:** `g = Σ(activations²)` — positive data should have high goodness, negative data low

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lr` | 0.01 | Learning rate |
| `decay` | 0.001 | Weight decay |

## Usage

```python
from sc_neurocore.contrastive import SpikeContrastiveLoss, CSDPRule
import numpy as np

# InfoNCE training
loss_fn = SpikeContrastiveLoss(temperature=0.5)
view_a = np.random.randn(32, 128)  # batch of 32, 128 features
view_b = np.random.randn(32, 128)  # augmented version
loss = loss_fn.compute(view_a, view_b)

# CSDP local learning
csdp = CSDPRule(lr=0.01)
W = np.random.randn(64, 32) * 0.1
W = csdp.contrastive_step(
    W,
    pos_pre=real_spikes, pos_post=real_activations,
    neg_pre=noise_spikes, neg_post=noise_activations,
)
```

**Reference:** Ororbia 2024, Science Advances.

See [Tutorial 80: Contrastive SSL](../tutorials/80_contrastive.md).

::: sc_neurocore.contrastive.ssl
    options:
      show_root_heading: true
      members:
        - SpikeContrastiveLoss
        - CSDPRule
