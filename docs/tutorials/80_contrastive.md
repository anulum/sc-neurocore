# Tutorial 80: Contrastive Self-Supervised Learning

Self-supervised learning for SNNs without labeled data. InfoNCE contrastive
loss for spike representations, and CSDP — a biologically plausible local
learning rule (Forward-Forward generalized to spiking circuits).

## SpikeContrastiveLoss (InfoNCE)

```python
import numpy as np
from sc_neurocore.contrastive import SpikeContrastiveLoss

loss_fn = SpikeContrastiveLoss(temperature=0.5)

rng = np.random.RandomState(42)
view_a = np.abs(rng.randn(16, 128))  # augmentation A
view_b = view_a + rng.randn(16, 128) * 0.1  # augmentation B

loss = loss_fn.compute(view_a, view_b)
print(f"Contrastive loss: {loss:.3f}")
```

## CSDP: Biologically Plausible Learning

Positive phase (real data) → Hebbian. Negative phase (corrupted) → anti-Hebbian.

```python
from sc_neurocore.contrastive import CSDPRule

csdp = CSDPRule(lr=0.01, decay=0.001)
W = np.random.randn(64, 128) * 0.01

pos_pre = (np.random.rand(128) > 0.5).astype(float)
pos_post = (np.random.rand(64) > 0.5).astype(float)
neg_pre = np.random.rand(128)
neg_post = (np.random.rand(64) > 0.5).astype(float)

W = csdp.contrastive_step(W, pos_pre, pos_post, neg_pre, neg_post)

# Goodness score: positive data should score higher
print(f"Pos goodness: {csdp.goodness(pos_post):.2f}")
```

Reference: Ororbia 2024, Science Advances

## API Reference

::: sc_neurocore.contrastive.ssl
    options:
      show_root_heading: true
