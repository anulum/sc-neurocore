# Contrastive Self-Supervised Learning

Self-supervised learning for SNNs without labeled data.

- `SpikeContrastiveLoss` — InfoNCE contrastive loss adapted for spike-rate representations. Matches positive pairs (same input, different augmentation) and separates negative pairs.
- `CSDPRule` — Contrastive Signal-Dependent Plasticity. Biologically plausible local learning: Hebbian on real data, anti-Hebbian on corrupted data. Generalizes Forward-Forward to spiking circuits. (Ororbia 2024, Science Advances)

```python
from sc_neurocore.contrastive import SpikeContrastiveLoss, CSDPRule
```

See [Tutorial 80: Contrastive SSL](../tutorials/80_contrastive.md) for usage examples.

::: sc_neurocore.contrastive.ssl
    options:
      show_root_heading: true
      members:
        - SpikeContrastiveLoss
        - CSDPRule
