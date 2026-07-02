# Homeostasis

Homeostatic regulation: self-stabilising SNN without manual tuning.

Adjusts population firing thresholds and learning-rate scaling to maintain
target firing rates. The public regulator rejects malformed scalar and array
inputs before changing thresholds or learning rates, including bool aliases,
empty rate vectors, non-finite values, non-numeric arrays, and invalid
sleep-consolidation seeds.

- Threshold adaptation: overactive populations raise thresholds; quiet
  populations lower thresholds.
- Learning-rate scaling: high firing-rate variance reduces the caller-provided
  learning rate.
- Sleep consolidation: finite non-empty weight arrays receive deterministic
  power-law decay plus optional replay noise.

```python
from sc_neurocore.homeostasis import NetworkRegulator, SleepConsolidation
```

See [Tutorial 68: Homeostasis](../tutorials/68_homeostasis.md).

::: sc_neurocore.homeostasis
    options:
      show_root_heading: true
