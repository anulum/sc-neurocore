# Homeostasis

Homeostatic regulation: self-stabilizing SNN without manual tuning.

Adjusts firing thresholds and synaptic scaling to maintain target firing rates. Prevents both silence (no spikes) and epileptic runaway (all spikes). Works at population level.

- Threshold adaptation: neurons that fire too much raise their threshold, and vice versa
- Synaptic scaling: global scaling of excitatory/inhibitory balance

```python
from sc_neurocore.homeostasis import HomeostaticRegulator
```

See [Tutorial 68: Homeostasis](../tutorials/68_homeostasis.md).

::: sc_neurocore.homeostasis
    options:
      show_root_heading: true
