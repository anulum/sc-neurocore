# Spike Augmentation

Spike-aware data augmentation: temporal jitter, spike dropout, rate scaling, noise injection, time reversal. Preserves spike structure unlike image augmentation.

```python
from sc_neurocore.augmentation import SpikeAugmenter

aug = SpikeAugmenter(jitter_ms=1.0, dropout_rate=0.1)
augmented = aug.transform(spike_train)
```

See [Tutorial 57: Spike Augmentation](../tutorials/57_spike_augmentation.md).

::: sc_neurocore.augmentation
    options:
      show_root_heading: true
