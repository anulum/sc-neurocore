# Spike Normalization

5 SNN-specific batch normalization variants that handle temporal dynamics and threshold interaction.

| Class | Technique | Reference |
|-------|-----------|-----------|
| `ThresholdDependentBN` | BN scaled by firing threshold | Zheng 2021 |
| `PerTimestepBN` | Separate statistics per timestep | Kim & Panda 2021 |
| `TemporalEffectiveBN` | Per-timestep scaling factor | Duan 2022 (NeurIPS) |
| `MembranePotentialBN` | BN on membrane, folds into threshold at inference | Guo 2023 (ICCV) |
| `TemporalAccumulatedBN` | Normalizes accumulated membrane | Jiang 2024 (ICLR) |

`MembranePotentialBN.fused_threshold()` returns per-neuron thresholds that absorb BN at inference — zero compute overhead on hardware.

```python
from sc_neurocore.spike_norm import (
    ThresholdDependentBN, PerTimestepBN, TemporalEffectiveBN,
    MembranePotentialBN, TemporalAccumulatedBN,
)
```

See [Tutorial 75: Spike Normalization](../tutorials/75_spike_norm.md) for usage examples.

::: sc_neurocore.spike_norm.normalizers
    options:
      show_root_heading: true
      members:
        - ThresholdDependentBN
        - PerTimestepBN
        - TemporalEffectiveBN
        - MembranePotentialBN
        - TemporalAccumulatedBN
