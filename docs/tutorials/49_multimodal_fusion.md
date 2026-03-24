<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 49: Multimodal Spike Fusion

Fuse spike trains from DVS cameras, audio, IMU, and other sensors
into a unified stream for SNN processing.

## 1. Define Modalities

```python
from sc_neurocore.fusion import MultiModalFusion
from sc_neurocore.fusion.multimodal import ModalityConfig

dvs = ModalityConfig("dvs", n_channels=128, dt_us=1000)
audio = ModalityConfig("audio", n_channels=32, dt_us=500)
```

## 2. Fuse

```python
import numpy as np

fuser = MultiModalFusion([dvs, audio], output_dt_us=1000, mode="concatenate")
fused = fuser.fuse({
    "dvs": np.random.randint(0, 2, (100, 128)),
    "audio": np.random.randint(0, 2, (200, 32)),
}, duration_us=100000)
print(f"Fused: {fused.shape}")  # (100, 160)
```

## 3. Fusion Modes

- `concatenate`: stack all channels (default)
- `sum`: element-wise OR across modalities
- `attention`: weighted combination per modality

## Further Reading

- [Tutorial 45: DVS Pipeline](45_dvs_pipeline.md)
- [API: Fusion](../api/fusion.md)
