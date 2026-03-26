# Multimodal Fusion — Cross-Sensor Spike Train Merging

Fuse spike trains from multiple sensor modalities (vision/DVS, audio/cochlea, IMU) into a unified representation. Handles different time resolutions, firing rates, and channel counts.

## Fusion Modes

| Mode | Description | Output Channels |
|------|------------|-----------------|
| `concatenate` | Stack channels from all modalities | sum(n_channels) |
| `sum` | Element-wise OR (any-modality spike), pad smaller modalities | max(n_channels) |
| `attention` | Learned cross-modal weighting per modality | sum(n_channels) |

All modes include automatic timebase resampling (bin mapping from modality dt to output dt) and rate normalization (scale so max rate maps to 1.0).

## Components

- **`ModalityConfig`** — Configuration for one sensor modality.

| Field | Type | Meaning |
|-------|------|---------|
| `name` | str | Modality identifier (e.g., "dvs", "audio") |
| `n_channels` | int | Channel count |
| `dt_us` | float | Time bin width in microseconds |
| `max_rate_hz` | float | Maximum expected firing rate (default 1000) |

- **`MultiModalFusion`** — Main fusion engine.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `modalities` | (required) | List of ModalityConfig |
| `output_dt_us` | 1000.0 | Output time bin width (common timebase) |
| `mode` | "concatenate" | Fusion mode |

## Usage

```python
from sc_neurocore.fusion.multimodal import ModalityConfig, MultiModalFusion
import numpy as np

# Define sensor modalities
dvs = ModalityConfig("dvs", n_channels=128, dt_us=100.0)
audio = ModalityConfig("audio", n_channels=64, dt_us=500.0)

# Create fuser
fuser = MultiModalFusion([dvs, audio], output_dt_us=100.0, mode="concatenate")

# Fuse spike trains
spikes = {
    "dvs": np.random.randint(0, 2, (100, 128)),
    "audio": np.random.randint(0, 2, (100, 64)),
}
fused = fuser.fuse(spikes, duration_us=10000.0)
print(f"Output shape: {fused.shape}")  # (100, 192)

# Missing modality → zero-filled
fused_partial = fuser.fuse({"dvs": spikes["dvs"]}, duration_us=10000.0)
```

See [Tutorial 49: Multimodal Fusion](../tutorials/49_multimodal_fusion.md).

::: sc_neurocore.fusion
    options:
      show_root_heading: true
