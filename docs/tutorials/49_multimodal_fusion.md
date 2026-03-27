<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 49: Multimodal Spike Fusion

Fuse spike trains from multiple sensors — DVS cameras, microphones,
IMUs, tactile arrays — into a unified temporal representation for
SNN processing. Each modality may have different sampling rates,
channel counts, and event densities.

## Why Multimodal Fusion

Real-world neuromorphic systems combine multiple sensors:

| Application | Modalities | Why Fusion |
|-------------|-----------|------------|
| Autonomous drone | DVS + IMU + sonar | Obstacle avoidance needs vision + motion + distance |
| Prosthetic hand | Tactile + EMG + joint angles | Grasp control needs touch + intent + position |
| Smart home | Audio + DVS + PIR | Activity recognition from sound + vision + motion |
| Robotics | DVS + LiDAR + force | Manipulation needs vision + depth + contact |

## Define Modalities

Each modality has a channel count and native sampling rate:

```python
from sc_neurocore.fusion import MultiModalFusion
from sc_neurocore.fusion.multimodal import ModalityConfig
import numpy as np

# DVS camera: 128 channels (pixels), 1ms resolution
dvs = ModalityConfig("dvs", n_channels=128, dt_us=1000)

# Audio (cochlear): 32 frequency bands, 0.5ms resolution
audio = ModalityConfig("audio", n_channels=32, dt_us=500)

# IMU: 6 channels (3 accel + 3 gyro), 10ms resolution
imu = ModalityConfig("imu", n_channels=6, dt_us=10000)
```

## Fuse

The fuser resamples all modalities to a common timestep and combines
them:

```python
rng = np.random.default_rng(42)

fuser = MultiModalFusion(
    [dvs, audio, imu],
    output_dt_us=1000,       # unified 1ms output timestep
    mode="concatenate",      # stack all channels
)

# Generate spike data (different lengths due to different dt)
data = {
    "dvs": rng.integers(0, 2, (100, 128)).astype(np.float32),    # 100ms at 1ms
    "audio": rng.integers(0, 2, (200, 32)).astype(np.float32),   # 100ms at 0.5ms
    "imu": rng.integers(0, 2, (10, 6)).astype(np.float32),       # 100ms at 10ms
}

fused = fuser.fuse(data, duration_us=100000)
print(f"DVS:   {data['dvs'].shape}")     # (100, 128)
print(f"Audio: {data['audio'].shape}")    # (200, 32)
print(f"IMU:   {data['imu'].shape}")      # (10, 6)
print(f"Fused: {fused.shape}")            # (100, 166) = 128 + 32 + 6
```

### Resampling

Modalities faster than `output_dt_us` are binned (OR within each bin).
Modalities slower are upsampled (repeat last value). This preserves
event timing for fast sensors and avoids aliasing for slow ones.

## Fusion Modes

| Mode | Operation | Use Case |
|------|-----------|----------|
| `concatenate` | Stack channels: [dvs \| audio \| imu] | Separate processing per modality |
| `sum` | Element-wise OR across modalities | Shared representation (same channel count required) |
| `attention` | Weighted combination per modality | Learned importance weighting |

### Attention-Based Fusion

```python
fuser = MultiModalFusion(
    [dvs, audio],
    output_dt_us=1000,
    mode="attention",
    attention_dim=32,  # attention bottleneck dimension
)

# Attention weights are learned — modality importance adapts per timestep
fused = fuser.fuse(data, duration_us=100000)
weights = fuser.attention_weights()
print(f"DVS attention: {weights['dvs'].mean():.3f}")
print(f"Audio attention: {weights['audio'].mean():.3f}")
```

## Feed Into SNN

```python
from sc_neurocore.training import SpikingNet

# Fused input dimensionality = sum of all channels
n_input = sum(m.n_channels for m in [dvs, audio, imu])  # 166

model = SpikingNet(
    n_input=n_input,
    n_hidden=256,
    n_output=10,  # activity classes
)

# Train on fused multimodal spike data
# Each timestep of fused data feeds all modality channels simultaneously
```

## Temporal Alignment

When sensors have different latencies (DVS: ~1ms, audio: ~10ms,
IMU: ~5ms), the fuser can apply per-modality delay compensation:

```python
dvs = ModalityConfig("dvs", n_channels=128, dt_us=1000, latency_us=1000)
audio = ModalityConfig("audio", n_channels=32, dt_us=500, latency_us=10000)

# The fuser shifts audio data back by 9ms to align with DVS
```

## FPGA Deployment

On FPGA, each modality has its own AER input port. The fusion module
merges events from all ports into a single timestep buffer:

```
DVS AER  ──┐
Audio AER ─┤→ Fusion crossbar → Unified spike buffer → SNN
IMU AER  ──┘
```

## References

- Amir et al. (2017). "A Low Power, Fully Event-Based Gesture
  Recognition System." CVPR 2017.
- Shrestha & Orchard (2018). "SLAYER: Spike Layer Error Reassignment
  in Time." NeurIPS 2018.
- Li et al. (2022). "Neuromorphic Multimodal Sensor Fusion." Nature
  Electronics 5:830-840.
