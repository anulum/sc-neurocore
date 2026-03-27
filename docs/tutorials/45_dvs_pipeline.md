<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 45: DVS Event Camera Pipeline

Dynamic Vision Sensors (DVS) output asynchronous events instead of
frames — each pixel independently reports brightness changes with
microsecond resolution. This event-driven data is a natural match for
spiking neural networks. This tutorial covers the full pipeline: load
events, convert to spike representations, classify with an SNN, and
deploy to FPGA.

## Event Camera Basics

A DVS pixel fires when log-intensity changes by a threshold:

```
event = (x, y, timestamp_us, polarity)
```

- **x, y**: pixel coordinates
- **timestamp**: microsecond precision
- **polarity**: +1 (ON, brightness increase) or -1 (OFF, decrease)

Advantages over frame cameras: no motion blur, >120 dB dynamic range,
microsecond latency, low power (~10 mW). Datasets: N-MNIST, CIFAR10-DVS,
DVS128 Gesture, Gen1 Automotive.

## 1. Load Events

```python
import numpy as np
from sc_neurocore.sensors import DVSLoader

loader = DVSLoader(width=128, height=128)

# From structured NumPy array (standard event format)
events = np.zeros(10000, dtype=[
    ('x', np.int32), ('y', np.int32),
    ('t', np.int64), ('p', np.int8),
])
events['x'] = np.random.randint(0, 128, 10000)
events['y'] = np.random.randint(0, 128, 10000)
events['t'] = np.sort(np.random.randint(0, 1_000_000, 10000))  # 1 second
events['p'] = np.random.choice([-1, 1], 10000)

loaded = loader.from_numpy(events)
print(f"Loaded {len(loaded)} events over {loaded['t'].ptp() / 1e6:.2f}s")
```

### From Tonic (community datasets)

```python
# pip install tonic
# Tonic provides N-MNIST, CIFAR10-DVS, DVS128 Gesture, etc.
events, label = loader.from_tonic("nmnist", index=0)
print(f"N-MNIST digit {label}: {len(events)} events")
```

## 2. Convert to Spike Trains

Bin events into discrete timesteps for SNN processing:

```python
from sc_neurocore.sensors import events_to_spike_trains

spike_trains = events_to_spike_trains(
    events, width=128, height=128,
    dt_us=1000,  # 1ms bins
)
# Shape: (n_bins, 2 * 128 * 128) — ON and OFF channels concatenated
print(f"Spike tensor: {spike_trains.shape}")
print(f"Mean spike rate: {spike_trains.mean():.4f}")
```

The `dt_us` parameter controls temporal resolution:
- 1000 µs (1 ms): standard SNN timestep, good balance
- 100 µs: high temporal resolution, many bins, sparse
- 10000 µs (10 ms): coarse, fewer bins, denser per bin

## 3. Convert to Frames

For hybrid approaches or visualisation, accumulate events into frames:

```python
from sc_neurocore.sensors import events_to_frames

frames = events_to_frames(
    events, width=128, height=128,
    dt_us=10000,  # 10ms per frame = 100 FPS equivalent
)
# Shape: (n_frames, 2, 128, 128) — ON and OFF channels
print(f"Frame tensor: {frames.shape}")
```

Frame representation loses temporal precision but enables use of
standard CNN architectures. For SNN-native processing, use spike trains.

## 4. Classify with SNN

Feed spike trains into a spiking network:

```python
from sc_neurocore.training import SpikingNet, train_epoch, auto_device
import torch
from torch.utils.data import TensorDataset, DataLoader

device = auto_device()

# Create model: 2 * 128 * 128 inputs → 256 hidden → 10 classes
model = SpikingNet(
    n_input=2 * 128 * 128,  # ON + OFF channels
    n_hidden=256,
    n_output=10,
    n_layers=2,
    learn_beta=True,
).to(device)

# Prepare data (spike_trains is already temporal)
# For N-MNIST: shape is (T, batch, 2*128*128)
x = torch.from_numpy(spike_trains).float().unsqueeze(1)  # add batch dim
y = torch.tensor([label])

# Training loop uses the standard train_epoch
# (spike_trains are already in temporal format)
```

## 5. Deploy to FPGA

The entire pipeline from event input to classification can run on FPGA:

```bash
# In the Studio:
# 1. Design your network on the Canvas
# 2. Train in the Training Monitor
# 3. Click Pipeline → ice40 to compile and synthesise

# Or via CLI:
sc-neurocore deploy model.nir --target ice40
```

The event-driven nature of DVS data matches SC-NeuroCore's event-driven
RTL (AER encoder/decoder/router). Events arrive asynchronously, get
routed to the correct neuron population, and spikes propagate through
the network — all in hardware, with microsecond latency.

## Performance Comparison

| Framework | DVS Support | SNN Training | FPGA Deploy |
|-----------|:-----------:|:------------:|:-----------:|
| SC-NeuroCore | Load + convert + train + deploy | Yes | Yes |
| Tonic | Load + convert | No | No |
| snnTorch | Via Tonic | Yes | No |
| SpikingJelly | Load + convert + train | Yes | No |
| Lava | Load + train | Yes | Loihi only |

SC-NeuroCore is the only framework that takes DVS events from raw
input through to FPGA deployment in a single tool.

## Datasets

| Dataset | Resolution | Classes | Events/sample | Duration |
|---------|-----------|---------|---------------|----------|
| N-MNIST | 34 × 34 | 10 | ~5K | 300 ms |
| CIFAR10-DVS | 128 × 128 | 10 | ~30K | 1 s |
| DVS128 Gesture | 128 × 128 | 11 | ~200K | 6 s |
| N-Caltech101 | 240 × 180 | 101 | ~50K | 300 ms |
| Gen1 Automotive | 304 × 240 | 2 | ~100K/frame | continuous |

All loadable via Tonic integration or direct NumPy structured arrays.

## References

- Lichtsteiner et al. (2008). "A 128×128 120dB 15µs Latency Asynchronous
  Temporal Contrast Vision Sensor." IEEE JSSC 43(2):566-576.
- Orchard et al. (2015). "Converting Static Image Datasets to Spiking
  Neuromorphic Datasets Using Saccades." Front. Neurosci. 9:437.
- Li et al. (2017). "CIFAR10-DVS: An Event-Stream Dataset for Object
  Classification." Front. Neurosci. 11:309.
- Amir et al. (2017). "A Low Power, Fully Event-Based Gesture
  Recognition System." CVPR 2017.
