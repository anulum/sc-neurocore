<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 45: DVS Event Camera Pipeline

Process Dynamic Vision Sensor data: load events, convert to spike trains
or frames, feed into SNN networks, deploy to FPGA.

## 1. Load Events

```python
import numpy as np
from sc_neurocore.sensors import DVSLoader

loader = DVSLoader(width=128, height=128)
events = np.zeros(1000, dtype=[('x', np.int32), ('y', np.int32),
                                ('t', np.int64), ('p', np.int8)])
events['x'] = np.random.randint(0, 128, 1000)
events['y'] = np.random.randint(0, 128, 1000)
events['t'] = np.sort(np.random.randint(0, 100000, 1000))
events['p'] = np.random.choice([0, 1], 1000)
loaded = loader.from_numpy(events)
```

## 2. Convert to Spike Trains

```python
from sc_neurocore.sensors import events_to_spike_trains
spikes = events_to_spike_trains(events, 128, 128, dt_us=1000)
# Shape: (n_bins, 2 * width * height) with ON/OFF channels
```

## 3. Convert to Frames

```python
from sc_neurocore.sensors import events_to_frames
frames = events_to_frames(events, 128, 128, dt_us=10000)
# Shape: (n_frames, 2, height, width)
```

## 4. Tonic Integration

```python
# pip install tonic
events, label = loader.from_tonic("nmnist", index=0)
```

## Further Reading

- [Tutorial 49: Multimodal Fusion](49_multimodal_fusion.md)
- [API: Sensors](../api/sensors.md)
