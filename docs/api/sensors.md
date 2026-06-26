# Sensors and DVS Pipeline

Event camera (DVS) data loading, preprocessing, and spike encoding. The public
package exports `DVSLoader`, `events_to_spike_trains`, and `events_to_frames`.

```python
from sc_neurocore.sensors import DVSLoader, events_to_spike_trains

loader = DVSLoader(width=128, height=128)
events = loader.from_numpy(raw_events)
spikes = events_to_spike_trains(events, width=128, height=128)
```

See [Tutorial 45: DVS Pipeline](../tutorials/45_dvs_pipeline.md).

## API

::: sc_neurocore.sensors.dvs
    options:
      show_root_heading: true
