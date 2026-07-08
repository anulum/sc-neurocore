# Sensors and DVS Pipeline

Event camera (DVS) data loading, preprocessing, spike encoding, and bit-true
ADC-to-spike window encoding. The public package exports `DVSLoader`,
`events_to_spike_trains`, `events_to_frames`, `ADCSpikeWindowConfig`,
`ADCSpikeWindowResult`, `adc_to_spike_windows`, `adc_to_spike_windows_q`,
`available_backends`, and `quantise_adc`.

```python
from sc_neurocore.sensors import DVSLoader, events_to_spike_trains

loader = DVSLoader(width=128, height=128)
events = loader.from_numpy(raw_events)
spikes = events_to_spike_trains(events, width=128, height=128)
```

```python
from sc_neurocore.sensors import ADCSpikeWindowConfig, adc_to_spike_windows

config = ADCSpikeWindowConfig(decimation=8, threshold_q=256)
windows = adc_to_spike_windows(raw_adc_samples, config, backend="auto")
```

See [Tutorial 45: DVS Pipeline](../tutorials/45_dvs_pipeline.md).

## API

::: sc_neurocore.sensors.dvs
    options:
      show_root_heading: true

::: sc_neurocore.sensors.adc_to_spike_kernel
    options:
      show_root_heading: true
