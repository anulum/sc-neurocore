# Recorders

Spike train recording and analysis utilities.

- `BitstreamSpikeRecorder` — Records spikes as 1D bitstream (0/1 per timestep). Provides: total spike count, firing rate (Hz given dt in ms), ISI histogram, raster data for plotting.

```python
from sc_neurocore import BitstreamSpikeRecorder

recorder = BitstreamSpikeRecorder()
for t in range(1000):
    spike = neuron.step(current)
    recorder.record(spike)

print(f"Total spikes: {recorder.total_spikes}")
print(f"Firing rate: {recorder.firing_rate(dt=1.0):.1f} Hz")
```

::: sc_neurocore.recorders.spike_recorder.BitstreamSpikeRecorder
    options:
      show_root_heading: true
