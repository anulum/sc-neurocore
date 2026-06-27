# Recorders

Spike train recording and analysis utilities.

- `BitstreamSpikeRecorder` — records a 1D binary spike train (`0`/`1` per
  timestep), returns a `uint8` NumPy view, counts spikes, computes mean firing
  rate from `dt_ms`, and builds inter-spike-interval histograms.

```python
from sc_neurocore import BitstreamSpikeRecorder

recorder = BitstreamSpikeRecorder(dt_ms=1.0)
for _ in range(1000):
    spike = neuron.step(current)
    recorder.record(spike)

print(f"Total spikes: {recorder.total_spikes()}")
print(f"Firing rate: {recorder.firing_rate_hz():.1f} Hz")
```

The recorder rejects non-binary spikes, negative `dt_ms` values, and non-positive
histogram bin counts. `dt_ms=0.0` is still accepted for legacy dry-run callers and
returns a firing rate of `0.0`.

## Polyglot Mirrors

The Python API is the canonical recorder surface. Parity helper surfaces are kept
in `accel/julia/recorders/spike_recorder.jl`,
`accel/rust/safety/spike_recorder.rs`, and
`accel/mojo/kernels/spike_recorder.mojo` for backend validation and low-level
firing-rate/ISI helpers. The Rust safety mirror owns crate-level unit tests; the
Julia and Mojo mirrors are syntax-checked in the recorder maintenance lane.

::: sc_neurocore.recorders.spike_recorder.BitstreamSpikeRecorder
    options:
      show_root_heading: true
