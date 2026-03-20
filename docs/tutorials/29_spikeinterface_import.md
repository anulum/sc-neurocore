# Tutorial 29: Import Experimental Data via SpikeInterface

SC-NeuroCore includes a SpikeInterface adapter for importing real
electrophysiology data into the SC simulation pipeline.

## From Spike Times (No Dependencies)

```python
import numpy as np
from sc_neurocore.adapters.spikeinterface import (
    spike_trains_to_bitstreams,
    spike_trains_to_population_input,
    firing_rates_to_sc_probs,
)

# Your spike data: unit_id → spike times (ms)
spike_times = {
    0: np.array([10.0, 25.0, 40.0, 80.0, 120.0]),
    1: np.array([15.0, 50.0, 90.0]),
    2: np.array([5.0, 30.0, 60.0, 100.0, 140.0, 180.0]),
}

# Convert to binary bitstream matrix (n_units × n_bins)
bitstreams = spike_trains_to_bitstreams(spike_times, duration_ms=200.0, dt=1.0)
print(f"Bitstream matrix: {bitstreams.shape}")  # (3, 200)

# Convert to Population input currents (n_timesteps × n_units)
currents = spike_trains_to_population_input(spike_times, duration_ms=200.0)
print(f"Current matrix: {currents.shape}")  # (200, 3)

# Convert firing rates to SC probabilities
probs = firing_rates_to_sc_probs(spike_times, duration_ms=200.0, max_rate_hz=50.0)
print(f"SC probabilities: {probs}")  # [0.5, 0.3, 0.6]
```

## Feed Into SC Layer

```python
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer

layer = SCDenseLayer(n_inputs=3, n_neurons=2, length=256)
# Use firing rates as input probabilities
output = layer.forward(probs.tolist())
print(f"SC layer output: {output}")
```

## From SpikeInterface SortingExtractor

If you have SpikeInterface installed:

```python
from sc_neurocore.adapters.spikeinterface import from_sorting

# sorting = si.read_sorting("path/to/results")
bitstreams = from_sorting(sorting, dt=1.0)
```
