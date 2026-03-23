<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 29: Import Experimental Data via SpikeInterface

SC-NeuroCore includes a SpikeInterface adapter for importing real
electrophysiology data into the SC simulation pipeline. Use recorded
spike trains from Neuropixels, Utah arrays, or any spike-sorted data
to drive SC networks or validate your models against biology.

## Overview

The adapter converts between three representations:

| Input format | Output format | Use case |
|---|---|---|
| Spike times (dict) | Binary bitstream matrix | Drive SC layers directly |
| Spike times (dict) | Population current matrix | Drive Network engine |
| Spike times (dict) | SC probabilities (firing rates) | Input to VectorizedSCLayer |
| SpikeInterface SortingExtractor | Any of the above | Lab data import |

## 1. From Spike Times (No Dependencies)

No SpikeInterface installation needed — just provide spike times as a dictionary:

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
print(f"Unit 0 spike count: {bitstreams[0].sum()}")
print(f"Unit 2 spike count: {bitstreams[2].sum()}")

# Convert to Population input currents (n_timesteps × n_units)
currents = spike_trains_to_population_input(spike_times, duration_ms=200.0)
print(f"Current matrix: {currents.shape}")  # (200, 3)

# Convert firing rates to SC probabilities
probs = firing_rates_to_sc_probs(spike_times, duration_ms=200.0, max_rate_hz=50.0)
print(f"SC probabilities: {probs}")  # [0.5, 0.3, 0.6] (rates normalized to [0,1])
```

## 2. Feed Into SC Layer

Use firing rates as input probabilities to an SC dense layer:

```python
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer

layer = SCDenseLayer(n_inputs=3, n_neurons=2, length=256)
output = layer.forward(probs.tolist())
print(f"SC layer output: {output}")
```

Or use a VectorizedSCLayer for higher performance:

```python
from sc_neurocore import VectorizedSCLayer

layer = VectorizedSCLayer(n_inputs=3, n_neurons=4, length=512)
output = layer.forward(probs.tolist())
print(f"Vectorized output: {output}")
```

## 3. Drive the Network Engine

Feed experimental data as time-varying stimulus into a population:

```python
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import TimedArray
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

# Convert spike trains to current matrix
currents = spike_trains_to_population_input(spike_times, duration_ms=200.0)

# Create a population driven by experimental data
pop = Population(HodgkinHuxleyNeuron, n=3, label="driven")
stimulus = TimedArray(currents * 5.0, dt=0.001)  # scale current
monitor = SpikeMonitor(pop)

net = Network(pop, stimulus, monitor)
net.run(duration=0.2, dt=0.001)
print(f"Model spikes: {monitor.count}")
```

## 4. From SpikeInterface SortingExtractor

If you have SpikeInterface installed (`pip install spikeinterface`):

```python
import spikeinterface as si
from sc_neurocore.adapters.spikeinterface import from_sorting

# Load spike sorting results
sorting = si.read_sorting("path/to/sorting_results")

# Convert to bitstream matrix
bitstreams = from_sorting(sorting, dt=1.0)
print(f"Units: {bitstreams.shape[0]}, Time bins: {bitstreams.shape[1]}")

# Convert sorting to spike times dict, then to SC probabilities
spike_dict = {uid: sorting.get_unit_spike_train(uid) / sorting.get_sampling_frequency() * 1000
              for uid in sorting.get_unit_ids()}
probs = firing_rates_to_sc_probs(spike_dict, duration_ms=1000.0, max_rate_hz=100.0)
```

### Supported formats

SpikeInterface reads data from:
- **Neuropixels** (SpikeGLX, Open Ephys)
- **Utah arrays** (Blackrock NEV/NSx)
- **Plexon** (PLX, PL2)
- **Neuralynx** (NCS, NTT)
- **Intan** (RHD, RHS)
- **NWB** (Neurodata Without Borders)
- **MDA** (MountainSort)

Any format SpikeInterface can read, SC-NeuroCore can consume.

## 5. Model Validation Against Biology

Compare your SC model's output against recorded data:

```python
import numpy as np
from sc_neurocore.analysis import (
    firing_rate,
    coefficient_of_variation,
    spike_train_correlation,
)

# Experimental spike trains
exp_trains = spike_times

# Model spike trains (from SpikeMonitor)
model_trains = monitor.spike_trains

# Compare statistics
for unit_id in exp_trains:
    exp_rate = firing_rate(exp_trains[unit_id], duration=0.2)
    model_rate = firing_rate(model_trains.get(unit_id, np.array([])), duration=0.2)
    print(f"Unit {unit_id}: exp={exp_rate:.1f} Hz, model={model_rate:.1f} Hz")
```

## Workflow Summary

```
Lab Recording
     │
     ▼
SpikeInterface SortingExtractor
     │
     ├──► spike_trains_to_bitstreams() ──► SCDenseLayer.forward()
     │
     ├──► spike_trains_to_population_input() ──► Network.run()
     │
     └──► firing_rates_to_sc_probs() ──► VectorizedSCLayer.forward()
```

## Further Reading

- [Tutorial 23: Spike Train Analysis](23_spike_train_analysis.md) — 125 analysis functions
- [Tutorial 31: Network Simulation Engine](31_network_simulation_engine.md) — Population/Projection/Network
- [API: Interfaces](../api/interfaces.md) — SpikeInterface adapter API
- [SpikeInterface docs](https://spikeinterface.readthedocs.io/) — upstream documentation
