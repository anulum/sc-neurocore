<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 31: Network Simulation Engine

SC-NeuroCore's network engine provides a Population-Projection-Network architecture
with three backends (Python/NumPy, Rust NetworkRunner, MPI distributed). Build networks
from the Python model catalogue or the 161-model Rust NetworkRunner dispatch
list, connect them with configurable topologies, inject stimuli, record spikes,
and run with automatic backend selection.

## Core Concepts

| Component | Class | Purpose |
|-----------|-------|---------|
| **Population** | `Population` | A group of identical neurons (e.g. 100 LIF neurons) |
| **Projection** | `Projection` | Weighted connections between two populations |
| **Network** | `Network` | Container that runs populations + projections together |
| **SpikeMonitor** | `SpikeMonitor` | Records spike times from a population |
| **Stimulus** | `PoissonInput`, `StepCurrent`, `TimedArray` | Drives input current into a population |

## 1. First Network: Two Populations

```python
import numpy as np
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

# 80 excitatory HH neurons, 20 inhibitory HH neurons
exc = Population(HodgkinHuxleyNeuron, n=80, label="exc")
inh = Population(HodgkinHuxleyNeuron, n=20, label="inh")

# Connect them
exc_to_exc = Projection(exc, exc, weight=0.05, topology="random", probability=0.1)
exc_to_inh = Projection(exc, inh, weight=0.1, topology="random", probability=0.3)
inh_to_exc = Projection(inh, exc, weight=-0.15, topology="random", probability=0.3)

# Drive excitatory neurons with Poisson input
drive = PoissonInput(n=80, rate_hz=100.0, weight=2.0, dt=0.001)

# Record spikes
mon_exc = SpikeMonitor(exc)
mon_inh = SpikeMonitor(inh)

# Build and run
net = Network(exc, inh, exc_to_exc, exc_to_inh, inh_to_exc, drive, mon_exc, mon_inh)
net.run(duration=0.5, dt=0.001)  # 500 ms

print(f"Excitatory spikes: {mon_exc.count}")
print(f"Inhibitory spikes: {mon_inh.count}")
```

## 2. Topology Generators

Six built-in connectivity patterns:

```python
from sc_neurocore.network.topology import (
    random_connectivity,
    small_world,
    scale_free,
    ring_topology,
    grid_topology,
    all_to_all,
)

# Random (Erdos-Renyi) — works directly as string
proj = Projection(exc, inh, weight=0.1, topology="random", probability=0.2)

# All-to-all — works directly as string
proj = Projection(exc, inh, weight=0.01, topology="all_to_all")

# Small-world, scale-free, ring, grid — build CSR tuple first, then pass
sw_csr = small_world(n=80, k=4, p_rewire=0.1, weight=0.05, seed=42)
proj = Projection(exc, exc, weight=0.05, topology=sw_csr)

sf_csr = scale_free(n=80, m=3, weight=0.05, seed=42)
proj = Projection(exc, exc, weight=0.05, topology=sf_csr)

ring_csr = ring_topology(n=80, k=4, weight=0.05)
proj = Projection(exc, exc, weight=0.05, topology=ring_csr)

grid_csr = grid_topology(rows_count=10, cols_count=8, radius=2, weight=0.05)
proj = Projection(exc, exc, weight=0.05, topology=grid_csr)
```

## 3. Stimulus Types

Three input sources drive current into populations:

```python
from sc_neurocore.network.stimulus import PoissonInput, StepCurrent, TimedArray

# Poisson spike train input
poisson = PoissonInput(n=80, rate_hz=50.0, weight=1.5, dt=0.001, seed=42)

# Step current (on at 100ms, off at 400ms)
step = StepCurrent(onset=0.1, offset=0.4, amplitude=5.0)

# Arbitrary time-varying current
values = np.sin(np.linspace(0, 4*np.pi, 500)) * 3.0
timed = TimedArray(values, dt=0.001)
```

## 4. STDP Plasticity

Projections can learn via spike-timing dependent plasticity:

```python
# Run network
net.run(duration=1.0, dt=0.001)

# STDP is applied per-timestep during run() if projection has plasticity="stdp".
# For manual STDP after simulation, pass binary spike arrays (one timestep):
import numpy as np
src_spikes = np.random.randint(0, 2, size=exc.n)  # example binary array
tgt_spikes = np.random.randint(0, 2, size=exc.n)
exc_to_exc.update_plasticity(
    src_spikes=src_spikes,
    tgt_spikes=tgt_spikes,
    a_plus=0.01,
    a_minus=0.012,
    tau=20.0,
)
```

## 5. Three Backends

The `Network.run()` method auto-selects the fastest available backend:

```python
# Auto-select (Rust if available, else Python)
net.run(duration=0.5, dt=0.001, backend="auto")

# Force Python/NumPy
net.run(duration=0.5, dt=0.001, backend="python")

# Force Rust NetworkRunner (111 models, Rayon-parallel, 100K+ neurons)
net.run(duration=0.5, dt=0.001, backend="rust")

# MPI distributed (requires mpi4py, launch with mpirun)
net.run(duration=0.5, dt=0.001, backend="mpi")
```

| Backend | Scale | When to use |
|---------|-------|-------------|
| Python | < 1K neurons | Development, debugging, any neuron model |
| Rust | 1K-100K neurons | Production, 111 supported models, near-linear scaling |
| MPI | 100K+ neurons | Multi-node HPC, billion-neuron simulations |

## 6. Using the Model Zoo

Pre-built network configurations with optional pre-trained weights:

```python
from sc_neurocore.model_zoo.configs import brunel_balanced_network
from sc_neurocore.model_zoo.pretrained import load_pretrained

# Create a pre-built network configuration
net = brunel_balanced_network(n_exc=800, n_inh=200)
net.run(duration=1.0, dt=0.001)

# Or load with pre-trained weights
net = load_pretrained("mnist")
```

## 7. Spike Analysis

After simulation, use the 127-function analysis toolkit:

```python
from sc_neurocore.analysis import (
    firing_rate,
    coefficient_of_variation,
    spike_train_correlation,
    victor_purpura_distance,
)

trains = mon_exc.spike_trains
rates = {i: firing_rate(t, duration=0.5) for i, t in trains.items()}
print(f"Mean rate: {np.mean(list(rates.values())):.1f} Hz")
```

## Further Reading

- [Tutorial 06: Brunel Network](06_brunel_network_translation.md) — Brian2 comparison
- [Tutorial 08: Online Learning (STDP)](08_online_learning_stdp.md) — plasticity
- [Tutorial 23: Spike Train Analysis](23_spike_train_analysis.md) — full analysis toolkit
- [API: Network](../api/models.md) — auto-generated API docs
