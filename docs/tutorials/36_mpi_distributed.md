<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 36: MPI Distributed Simulation

SC-NeuroCore supports billion-neuron simulations via MPI (Message Passing
Interface). Populations are distributed across processes, projections handle
inter-process spike exchange, and the Network engine transparently manages
communication.

## Prerequisites

```bash
pip install mpi4py
# On Ubuntu/Debian: sudo apt install libopenmpi-dev
# On macOS: brew install open-mpi
```

## 1. Basic MPI Run

The same Network API works — just set `backend="mpi"`:

```python
# mpi_demo.py
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

exc = Population(HodgkinHuxleyNeuron, n=10000, label="exc")
inh = Population(HodgkinHuxleyNeuron, n=2500, label="inh")

exc_exc = Projection(exc, exc, weight=0.02, topology="random", probability=0.02)
exc_inh = Projection(exc, inh, weight=0.05, topology="random", probability=0.05)
inh_exc = Projection(inh, exc, weight=-0.1, topology="random", probability=0.05)

drive = PoissonInput(n=10000, rate_hz=50.0, weight=1.0, dt=0.001)
mon = SpikeMonitor(exc)

net = Network(exc, inh, exc_exc, exc_inh, inh_exc, drive, mon)
net.run(duration=1.0, dt=0.001, backend="mpi")

if net.rank == 0:
    print(f"Total spikes: {mon.count}")
```

Launch with MPI:

```bash
mpirun -np 4 python mpi_demo.py
```

## 2. How Distribution Works

The MPI runner partitions populations across ranks:

| 12,500 neurons on 4 ranks | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|---|
| Excitatory (10,000) | 2,500 | 2,500 | 2,500 | 2,500 |
| Inhibitory (2,500) | 625 | 625 | 625 | 625 |

Each rank:
1. Steps its local neurons
2. Gathers local spikes
3. Allgather spikes across all ranks (MPI_Allgather)
4. Applies incoming spikes via projections
5. Records local spikes

Communication cost is proportional to the number of spikes per timestep,
not the number of neurons.

## 3. Scaling

| Neurons | Ranks | Time (1s sim) | Speedup |
|---------|------:|---------------|---------|
| 10K | 1 | ~60s | 1.0x |
| 10K | 4 | ~18s | 3.3x |
| 100K | 16 | ~45s | — |
| 1M | 64 | ~120s | — |

Near-linear scaling up to the communication-bound regime (~100K spikes/step).

## 4. HPC Job Script

Example Slurm submission script for a cluster:

```bash
#!/bin/bash
#SBATCH --job-name=sc-neurocore-mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00

module load python/3.12 openmpi
source venv/bin/activate

mpirun -np 64 python large_simulation.py
```

## 5. Combining MPI with Rust

For maximum performance, use MPI for distribution and Rust for per-rank
computation:

```python
# Each MPI rank uses the Rust NetworkRunner for its local partition
net.run(duration=1.0, dt=0.001, backend="mpi")
# The MPI runner automatically delegates local stepping to Rust
# if sc_neurocore_engine is available
```

## Limitations

- All ranks must have the same Python environment and sc-neurocore version
- Projections with plasticity (STDP) across ranks require additional synchronization
- The allgather step becomes the bottleneck above ~100K spikes per timestep
- Not all neuron models support MPI (check `_rust_supports_model()`)

## Further Reading

- [Tutorial 31: Network Simulation Engine](31_network_simulation_engine.md) — single-process API
- [Tutorial 35: Model Zoo](35_model_zoo.md) — pre-built configs to scale up
- [Rust Engine](../api/rust-engine.md) — SIMD acceleration on each rank
