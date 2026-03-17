# Getting Started

## Installation

```bash
# From PyPI
pip install sc-neurocore

# From source (editable)
pip install -e .

# With development tools
pip install -e ".[dev]"

# With GPU acceleration (CuPy)
pip install -e ".[gpu]"

# Full research stack
pip install -e ".[research]"
```

## Requirements

- Python >= 3.10
- NumPy >= 1.22
- SciPy >= 1.7
- Numba (optional, `pip install sc-neurocore[accel]`)
- Matplotlib (optional, `pip install sc-neurocore[full]`)

## Running Tests

```bash
# Full suite (2 055 tests, 100% coverage gate)
pytest tests/ -v --cov=sc_neurocore --cov-report=term

# Quick smoke test
pytest tests/test_integration.py -v
```

## First Steps

### 1. Create a Bitstream Encoder

```python
from sc_neurocore import BitstreamEncoder, bitstream_to_probability

encoder = BitstreamEncoder(x_min=0.0, x_max=1.0, length=1024)
bitstream = encoder.encode(0.7)
recovered = bitstream_to_probability(bitstream)
print(f"Encoded 0.7 -> recovered {recovered:.3f}")
```

### 2. Build a Neuron Layer

```python
from sc_neurocore import VectorizedSCLayer

layer = VectorizedSCLayer(n_inputs=4, n_neurons=2, length=512)
output = layer.forward([0.3, 0.7, 0.5, 0.2])
print(f"Output firing rates: {output}")
```

### 3. Single Neuron with Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sc_neurocore import StochasticLIFNeuron

neuron = StochasticLIFNeuron(tau_mem=10.0, dt=1.0, v_threshold=1.0, noise_std=0.05)

potentials, spikes = [], []
for t in range(100):
    spike = neuron.step(0.15)
    potentials.append(neuron.v)
    spikes.append(spike)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(potentials)
ax1.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
ax1.set_ylabel('Voltage')
ax1.legend()
ax2.stem(spikes, linefmt='g-', markerfmt='go', basefmt=' ')
ax2.set_ylabel('Spike')
ax2.set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig('quickstart_output.png')
```

### 4. Stochastic Network (Source → Synapse → Neuron)

```python
from sc_neurocore.sources.bitstream_current_source import BitstreamCurrentSource
from sc_neurocore import StochasticLIFNeuron

source = BitstreamCurrentSource(
    x_inputs=[0.8, 0.5, 0.2],
    weight_values=[1.0, 0.5, 0.0],
    x_min=0.0, x_max=1.0,
    w_min=0.0, w_max=1.0,
)
neuron = StochasticLIFNeuron()

for t in range(100):
    current = source.step()
    neuron.step(current)
```

### 5. Run the Full SCPN Stack

```python
from sc_neurocore.scpn import create_full_stack, run_integrated_step, get_global_metrics

stack = create_full_stack()
outputs = run_integrated_step(stack, dt=0.01)
metrics = get_global_metrics(stack)
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
```

### 6. Hardware Deployment (PYNQ)

```python
from sc_neurocore.drivers.sc_neurocore_driver import SC_NeuroCore_Driver

driver = SC_NeuroCore_Driver(mode="EMULATION")
result = driver.run_step(input_vector=[0.5, 0.8, 0.1])
```

For physical FPGA deployment on PYNQ-Z1/Z2, synthesize the bitstream
using the [FPGA Toolchain Guide](../hardware/FPGA_TOOLCHAIN_GUIDE.md),
then switch to `mode="HARDWARE"`.

## Dense Path Selection

- Single-sample and tiny batches (1-4): use `DenseLayer.forward_fast`
- Larger batches (>=10): use `DenseLayer.forward_batch_numpy`
- Default `DenseLayer.forward` auto-selects

## Analysis Toolkit

SC-NeuroCore includes 125 spike train analysis functions across 23 modules --
statistics, variability, rate estimation, distance metrics, correlation,
spectral, temporal, stimulus, LFP coupling, surrogates, information theory,
causality, dimensionality reduction, decoding, network, point process,
sorting quality, waveform, statistics, patterns, SPADE, GPFA, and
explainability. Pure NumPy, zero external dependencies.

```python
from sc_neurocore.analysis import (
    firing_rate, cv_isi, fano_factor, cross_correlation,
    victor_purpura_distance, phase_locking_value, gpfa, spade_detect,
)
```

See the [Spike Train Analysis tutorial](../tutorials/23_spike_train_analysis.md)
and [API reference](../api/analysis.md) for the full 125-function listing.

## Network Simulation

Build multi-population networks with the Population-Projection-Network engine:

```python
from sc_neurocore.network import Population, Projection, Network

exc = Population("Izhikevich", 800)
inh = Population("Izhikevich", 200)
net = Network()
net.add_population(exc)
net.add_population(inh)
net.add_projection(Projection(exc, inh, probability=0.1, weight=5.0))
net.add_projection(Projection(inh, exc, probability=0.1, weight=-4.0))
net.run(duration_ms=1000, dt=0.1)
```

Three backends: Python (NumPy), Rust (NetworkRunner, 64 models, Rayon parallel),
and MPI (billion-neuron distributed via mpi4py).

## Loading Pre-Trained Models

The model zoo provides 10 ready-to-run network configurations and 3 pre-trained
weight sets:

```python
from sc_neurocore.model_zoo.configs import brunel_balanced_network, mnist_classifier

# Pre-configured Brunel balanced network
net = brunel_balanced_network()

# MNIST digit classifier with pre-trained weights
classifier = mnist_classifier()
```

Available weight sets: MNIST (784-128-10), SHD speech (700-256-20),
DVS gesture (256-256-11).

## MPI Distributed Simulation

For billion-neuron scale runs across multiple nodes:

```bash
pip install sc-neurocore[mpi]
mpirun -np 4 python my_large_network.py
```

```python
from sc_neurocore.network import Network

net = Network(backend="mpi")
# Populations are automatically distributed across MPI ranks
```

## What's Next?

- [Learning Path](../LEARNING_PATH.md) — 8-level progression from beginner to FPGA deployment
- [SC for Neuroscientists](SC_FOR_NEUROSCIENTISTS.md) — if you know Brian2/NEST
- [SC for ML Engineers](SC_FOR_ML_ENGINEERS.md) — if you know PyTorch/JAX
- [SC for Hardware Engineers](SC_FOR_HARDWARE_ENGINEERS.md) — if you know Verilog/VHDL
- [Technical Manual](USER_MANUAL.md) — deep dive into SCPN architecture
