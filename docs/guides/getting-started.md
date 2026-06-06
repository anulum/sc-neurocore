# Getting Started

This guide gets a new user from installation to the first useful SC-NeuroCore workflows. If you are evaluating the project rather than coding immediately, first read [Product Overview](../product_overview.md) and [Applications and Market](../applications_and_market.md).

The base install is intentionally small. Optional training, hardware, quantum, JAX, GPU, Studio, and research surfaces are installed only when the relevant workflow needs them.

## Choose your first route

| Goal | Start with | Why |
| --- | --- | --- |
| Learn the basic stochastic-computing idea | Bitstream encoder and `VectorizedSCLayer` examples below | They show probability-as-bitstream without optional dependencies. |
| Evaluate SNN modelling | Single-neuron, model-zoo, and network examples | They expose neuron dynamics before hardware-specific concerns. |
| Evaluate hardware potential | [FPGA in 20 Minutes](../tutorials/fpga_in_20_minutes.md) after this page | It connects quantisation, SC simulation, RTL export, and report boundaries. |
| Evaluate market or industrial fit | [Applications and Market](../applications_and_market.md) and [Industrial Applications](../api/industrial_applications.md) | They map use cases to evidence categories and deployment gaps. |
| Review evidence notebooks | [Notebook Guide](notebook_guide.md) | It separates onboarding notebooks from readiness and evidence notebooks. |

## Installation

```bash
# From PyPI
pip install sc-neurocore

# From source (editable Python package)
pip install -e .

# With development tools
pip install -e ".[dev]"

# Training with PyTorch-backed cells
pip install -e ".[training]"

# NIR interop (Norse, snnTorch, Lava-DL)
pip install -e ".[nir]"

# Research plotting/export stack
pip install -e ".[research]"

# GPU acceleration experiments (CuPy)
pip install -e ".[gpu]"

# Local full research environment only
pip install -e ".[full]"
```

`pip install sc-neurocore` installs the public Python package. If you are
editing the Rust bridge locally, install `bridge/` in the same environment or
run source-tree commands with `PYTHONPATH=src:bridge`. For the complete
dependency boundary, see [Install Profiles](install_profiles.md).

## Requirements

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- PyTorch (optional, `pip install sc-neurocore[training]`)
- Numba (optional, `pip install sc-neurocore[accel]`)
- Matplotlib (optional, `pip install sc-neurocore[research]`)
- Hardware toolchains such as Vivado, Quartus, and Yosys are external and only
  needed for synthesis or implementation, not for the base package.

## Running Tests

```bash
# Full repository suite (long-running)
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

### 6. ArcaneNeuron — Self-Referential Cognition

```python
from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron

neuron = ArcaneNeuron()
for t in range(500):
    spike = neuron.step(current=0.8 if t < 200 else 0.1)

state = neuron.get_state()
print(f"Identity (deep compartment): {state['v_deep']:.4f}")
print(f"Confidence: {state['confidence']:.3f}")
print(f"Meta-learning rate: {state['meta_lr']:.4f}")
```

### 7. Identity Substrate

```python
from sc_neurocore.identity import IdentitySubstrate, Checkpoint, StateDecoder

substrate = IdentitySubstrate(n_cortical=200, n_inhibitory=80, n_memory=40)
substrate.inject_experience("The cat sat on the mat. It was a rainy day.")
substrate.run(duration=0.5)

decoder = StateDecoder(substrate)
print(decoder.generate_priming_context())

Checkpoint.save(substrate, "identity_checkpoint.npz")
restored = Checkpoint.load("identity_checkpoint.npz")
```

### 8. Model Zoo

```python
from sc_neurocore.model_zoo import brunel_balanced_network, load_pretrained

net = brunel_balanced_network()
net.run(0.1)

classifier = load_pretrained("mnist")
```

10 configurations + 3 pre-trained weight sets (MNIST, SHD, DVS gesture).

### 9. Source-Only Visualization Helpers

```python
from sc_neurocore.viz.plots import raster_plot, voltage_trace, firing_rate_plot

# After running a network with monitors:
raster_plot(spike_monitor)
voltage_trace(state_monitor, neuron_ids=[0, 1, 2])
firing_rate_plot(spike_monitor, bin_ms=10)
```

`sc_neurocore.viz` is excluded from the default wheel. Use a source checkout
(`pip install -e ".[dev]"`) before importing it.

### 10. Source-Only Hardware Deployment (PYNQ)

```python
from sc_neurocore.drivers.sc_neurocore_driver import SC_NeuroCore_Driver

driver = SC_NeuroCore_Driver(mode="EMULATION")
result = driver.run_step(input_vector=[0.5, 0.8, 0.1])
```

For physical FPGA deployment on PYNQ-Z1/Z2, synthesize the bitstream
using the [FPGA Toolchain Guide](../hardware/FPGA_TOOLCHAIN_GUIDE.md),
then switch to `mode="HARDWARE"`.

`sc_neurocore.drivers` is excluded from the default wheel. Use a source checkout
before importing it.

## Dense Path Selection

- Single-sample and tiny batches (1-4): use `DenseLayer.forward_fast`
- Larger batches (>=10): use `DenseLayer.forward_batch_numpy`
- Default `DenseLayer.forward` auto-selects

## Analysis Toolkit

The analysis toolkit lives in the source distribution, not the default wheel.
Install from a source checkout before importing `sc_neurocore.analysis`.

SC-NeuroCore includes 132 spike train analysis functions across 24 modules --
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
and [API reference](../api/analysis.md) for the full 126-function listing.

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

Three backends: Python (NumPy), Rust (NetworkRunner with 81 models, Rayon parallel),
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
