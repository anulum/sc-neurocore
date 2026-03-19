# SC-NeuroCore

**Universal Stochastic Computing Framework for Neuromorphic Hardware**

SC-NeuroCore provides a complete stack for building, simulating, and deploying
stochastic computing (SC) neural networks — from individual neurons to full
SCPN layer hierarchies, with both software simulation and Verilog RTL for
FPGA deployment.

**Version 3.13.2** | 2 112 passing Python tests + 336 Rust tests | 100% Coverage | 122 Neuron Models | 111-Model NetworkRunner | [PyPI](https://pypi.org/project/sc-neurocore/) | [GitHub](https://github.com/anulum/sc-neurocore)

![SC-NeuroCore train-to-hardware pipeline](assets/pipeline.png)
*Train in PyTorch → Quantise to Q8.8 → Simulate with stochastic bitstreams → Compile to SystemVerilog → Synthesise for FPGA. The Rust SIMD engine accelerates all stages.*

## Key Features

- **122 neuron models** — McCulloch-Pitts (1943) through ArcaneNeuron (2026), 9 hardware chip emulators, 9 AI-optimized
- **111 Rust neuron models** — PyO3 bindings, 111-model NetworkRunner with Rayon parallelism
- **ArcaneNeuron** — flagship self-referential cognition model with 5 coupled subsystems (fast/working/deep/gate/predictor)
- **Identity substrate** — persistent spiking network with checkpointing, trace encoding/decoding, L16 Director control
- **Network simulation** — Population-Projection-Network with 3 backends (Python, Rust, MPI)
- **MPI distributed** — billion-neuron scale via mpi4py
- **Model zoo** — 10 pre-built configs, 3 pre-trained weight sets (MNIST, SHD, DVS)
- **126-function analysis toolkit** — spike train stats, distance, correlation, causality, decoding (23 modules)
- **12 visualization plots** — raster, voltage, ISI, PSD, cross-correlogram, and more
- **7 advanced plasticity rules** — BPTT, e-prop, R-STDP, MAML, homeostatic, STP, structural
- **Packed bitwise layers** — 64-bit vectorised AND/popcount for high throughput
- **Rust SIMD engine** — 512x real-time, AVX-512/AVX2/NEON/SVE/RVV dispatch
- **GPU acceleration** — CuPy backend + JAX JIT training + CuPy sparse GPU
- **SNN training** — Surrogate gradient training (ATan, FastSigmoid, SuperSpike) with `to_sc_weights()` bridge
- **SCPN layer stack** — 16-layer holonomic model (L1 Quantum → L16 Meta) with JAX acceleration
- **Verilog RTL** — 17 synthesisable modules, 7 formal verification files (64 properties), bit-exact co-simulation
- **HDC/VSA** — Hyper-dimensional computing for symbolic AI workloads
- **conda-forge recipe** — ready for conda-forge distribution

The default `pip install sc-neurocore` wheel ships the public
core/simulation/domain-bridge package surface under the `sc-neurocore`
product name. Frontier modules such as `analysis`, `viz`, `audio`,
`dashboard`, and `swarm` remain source-checkout features.

## Quick Start

```bash
pip install sc-neurocore
```

This installs the public `sc-neurocore` package from PyPI. The optional Rust
engine remains available from source builds and release assets.

```python
from sc_neurocore import VectorizedSCLayer, BitstreamEncoder

layer = VectorizedSCLayer(n_inputs=8, n_neurons=4, length=1024)
output = layer.forward([0.3, 0.5, 0.7, 0.2, 0.8, 0.1, 0.6, 0.4])
print(output)  # array of firing-rate probabilities
```

## Architecture

| Tier | Modules | Ships in wheel |
|------|---------|:--------------:|
| **Core** | neurons, synapses, layers, sources, utils, recorders, accel, compiler, hdl_gen, hardware | Yes |
| **Simulation** | hdc, solvers, transformers, learning, graphs, ensembles, export, pipeline, training | Yes |
| **Domain bridges** | quantum (Qiskit/PennyLane), adapters/holonomic (JAX), scpn (Petri nets) | Yes |
| **Research** | robotics, physics, bio, optics, chaos, sleep, interfaces | Source only |
| **Frontier** | analysis, viz, audio, dashboard, generative, world_model, swarm | Source only |

See [Architecture](architecture/architecture.md) for the full package map.

## Tutorials

| Tutorial | Topic |
|----------|-------|
| [SC Fundamentals](tutorials/01_stochastic_computing_fundamentals.md) | Bitstream encoding, arithmetic, noise analysis |
| [Building Your First SNN](tutorials/02_building_your_first_snn.md) | Neurons, synapses, layers, simulation |
| [Surrogate Gradient Training](tutorials/03_surrogate_gradient_training.md) | Train SNNs with backpropagation |
| [Hyper-Dimensional Computing](tutorials/04_hyperdimensional_computing.md) | Symbolic AI with high-dimensional vectors |
| [FPGA in 20 Minutes](tutorials/fpga_in_20_minutes.md) | Train → quantise → synthesise → deploy |
| [Rust Engine & Performance](tutorials/05_rust_engine_performance.md) | SIMD tiers, GPU, benchmarking |
| [Brunel Network Translation](tutorials/06_brunel_network_translation.md) | Brian2 → SC conversion workflow |

## Documentation

- **[Getting Started](guides/getting-started.md)** — Installation and first steps
- **[API Reference](api/API_REFERENCE.md)** — Python package API
- **[Rust Engine API](api/rust-engine.md)** — High-performance Rust engine docs
- **[Hardware Guide](hardware/HARDWARE_GUIDE.md)** — FPGA deployment workflow
- **[Benchmarks](benchmarks/BENCHMARKS.md)** — Performance measurements
- **[For Research Labs](guides/FOR_RESEARCH_LABS.md)** — Setup guide for neuroscience, hardware, and ML labs
- **[Pricing](pricing.md)** — Free for research, commercial licenses available

## Demo

<!-- TODO: Replace with YouTube embed or hosted video when recorded -->
See the [Neuron Explorer Notebook](https://github.com/anulum/sc-neurocore/blob/main/notebooks/04_neuron_explorer.ipynb)
for an interactive walkthrough of all 117 neuron models with voltage traces,
phase portraits, and F-I curves. Or try the
[Quickstart on Google Colab](https://colab.research.google.com/github/anulum/sc-neurocore/blob/main/notebooks/quickstart_colab.ipynb)
— no installation required.

## Community & Ecosystem

SC-NeuroCore integrates with the [NIR](https://neuroir.org/) (Neuromorphic Intermediate Representation)
ecosystem, connecting to Norse, snnTorch, Lava-DL, and hardware targets including
BrainScaleS-2, Loihi, and SpiNNaker2. SC-NeuroCore adds the missing FPGA deployment
backend via bit-true Verilog co-simulation.

**Contact:** [neurocore@anulum.li](mailto:neurocore@anulum.li) |
[GitHub Discussions](https://github.com/anulum/sc-neurocore/discussions) |
[www.anulum.li](https://www.anulum.li)
