# SC-NeuroCore

**Universal Stochastic Computing Framework for Neuromorphic Hardware**

SC-NeuroCore provides a complete stack for building, simulating, and deploying
stochastic computing (SC) neural networks — from individual neurons to full
SCPN layer hierarchies, with both software simulation and Verilog RTL for
FPGA deployment.

**Version 3.12.0** | 1 560 Python + 105 Rust Tests | 100% Coverage | 100% Rust Parity | [PyPI](https://pypi.org/project/sc-neurocore/) | [GitHub](https://github.com/anulum/sc-neurocore)

## Key Features

- **Stochastic neurons** — LIF, Izhikevich, dendritic, homeostatic variants
- **Packed bitwise layers** — 64-bit vectorised AND/popcount for high throughput
- **Rust engine** — SIMD-accelerated backend (512x real-time), PyO3 bindings
- **GPU acceleration** — CuPy backend with automatic CPU fallback
- **SNN training** — Surrogate gradient training (ATan, FastSigmoid, SuperSpike) with `to_sc_weights()` bridge
- **SCPN layer stack** — 16-layer holonomic model (L1 Quantum → L16 Meta) with JAX acceleration
- **Verilog RTL** — 10 synthesisable modules, formal verification, bit-exact Python co-simulation
- **HDC/VSA** — Hyper-dimensional computing for symbolic AI workloads

## Quick Start

```bash
pip install sc-neurocore
```

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
