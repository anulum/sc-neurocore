# SC-NeuroCore

**Universal Stochastic Computing Framework for Neuromorphic Hardware**

SC-NeuroCore provides a complete stack for building, simulating, and deploying
stochastic computing (SC) neural networks — from individual neurons to full
SCPN layer hierarchies, with both software simulation and Verilog RTL for
FPGA deployment.

## Key Features

- **Stochastic neurons** — LIF, Izhikevich, dendritic, homeostatic variants
- **Packed bitwise layers** — 64-bit vectorised AND/popcount for high throughput
- **GPU acceleration** — CuPy backend with automatic CPU fallback
- **SCPN layer stack** — 7-layer consciousness model (L1 Quantum → L7 Symbolic)
- **Verilog RTL** — Synthesisable hardware for FPGA deployment
- **Tiered modules** — Core (production), Research (experimental), Contrib (speculative)

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov
```

```python
from sc_neurocore import VectorizedSCLayer, BitstreamEncoder

layer = VectorizedSCLayer(n_inputs=8, n_neurons=4, length=1024)
output = layer.forward([0.3, 0.5, 0.7, 0.2, 0.8, 0.1, 0.6, 0.4])
print(output)  # array of firing-rate probabilities
```

## Architecture

SC-NeuroCore uses a three-tier module system:

| Tier | Purpose | Packages |
|------|---------|----------|
| **Core** | Production-ready | neurons, synapses, layers, sources, recorders, utils, accel |
| **Research** | Experimental | analysis, bio, core, dashboard, generative, learning, pipeline, scpn, ... |
| **Contrib** | Speculative | eschaton, exotic, meta, post_silicon, transcendent |

See [Architecture](architecture.md) for the full package map.
