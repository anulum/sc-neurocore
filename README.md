CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
Contact us: www.anulum.li  protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AFFERO GENERAL PUBLIC LICENSE v3
Commercial Licensing: Available

# SC-NeuroCore

[![CI](https://github.com/anulum/sc-neurocore/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/sc-neurocore/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-3.7.0-blue)](https://github.com/anulum/sc-neurocore/releases)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/engine-Rust-orange)](https://www.rust-lang.org/)

**Version:** 3.7.0
**Status:** Production Core Verified | 976 Tests Passing | CI/CD Active

SC-NeuroCore is a universal stochastic computing framework for neuromorphic
hardware simulation. It provides bit-true Python models that match Verilog RTL
cycle-exactly, a high-performance Rust engine (512x real-time), GPU-accelerated
inference, and a tiered module system spanning production hardware to
theoretical research.

## Quick Start

```bash
# Clone
git clone https://github.com/anulum/sc-neurocore.git
cd sc-neurocore

# Install core package
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/benchmark_suite.py

# GPU acceleration (requires CUDA)
pip install -e ".[gpu]"
```

## Performance Routing

Use explicit path selection for dense inference to avoid small-batch regressions:

- Single sample or micro-batch (1-4 samples): call `DenseLayer.forward_fast(...)`.
- Medium/large batch (>=10 samples): call `DenseLayer.forward_batch_numpy(...)`.
- Validation/reference path: use `DenseLayer.forward(...)` and compare to fast paths in tests.

For benchmark reports, always include batch size, bitstream length, seed policy, and CPU SIMD tier.

## Architecture

### Module Tiers

| Tier | Location | Description |
|------|----------|-------------|
| **core** | `src/sc_neurocore/` (neurons, synapses, layers, sources, utils, recorders, accel) | Production-ready. Imported by default. |
| **research** | `src/sc_neurocore/` (hdc, solvers, transformers, quantum, robotics, bio, physics, +18 more) | Functional but experimental. Import explicitly. |
| **speculative** | `research/` (eschaton, exotic, meta, post_silicon, transcendent) | Theoretical explorations. Not part of the installable package. See `research/README.md`. |

### Core API (28 symbols)

```python
from sc_neurocore import (
    # Neurons
    StochasticLIFNeuron, FixedPointLIFNeuron, FixedPointLFSR,
    FixedPointBitstreamEncoder, HomeostaticLIFNeuron,
    StochasticDendriticNeuron, SCIzhikevichNeuron,
    # Synapses
    BitstreamSynapse, BitstreamDotProduct,
    StochasticSTDPSynapse, RewardModulatedSTDPSynapse,
    # Layers
    SCDenseLayer, SCConv2DLayer, SCLearningLayer,
    VectorizedSCLayer, SCRecurrentLayer, MemristiveDenseLayer,
    SCFusionLayer, StochasticAttention,
    # Utilities
    BitstreamEncoder, BitstreamAverager, RNG,
    generate_bernoulli_bitstream, generate_sobol_bitstream,
    bitstream_to_probability,
    # Sources & Recorders
    BitstreamCurrentSource, BitstreamSpikeRecorder,
)
```

### Hardware (Verilog RTL)

```
hdl/
  sc_bitstream_encoder.v   -- LFSR-based stochastic encoder (SEED_INIT param)
  sc_bitstream_synapse.v   -- AND-gate SC multiplier
  sc_dotproduct_to_current.v -- Popcount -> fixed-point current
  sc_lif_neuron.v          -- Q8.8 leaky integrate-and-fire
  sc_firing_rate_bank.v    -- Spike rate estimator
  sc_dense_layer_core.v    -- Full dense layer pipeline (decorrelated seeds)
  sc_neurocore_top.v       -- AXI-Lite configuration wrapper
  sc_axil_cfg.v            -- AXI-Lite register file
  tb_sc_lif_neuron.v       -- Co-simulation testbench
```

### GPU Acceleration

```python
from sc_neurocore.accel import xp, HAS_CUPY, to_device, to_host
from sc_neurocore.accel.gpu_backend import gpu_vec_mac

# VectorizedSCLayer auto-detects GPU
layer = VectorizedSCLayer(n_inputs=32, n_neurons=64, length=1024)
output = layer.forward(input_values)  # GPU if CuPy available, else CPU
```

## Hardware-Software Co-Simulation

The co-sim flow verifies bit-exact equivalence between the Python model and
Verilog RTL:

```bash
# 1. Generate stimuli + expected results (Python golden model)
python scripts/cosim_gen_and_check.py --generate

# 2. Run Verilog simulation (requires Icarus Verilog)
iverilog -o tb_lif hdl/sc_lif_neuron.v hdl/tb_sc_lif_neuron.v
vvp tb_lif

# 3. Compare results
python scripts/cosim_gen_and_check.py --check
```

### Key Technical Details

- **LFSR**: 16-bit maximal-length, polynomial x^16+x^14+x^13+x^11+1, period 65535
- **Seed strategy**: Input encoders `0xACE1 + i*7`, weight encoders `0xBEEF + i*13`
- **Fixed-point**: Q8.8 (DATA_WIDTH=16, FRACTION=8), signed two's complement
- **Overflow**: Explicit bit-width masking via `_mask()` function

## Examples

Runnable scripts in `examples/`:

| Script | Description |
|--------|-------------|
| `01_basic_sc_encoding.py` | Bernoulli & Sobol bitstream encoding/decoding |
| `02_sc_neuron_layer.py` | SCDenseLayer construction and forward pass |
| `03_ir_compile_demo.py` | IR graph building, verification, SystemVerilog emission (v3 Rust engine) |
| `04_vectorized_layer.py` | VectorizedSCLayer throughput benchmarking |
| `05_scpn_stack.py` | Full 7-layer SCPN consciousness stack with inter-layer coupling |
| `06_hdl_generation.py` | Verilog top-level generation from a network description |
| `07_ensemble_consensus.py` | Multi-agent ensemble orchestration and voting |
| `08_hdc_symbolic_query.py` | Hyper-Dimensional Computing symbolic memory (v3 Rust engine) |
| `09_safety_critical_logic.py` | Fault-tolerant Boolean logic with stochastic redundancy (v3 Rust engine) |
| `10_benchmark_report.py` | Head-to-head v2/v3 benchmark suite (v3 Rust engine) |
| `11_sc_training_demo.py` | Surrogate-gradient training of an SC dense layer (v3 Rust engine) |

```bash
PYTHONPATH=src:bridge python examples/01_basic_sc_encoding.py
```

Examples marked **(v3 Rust engine)** require the compiled `sc_neurocore_engine` wheel.
All other examples run with the pure-Python `sc_neurocore` package.

## CI/CD

GitHub Actions pipelines (`.github/workflows/`):
- **ci.yml**: Lint (black + mypy) + Test (Python 3.9 / 3.11 / 3.12 matrix, coverage >= 97%) + Build
- **v3-engine.yml**: Rust engine build + `cargo test`
- **v3-wheels.yml**: Cross-platform wheel builds (Linux, macOS, Windows)
- **docs.yml**: MkDocs documentation build

## Benchmarks

Run the benchmark suite:

```bash
python benchmarks/benchmark_suite.py           # quick mode
python benchmarks/benchmark_suite.py --full    # thorough (10x)
python benchmarks/benchmark_suite.py --markdown # output BENCHMARKS.md
```

Sample results (CPU, quick mode):

| Operation | Throughput |
|-----------|-----------|
| LFSR step | 2.25 Mstep/s |
| Bitstream encoder | 1.88 Mstep/s |
| LIF neuron step | 1.15 Mstep/s |
| vec_and (1024 words) | 45.67 Gbit/s |
| gpu_vec_mac (64x32x16w) | 6.15 GOP/s |

## Documentation

- [CHANGELOG.md](CHANGELOG.md) -- Version history
- [docs/](docs/) -- Full documentation (guides, API reference, benchmarks, research)
- [docs/guides/getting-started.md](docs/guides/getting-started.md) -- Installation & quickstart
- [docs/architecture/](docs/architecture/) -- Package architecture
- [docs/hardware/](docs/hardware/) -- FPGA deployment workflow

Build API docs locally:
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

## Install Extras

```bash
pip install -e ".[dev]"       # pytest, mypy, black
pip install -e ".[gpu]"       # CuPy CUDA acceleration
pip install -e ".[research]"  # networkx, onnx, torch
pip install -e ".[full]"      # networkx, onnx
```

## License

SC-NeuroCore is dual-licensed:

- **Open Source**: [GNU Affero General Public License v3.0](LICENSE) (AGPLv3)
- **Commercial**: Proprietary license available for integration into closed-source products

For commercial licensing enquiries, contact [protoscience@anulum.li](mailto:protoscience@anulum.li).
