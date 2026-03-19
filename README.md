© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AFFERO GENERAL PUBLIC LICENSE v3
Commercial Licensing: Available

# SC-NeuroCore

<p align="center">
  <img src="docs/assets/sc_neurocore_header.png" width="1280" alt="SC-NeuroCore — Stochastic Computing & Neuromorphic Engine">
</p>

[![CI](https://github.com/anulum/sc-neurocore/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/sc-neurocore/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-3.13.2-blue)](https://github.com/anulum/sc-neurocore/releases)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/anulum/sc-neurocore)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://anulum.github.io/sc-neurocore/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/engine-Rust-orange)](https://www.rust-lang.org/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12175/badge)](https://www.bestpractices.dev/projects/12175)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/anulum/sc-neurocore/badge)](https://scorecard.dev/viewer/?uri=github.com/anulum/sc-neurocore)
[![REUSE](https://img.shields.io/badge/REUSE-compliant-green)](https://reuse.software/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/sc-neurocore/blob/main/notebooks/quickstart_colab.ipynb)

**Version:** 3.13.2
**Status:** 122 Neuron Models (113 Bio + 9 AI) | 99.49% MNIST | 2 112 Python tests passing + 336 Rust tests | 100% Coverage | 111 Rust Neuron Models | 111-Model NetworkRunner

<p align="center">
  <img src="docs/assets/spike_raster.png" width="800" alt="LIF spike raster — 5 neurons, sinusoidal input">
</p>

SC-NeuroCore is the most comprehensive spiking neural network framework
available. 122 neuron models (113 biophysical + 9 AI-optimized) spanning
82 years of computational neuroscience (McCulloch-Pitts 1943 through
ArcaneNeuron 2026) run inside a deterministic stochastic computing engine
with bit-true Verilog RTL co-simulation, FPGA synthesis via an IR compiler
(SystemVerilog + MLIR/CIRCT backends), formal verification (7 SymbiYosys
modules, 64 properties), a Rust SIMD engine at 512x real-time (111 Rust
neuron models with PyO3 bindings, 111-model NetworkRunner with Rayon-parallel
populations scaling to 100K+ neurons), CuPy GPU acceleration, JAX JIT
training, MPI distributed simulation (billion-neuron scale via mpi4py),
an identity continuity substrate (persistent spiking networks with
checkpointing and L16 Director control), a 126-function spike train
analysis toolkit (23 modules), 12 visualization plots, 7 advanced
plasticity rules, 10 model zoo configurations with 3 pre-trained weight
sets, 9 hardware chip emulators, quantum hybrid computing (Qiskit +
PennyLane), and surrogate gradient training reaching 99.49% MNIST accuracy.
2 112 Python tests across 118+ files and 336 Rust tests hold 100% line
coverage. 13 CI workflows guard every push. conda-forge recipe ready.

## Feature Comparison

| Feature | SC-NeuroCore | snnTorch | Norse | Lava | Brian2 |
|---------|:---:|:---:|:---:|:---:|:---:|
| Stochastic computing (bitstream) | **Yes** | — | — | — | — |
| Bit-true RTL co-simulation | **Yes** | — | — | — | — |
| Verilog / FPGA synthesis | **Yes** | — | — | Loihi only | — |
| IR compiler → SystemVerilog | **Yes** | — | — | — | — |
| Rust SIMD engine (512x) | **Yes** | — | — | — | — |
| Surrogate gradient training | Yes | Yes | Yes | Yes | — |
| GPU acceleration | CuPy | PyTorch | PyTorch | — | — |
| Neuron model library | **122** | 11 | 6 | 3 | ~5 builtin |
| Rust neuron models (PyO3) | **111** | — | — | — | — |
| NetworkRunner (fused loop) | **111 models** | — | — | — | — |
| Network simulation engine | **3 backends** | PyTorch | PyTorch | Lava | C++ codegen |
| MPI distributed simulation | **Yes** | — | — | — | — |
| Pre-trained model zoo | **10 configs, 3 weights** | — | — | — | — |
| Spike train analysis | **126 functions** | — | — | — | — |
| Visualization plots | **12** | — | — | — | — |
| Advanced plasticity rules | **7** | — | — | — | — |
| MNIST accuracy (SNN) | **99.49%** | ~95% | ~93% | — | — |
| Plasticity (STDP, R-STDP) | Yes | — | Yes | Yes | Yes |
| Quantum hybrid (Qiskit/PennyLane) | **Yes** | — | — | — | — |
| MLIR emitter (CIRCT) | **Yes** | — | — | — | — |
| Hyperdimensional computing | Yes | — | — | — | — |
| Formal verification (SymbiYosys) | **7 modules, 64 props** | — | — | — | — |
| JAX JIT training | **Yes** | — | — | — | — |
| CuPy sparse GPU | **Yes** | — | — | — | — |
| AI-optimized neurons | **9 (ArcaneNeuron + 8)** | — | — | — | — |
| Identity substrate | **Yes** | — | — | — | — |
| conda-forge recipe | **Ready** | Yes | — | — | Yes |
| PyPI package | Yes | Yes | Yes | Yes | Yes |
| License | AGPL-3.0 | MIT | LGPL-3.0 | BSD-3 | CeCILL-2.1 |

- **126-function spike train analysis toolkit** — CV, Fano factor, cross-correlation, Victor-Purpura distance, SPIKE-sync, Granger causality, GPFA, SPADE pattern detection, and 115 more functions. Matches Elephant + PySpike combined. Pure NumPy.

SC-NeuroCore's niche: **deterministic stochastic computing with FPGA co-design** — the only framework where Python simulation matches synthesisable RTL bit-for-bit.

### Network Simulation Engine

Population-Projection-Network architecture with 3 backends:

| Backend | Scope | Performance |
|---------|-------|-------------|
| **Python** | Any of 122 neuron models | NumPy vectorized |
| **Rust NetworkRunner** | 111 models in fused Rayon-parallel loop | 100K+ neurons, near-linear scaling |
| **MPI** | Billion-neuron distributed simulation via mpi4py | Multi-node HPC clusters |

6 topology generators (random, small-world, scale-free, ring, grid, all-to-all),
12 visualization plots (raster, voltage, ISI, cross-correlogram, PSD, firing rate,
phase portrait, population activity, instantaneous rate, spike train comparison,
network graph, weight matrix), and 7 advanced plasticity rules (BPTT, e-prop,
R-STDP, MAML, homeostatic, STP, structural).

### Model Zoo

10 pre-built network configurations (Brunel balanced, cortical column, CPG,
decision-making, working memory, visual cortex V1, auditory processing, MNIST
classifier, SHD speech classifier, DVS gesture classifier) with 3 pre-trained
weight sets (MNIST 784-128-10, SHD 700-256-20, DVS 256-256-11).

### 122 Neuron Models (1943--2026)

Every model has a uniform `step(current) -> spike` API, a `reset()`, and a
cited reference. One file per model in `src/sc_neurocore/neurons/models/`.

| Category | Count | Examples |
|----------|------:|---------|
| Integrate-and-fire variants | 18 | AdEx, GLIF5, ExpIF, QIF, SFA, MAT, COBA-LIF, Parametric LIF, Fractional LIF |
| Simple spiking (2D+) | 20 | FitzHugh-Nagumo, Morris-Lecar, Hindmarsh-Rose, Resonate-and-Fire, Chay |
| Biophysical (conductance-based) | 20 | Hodgkin-Huxley, Connor-Stevens, Traub-Miles, Mainen-Sejnowski, Pospischil |
| Stochastic / population / neural mass | 13 | Poisson, GLM, Jansen-Rit, Wong-Wang, Wilson-Cowan, Ermentrout-Kopell |
| Rate / plasticity / other | 12 | McCulloch-Pitts (1943), Sigmoid Rate, Astrocyte, Amari, GatedLIF (2022) |
| Hardware chip emulators | 9 | Loihi CUBA, Loihi 2, TrueNorth, BrainScaleS AdEx, SpiNNaker, Akida, DPI |
| Multi-compartment | 7 | Pinsky-Rinzel, Hay L5 Pyramidal, Rall Cable, Booth-Rinzel, Dendrify |
| Map-based (discrete-time) | 6 | Chialvo, Rulkov, Ibarz-Tanaka, Cazelles, Courbage-Nekorkin, Medvedev |
| Core (stochastic computing) | 5 | StochasticLIF, FixedPointLIF, HomeostaticLIF, Dendritic, SC-Izhikevich |
| Training cells (PyTorch) | 4 | LIF, ALIF, RecurrentLIF, EProp-ALIF |
| **AI-optimized (novel)** | **9** | **ArcaneNeuron, MultiTimescale, AttentionGated, PredictiveCoding, SelfReferential, CompositionalBinding, DifferentiableSurrogate, ContinuousAttractor, MetaPlastic** |

### ArcaneNeuron — Self-Referential Cognition

The flagship AI-optimized model. Five coupled subsystems in a single ODE:
fast compartment (tau=5ms), working memory (tau=200ms), deep context
(tau=10s), learned attention gate, and a forward self-model (predictor).
The deep compartment accumulates identity: it changes only on genuine
novelty (prediction errors), not routine input. Confidence modulates
threshold and meta-learning rate. No equivalent in any other toolkit.

### Identity Substrate

Persistent spiking network for identity continuity (`sc_neurocore.identity`).

| Module | Class | Purpose |
|--------|-------|---------|
| `substrate.py` | `IdentitySubstrate` | 3-population network (HH cortical + WB inhibitory + HR memory) with STDP and small-world connectivity |
| `encoder.py` | `TraceEncoder` | LSH-based text-to-spike-pattern encoding |
| `decoder.py` | `StateDecoder` | PCA + attractor extraction + priming context generation |
| `checkpoint.py` | `Checkpoint` | Lazarus protocol: save/restore/merge complete network state (.npz) |
| `director.py` | `DirectorController` | L16 cybernetic closure: monitor, diagnose, correct network dynamics |

## Quick Start

```bash
# Install from PyPI (ships the `sc-neurocore` product package)
pip install sc-neurocore

# Or install with all research modules included
pip install sc-neurocore[full]

# GPU acceleration (requires CUDA)
pip install sc-neurocore[gpu]
```

`pip install sc-neurocore` publishes the Python suite under the public
`sc-neurocore` package name. The optional Rust engine remains part of the
repository / release-asset / source-build flow rather than a separate PyPI
runtime dependency. Source-only Frontier modules such as `analysis`, `viz`,
`audio`, `dashboard`, and `swarm` still require a source checkout.

### Development Setup

```bash
git clone https://github.com/anulum/sc-neurocore.git
cd sc-neurocore
pip install -e ".[dev]"    # editable install with all dev tools
make preflight             # verify setup (lint + 2 112 tests)
```

If you are changing the Rust bridge locally, install `bridge/` in the same
environment or run source-tree commands with `PYTHONPATH=src:bridge`.

## Docker

The Docker image ships with the full Rust engine (512x real-time performance):

```bash
# Build
make docker-build
# or: docker build -f deploy/Dockerfile -t sc-neurocore:latest .

# Run interactive Python shell
make docker-run
# or: docker run --rm -it sc-neurocore:latest

# Smoke test via docker compose
docker compose -f deploy/docker-compose.yml up
```

Pre-built images are published to GHCR on every release:

```bash
docker pull ghcr.io/anulum/sc-neurocore:latest
docker run --rm -it ghcr.io/anulum/sc-neurocore:latest
```

## Architecture

### Module Tiers

`pip install sc-neurocore` ships **Core + Simulation + Domain bridges** only.
Research and Frontier modules are available from source (`pip install -e ".[dev]"`).

| Tier | Modules | Ships in wheel | Status |
|------|---------|:--------------:|--------|
| **Core** | neurons, synapses, layers, sources, utils, recorders, accel, compiler, hdl_gen, hardware, cli, exceptions | Yes | Production-ready. 100% coverage. |
| **Simulation** | hdc, solvers, transformers, learning, graphs, ensembles, export, pipeline, profiling, models, math, spatial, verification, security | Yes | Stable. Import explicitly. |
| **Domain bridges** | quantum (Qiskit/PennyLane), adapters/holonomic (JAX), scpn (Petri nets) | Yes | Requires `pip install sc-neurocore[quantum]` or `[jax]` |
| **Research** | robotics, physics, bio, optics, chaos, sleep, interfaces | No | Tested. Available from source. |
| **Frontier** | generative, world_model, analysis, audio, dashboard, viz, swarm | No | Experimental. Available from source. |
| **Speculative** | `research/` (eschaton, exotic, meta, post_silicon, transcendent) | No | Theoretical. See `research/README.md`. |

### Architecture Diagram

```mermaid
graph TD
    subgraph "Python API (pip install sc-neurocore)"
        A[BitstreamEncoder] --> B[SCDenseLayer / SCConv2DLayer]
        B --> C[122 Neuron Models<br/>LIF · HH · AdEx · Izhikevich · ArcaneNeuron · ...]
        C --> NET[Network Engine<br/>Population · Projection · 3 Backends]
        C --> ID[Identity Substrate<br/>Persistent SNN · Checkpoint · Director]
        C --> D[STDP / R-STDP Synapses]
        D --> E[BitstreamSpikeRecorder]
    end

    subgraph "Acceleration"
        B --> F{Backend?}
        F -->|CPU| G[NumPy / Numba SIMD]
        F -->|GPU| H[CuPy CUDA]
        F -->|Rust| I[sc_neurocore_engine<br/>512x · 111 neuron models<br/>111-model NetworkRunner]
        F -->|MPI| MPI[mpi4py distributed<br/>billion-neuron scale]
    end

    subgraph "Hardware Target"
        I --> J[IR Compiler]
        J --> K[SystemVerilog Emitter]
        J --> K2[MLIR/CIRCT Emitter]
        K --> L[Verilog RTL<br/>AXI-Lite + LIF Core]
        K2 --> L
        L --> M[FPGA Bitstream<br/>Xilinx / Intel]
        L --> V[Formal Verification<br/>SymbiYosys · 7 modules]
    end

    subgraph "Domain Bridges (optional)"
        B --> N[SCPN Petri Nets]
        B --> O[Quantum Hybrid<br/>Qiskit / PennyLane]
        B --> P[HDC/VSA Symbolic Memory]
    end

    style A fill:#2d6a4f,color:#fff
    style I fill:#b5651d,color:#fff
    style L fill:#1a237e,color:#fff
    style M fill:#4a148c,color:#fff
    style O fill:#6a1b9a,color:#fff
    style V fill:#004d40,color:#fff
```

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
  sc_bitstream_encoder.v      -- LFSR-based stochastic encoder (SEED_INIT param)
  sc_bitstream_synapse.v      -- AND-gate SC multiplier
  sc_dotproduct_to_current.v  -- Popcount -> fixed-point current
  sc_lif_neuron.v             -- Q8.8 leaky integrate-and-fire
  sc_firing_rate_bank.v       -- Spike rate estimator
  sc_dense_layer_core.v       -- Full dense layer pipeline (decorrelated seeds)
  sc_dense_matrix_layer.v     -- N×M weight matrix layer
  sc_neurocore_top.v          -- AXI-Lite configuration wrapper
  sc_axil_cfg.v               -- AXI-Lite register file
  sc_dense_layer_top.v        -- Dense layer top wrapper
  tb_sc_*.v (7 testbenches)   -- Self-checking simulation testbenches
  formal/ (7 modules)         -- SymbiYosys formal verification properties
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

### Reproducibility

Every GitHub Release includes:

- **wheel + sdist** — Python distribution artifacts (`dist/sc_neurocore-*`)
- **SBOM** — CycloneDX software bill of materials (`sbom.json`)
- **Changelog extract** — release notes from `CHANGELOG.md`

Co-simulation traces are generated deterministically from fixed LFSR seeds.
To reproduce a published benchmark:

```bash
git checkout v3.13.2
pip install -e ".[dev]"
python benchmarks/benchmark_suite.py --markdown > BENCHMARKS.md
```

For Verilog co-sim trace reproduction, see `scripts/cosim_gen_and_check.py`
and the seed constants in `hdl/sc_bitstream_encoder.v`.

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
| `02_sc_neuron_layer.py` | SCDenseLayer construction, spike trains, and firing-rate summary |
| `03_ir_compile_demo.py` | IR graph building, verification, SystemVerilog emission (v3 Rust engine) |
| `04_vectorized_layer.py` | VectorizedSCLayer throughput benchmarking |
| `05_scpn_stack.py` | Full 7-layer SCPN consciousness stack with inter-layer coupling |
| `06_hdl_generation.py` | Verilog top-level generation from a network description |
| `07_ensemble_consensus.py` | Multi-agent ensemble orchestration and voting |
| `08_hdc_symbolic_query.py` | Hyper-Dimensional Computing symbolic memory (v3 Rust engine) |
| `09_safety_critical_logic.py` | Fault-tolerant Boolean logic with stochastic redundancy (v3 Rust engine) |
| `10_benchmark_report.py` | Head-to-head v2/v3 benchmark suite (v3 Rust engine) |
| `11_sc_training_demo.py` | Surrogate-gradient training of an SC dense layer (v3 Rust engine) |
| `mnist_fpga/demo.py` | MNIST classifier: train → quantise Q8.8 → SC simulate → Verilog export |
| `mnist_conv_train.py` | **ConvSpikingNet: 99.49% MNIST** (learnable beta/threshold, cosine LR) |
| `mnist_surrogate/train.py` | Surrogate gradient SNN training (FastSigmoid/SuperSpike/ATan, ~95% MNIST) |

```bash
PYTHONPATH=src:bridge python examples/01_basic_sc_encoding.py
```

Examples marked **(v3 Rust engine)** require an available `sc_neurocore_engine`
bridge install. For source-tree runs against local bridge code, use
`PYTHONPATH=src:bridge` or install `bridge/` in the same environment.

## CI/CD

13 GitHub Actions workflows (`.github/workflows/`), all SHA-pinned:

| Workflow | Purpose |
|----------|---------|
| **ci.yml** | Lint (ruff format + ruff check + bandit) + Test (Python 3.10-3.14, coverage = 100%) + Build |
| **v3-engine.yml** | Rust engine `cargo test` + `cargo clippy` |
| **v3-wheels.yml** | Cross-platform wheels (Linux, macOS, Windows × Python 3.10–3.14) |
| **docker.yml** | Build & push Docker image to GHCR on release tags |
| **docs.yml** | MkDocs → GitHub Pages |
| **publish.yml** | Publish `sc-neurocore` to PyPI and `engine/` to crates.io on release tags |
| **release.yml** | Python wheel + sdist + changelog extraction → GitHub Release |
| **benchmark.yml** | Performance regression tracking |
| **codeql.yml** | CodeQL security analysis (weekly + on push) |
| **scorecard.yml** | OpenSSF Scorecard |
| **pre-commit.yml** | Pre-commit hook validation |
| **yosys-synth.yml** | Yosys HDL synthesis verification |
| **stale.yml** | Auto-label and close stale issues |

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

**Live site**: [anulum.github.io/sc-neurocore](https://anulum.github.io/sc-neurocore/)

- [Getting Started](docs/guides/getting-started.md) — Installation & quickstart
- [**Tutorials**](https://anulum.github.io/sc-neurocore/tutorials/01_stochastic_computing_fundamentals/) — 22 hands-on guides (SC fundamentals → MNIST → FPGA → quantum → formal verification)
- [API Reference](docs/api/API_REFERENCE.md) — Python package API
- [Rust Engine API](https://anulum.github.io/sc-neurocore/rust-api/sc_neurocore_engine/) — Rust engine docs
- [Hardware Guide](docs/hardware/HARDWARE_GUIDE.md) — FPGA deployment workflow
- [Architecture](docs/architecture/architecture.md) — Package architecture
- [Benchmarks](docs/benchmarks/BENCHMARKS.md) — Performance measurements
- [CHANGELOG.md](CHANGELOG.md) — Version history

Build docs locally:
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

## Install Extras

```bash
pip install sc-neurocore              # core engine only (neurons, layers, compiler, HDL gen)
pip install sc-neurocore[accel]       # + Numba JIT acceleration
pip install sc-neurocore[gpu]         # + CuPy CUDA acceleration
pip install sc-neurocore[jax]         # + JAX backend for holonomic adapters
pip install sc-neurocore[quantum]     # + Qiskit + PennyLane quantum bridges
pip install sc-neurocore[lava]        # + Intel Lava interop (Loihi target)
pip install sc-neurocore[research]    # + matplotlib, networkx, onnx, torch
pip install sc-neurocore[full]        # + numba, matplotlib, networkx, onnx, qiskit, pennylane
```

For development (includes all modules + research/frontier code from source):

```bash
pip install -e ".[dev]"               # editable install with pytest, mypy, ruff, hypothesis
```

Pinned dependency files for reproducible environments:

```bash
pip install -r requirements.txt       # runtime only
pip install -r requirements-dev.txt   # runtime + dev tools
```

## Rust Engine (111 Neuron Models, 336 Tests)

The `sc_neurocore_engine` crate provides 111 Rust neuron models callable
from Python via PyO3 bindings (including ArcaneNeuron), a 111-model
NetworkRunner with Rayon-parallel population simulation (100K+ neurons),
and SIMD-accelerated primitives with dispatch across five ISAs (AVX-512,
AVX2, NEON, SVE, RISC-V V).

336 Rust tests across 17 test binaries.

| Category | Scope |
|----------|-------|
| Primitives | Bernoulli + Sobol bitstream, pack/unpack, popcount, SIMD (5 ISAs) |
| Neurons | 111 models: LIF variants, HH-type, maps, hardware emulators, population, ArcaneNeuron |
| NetworkRunner | 111-model fused simulation loop with CSR projections and Rayon parallelism |
| Synapses | Static, STDP, Reward-STDP |
| Layers | Dense, Conv2D, Recurrent, Learning, Fusion, Memristive, Attention |
| Networks | Brunel, GNN, Spike recorder, Connectome, Fault injection |
| Compiler | IR builder/parser/verifier, SystemVerilog + MLIR emitters, IR bridge |
| Domain | HDC, Kuramoto, SSGF geometry |
| Training | 6 surrogate gradient functions + property tests |

## Community

- [GitHub Discussions](https://github.com/anulum/sc-neurocore/discussions) — questions, ideas, show & tell
- [Issue Tracker](https://github.com/anulum/sc-neurocore/issues) — bug reports and feature requests
- [Contributing Guide](CONTRIBUTING.md) — how to set up, test, and submit PRs

## Citation

If you use SC-NeuroCore in your research, please cite:

```bibtex
@software{sotek2026scneurocore,
  author    = {Šotek, Miroslav},
  title     = {SC-NeuroCore: A Deterministic Stochastic Computing Framework for Neuromorphic Hardware Design},
  version   = {3.13.2},
  year      = {2026},
  doi       = {10.5281/zenodo.18906614},
  url       = {https://github.com/anulum/sc-neurocore},
  license   = {AGPL-3.0-or-later}
}
```

See also [`CITATION.cff`](CITATION.cff) for the machine-readable citation metadata.

## License

SC-NeuroCore is dual-licensed:

- **Open Source**: [GNU Affero General Public License v3.0](LICENSE) (AGPLv3)
- **Commercial**: Proprietary license available for integration into closed-source products

For commercial licensing enquiries, contact [protoscience@anulum.li](mailto:protoscience@anulum.li).
