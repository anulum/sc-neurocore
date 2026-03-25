# Architecture

SC-NeuroCore is a three-tier stochastic computing framework: a Rust SIMD
engine, a Python simulation layer, and Verilog RTL for FPGA deployment.

## Directory Map

```
sc-neurocore/
├── engine/                 Rust crate (PyO3 bindings via bridge/)
│   ├── src/
│   │   ├── bitstream.rs    Core bitstream type + SIMD popcount
│   │   ├── encoder.rs      LFSR stochastic encoder
│   │   ├── neuron.rs       Fixed-point LIF neuron
│   │   ├── layer.rs        Dense layer pipeline
│   │   ├── attention.rs    Stochastic attention head
│   │   ├── graph.rs        HDC hypervector graph
│   │   ├── ir/             Intermediate representation + SystemVerilog emitter
│   │   ├── grad/           Surrogate-gradient training
│   │   ├── simd/           AVX2 / AVX-512 / portable fallback
│   │   └── scpn/           Kuramoto phase solver + metrics
│   ├── tests/              Rust unit + integration tests
│   └── benches/            Criterion benchmarks
│
├── bridge/                 PyO3 Python ↔ Rust bridge (maturin build)
│
├── src/sc_neurocore/       Python package (pip install -e ".[dev]")
│   ├── neurons/            122 neuron models (models/ subdir) + core SC neurons
│   ├── synapses/           Bitstream synapses (static, STDP, R-STDP)
│   ├── layers/             SC dense, conv2d, recurrent, vectorized, fusion, memristive, attention
│   ├── sources/            Current sources (bitstream, Poisson)
│   ├── recorders/          Spike recorder, voltage recorder
│   ├── core/               Core bitstream types and SC arithmetic
│   ├── utils/              Bitstream encoding/decoding, RNG, helpers
│   ├── network/            Population-Projection-Network simulation engine
│   ├── accel/              GPU (CuPy), JAX, JIT, MPI backends
│   ├── compiler/           IR graph → SystemVerilog + MLIR/CIRCT pipeline
│   ├── hardware/           HDL generation (Verilog emitter)
│   ├── hdl_gen/            Equation-to-Verilog RTL compiler
│   ├── nir_bridge/         NIR import/export (18/18 primitives, multi-framework interop)
│   ├── identity/           Identity continuity substrate (SNN + checkpoint + director)
│   ├── learning/           STDP, BPTT, TBPTT, EWC, e-prop, R-STDP, MAML, STP, BCM
│   ├── training/           Surrogate gradient training cells
│   ├── hdc/                Hyper-dimensional computing (VSA)
│   ├── quantum/            Quantum circuit stochastic bridge (Qiskit + PennyLane)
│   ├── adapters/           SCPN layer adapters (L1-L16), holonomic (JAX)
│   ├── scpn/               SCPN layer implementations (L1-L16)
│   ├── analysis/           125-function spike train analysis toolkit
│   ├── math/               SC arithmetic (add, mul, div, CORDIV)
│   ├── solvers/            ODE solvers (RK4, Euler)
│   ├── graphs/             Petri nets, graph algorithms
│   ├── ensembles/          Multi-agent consensus
│   ├── transformers/       Stochastic transformer blocks
│   ├── model_zoo/          Pre-built network configs + pretrained weights
│   ├── models/             Pre-built network architectures
│   ├── export/             ONNX export
│   ├── pipeline/           Training/evaluation pipelines
│   ├── profiling/          Performance profiling tools
│   ├── datasets/           Dataset loaders (MNIST, SHD, DVS)
│   ├── bio/                Biological circuits (gap junctions, tripartite, cortical column)
│   ├── chaos/              Chaotic RNG
│   ├── optics/             Optical/photonic neuron models
│   ├── physics/            Wolfram hypergraph
│   ├── robotics/           Swarm coupling
│   ├── sleep/              Sleep protocol optimiser
│   ├── spatial/            3D voxel grids, point clouds
│   ├── audio/              Audio processing with spiking networks
│   ├── dashboard/          Web dashboard for network visualisation
│   ├── viz/                Visualisation (14 plots)
│   ├── swarm/              Swarm intelligence
│   ├── generative/         Generative spiking models
│   ├── world_model/        World model / predictive coding
│   ├── interfaces/         SpikeInterface adapter, external tool bridges
│   ├── integrations/       Third-party integrations
│   ├── drivers/            Hardware driver interfaces
│   ├── experiments/        Experiment management
│   ├── security/           Pickle allowlist, input sanitisation
│   ├── verification/       Property-based test helpers
│   ├── conversion/         ANN-to-SNN conversion (convert(), QCFS activation)
│   ├── cli/                Command-line interface (info, deploy, benchmark)
│   └── exceptions/         Custom exception hierarchy
│
├── tests/                  Python test suite (2 200+ tests, 100% coverage)
│
├── hdl/                    Verilog RTL (FPGA targets)
│   ├── sc_neurocore_top.v  AXI-Lite top-level wrapper
│   ├── sc_lif_neuron.v     Q8.8 leaky integrate-and-fire
│   ├── sc_dense_layer_core.v  Full dense pipeline
│   ├── sc_aer_encoder.v    AER spike encoder (event-driven output)
│   ├── sc_event_neuron.v   Event-triggered LIF (power ∝ spike rate)
│   ├── sc_aer_router.v     AER event distribution to target neurons
│   ├── formal/             SymbiYosys formal verification (7 modules, 67 properties)
│   └── tb_sc_*.v           Co-simulation testbenches
│
├── examples/               20 runnable demo scripts (SC, MNIST, NIR, JAX, snnTorch)
├── notebooks/              Jupyter notebooks
├── benchmarks/             Python benchmark suite
├── scripts/                Co-simulation + utility scripts
├── research/               Speculative / theoretical (not packaged)
├── paper/                  JOSS paper source
└── docs/                   MkDocs site source
```

## Data Flow

```
Python API  ──►  Rust Engine (PyO3)  ──►  SIMD kernels (AVX2/512)
    │                  │
    │                  ├──►  IR graph  ──►  SystemVerilog emitter
    │                  │
    ▼                  ▼
Co-simulation    Criterion benchmarks
testbench
    │
    ▼
Verilog RTL  ──►  FPGA bitstream (Vivado / Yosys)
```

## Build Targets

| Target | Command |
|--------|---------|
| Python package | `pip install -e ".[dev]"` |
| Rust engine | `cd engine && cargo build --release` |
| Python ↔ Rust bridge | `cd bridge && maturin develop --release` |
| Tests (Python) | `pytest tests/ -v` |
| Tests (Rust) | `cargo test --manifest-path engine/Cargo.toml` |
| Docs | `mkdocs serve` |
| Benchmarks (Rust) | `cargo bench --manifest-path engine/Cargo.toml` |
| Benchmarks (Python) | `python benchmarks/benchmark_suite.py` |
| Co-simulation | `python scripts/cosim_gen_and_check.py` |
