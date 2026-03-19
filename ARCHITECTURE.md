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
│   ├── accel/              GPU (CuPy), JAX, JIT, MPI backends
│   ├── adapters/           SCPN layer adapters (L1-L16)
│   ├── analysis/           Explainability, metrics
│   ├── chaos/              Chaotic RNG
│   ├── compiler/           IR graph → SystemVerilog pipeline
│   ├── ensembles/          Multi-agent consensus
│   ├── export/             ONNX export
│   ├── graphs/             Petri nets, graph algorithms
│   ├── hardware/           HDL generation (Verilog emitter)
│   ├── hdc/                Hyper-dimensional computing (VSA)
│   ├── learning/           Neuroevolution, STDP, RL
│   ├── math/               SC arithmetic (add, mul, div)
│   ├── models/             Pre-built network architectures
│   ├── physics/            Wolfram hypergraph, optics
│   ├── quantum/            Quantum circuit stochastic bridge
│   ├── robotics/           Swarm coupling
│   ├── scpn/               SCPN layer implementations (L1-L16)
│   ├── security/           Pickle allowlist, input sanitisation
│   ├── sleep/              Sleep protocol optimiser
│   ├── solvers/            ODE solvers (RK4, Euler)
│   ├── spatial/            3D voxel grids, point clouds
│   ├── transformers/       Stochastic transformer blocks
│   └── verification/       Property-based test helpers
│
├── tests/                  Python test suite (1 785 tests, 100% coverage)
│
├── hdl/                    Verilog RTL (FPGA targets)
│   ├── sc_neurocore_top.v  AXI-Lite top-level wrapper
│   ├── sc_lif_neuron.v     Q8.8 leaky integrate-and-fire
│   ├── sc_dense_layer_core.v  Full dense pipeline
│   └── tb_sc_lif_neuron.v  Co-simulation testbench
│
├── examples/               Runnable demo scripts (01-11)
├── benchmarks/             Python benchmark suite
├── scripts/                Co-simulation + utility scripts
├── research/               Speculative / theoretical (not packaged)
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
