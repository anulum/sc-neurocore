# Architecture

> **Canonical map: [`docs/architecture/SYSTEM_MAP.md`](docs/architecture/SYSTEM_MAP.md) (2026-06-23).**
> This file is retained for the SHD FPGA pipeline detail but its package figures
> are stale (e.g. "122 neuron models"; the verified count is 152 model source
> modules, matching `docs/_generated/capability_manifest.json`). Use SYSTEM_MAP.md.

> Last updated: 2026-04-13 (v3.14.0)

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
│   │   ├── gpu/            wgpu compute shader backend (feature-gated)
│   │   ├── simd/           AVX2 / AVX-512 / NEON / SVE / portable fallback
│   │   └── scpn/           Kuramoto phase solver + metrics
│   ├── tests/              Rust unit + integration tests (1,549)
├── crates/                 Research Rust crates (168 tests)
│   ├── tinysc_riscv/       RISC-V SC instruction set simulator
│   ├── core_engine/        SC arithmetic core
│   ├── autonomous_learning/ Self-modifying plasticity rules
│   ├── neuro_symbolic/     Hyperdimensional computing + predictive coding
│   └── stochastic_doctor_core/ Bitstream diagnostics engine
│   └── benches/            Criterion benchmarks (incl. gpu_bench.rs)
│
├── bridge/                 PyO3 Python ↔ Rust bridge (maturin build)
│
├── src/sc_neurocore/       Python package (pip install -e ".[dev]")
│   ├── neurons/            122 neuron models (models/ subdir) + core SC neurons
│   ├── synapses/           Bitstream synapses (static, STDP, R-STDP)
│   ├── layers/             SC dense, conv2d, recurrent, vectorized, fusion, attention
│   ├── sources/            Current sources (bitstream, Poisson)
│   ├── recorders/          Spike recorder, voltage recorder
│   ├── core/               Core bitstream types and SC arithmetic
│   ├── utils/              Bitstream encoding/decoding, RNG, helpers
│   ├── network/            Population-Projection-Network simulation engine
│   ├── accel/              GPU (CuPy), JAX, JIT, MPI, Mojo, Julia, Go backends
│   │   ├── mojo/           MojoKernelRunner + kernels.mojo SIMD primitives
│   │   ├── julia/solvers/  JuliaFusionSolver + DiffEq.jl reference ODEs
│   │   └── go/services/    aer_router, hil_debugger, services, services_ext
│   ├── arcane_zenith.py    ArcaneZenithCognitiveCore (neuron ⇄ 4-rule meta-plasticity)
│   ├── bioware/            MEA ↔ SC ↔ opto closed-loop (BioHybridSession, SpikeSorter)
│   ├── optics/             Photonic SC + FDTD (Berenger PML) + GDSII export
│   ├── evo_substrate/      Self-replicating SC organisms (Genome + FormalSafetyGuard)
│   ├── formal/             FormalProofEngine → Lean 4 safety_bounds.lean (6 theorems)
│   ├── edge/               AERRoutingDaemon — Python façade over Go AER UDP router
│   ├── debug/              Offline SpikeTracer + live HILServerDaemon / HILDebugger
│   ├── proto/              core.proto + telemetry.proto (wire contract for HIL)
│   ├── compiler/           IR graph → SystemVerilog + MLIR/CIRCT pipeline
│   ├── hardware/           HDL generation (Verilog emitter)
│   ├── hdl_gen/            Equation-to-Verilog RTL compiler + safety_monitor.sv
│   │   ├── safety/         neuro_safe_monitor (6 P-properties ↔ Lean theorems)
│   │   └── openroad_flow/  Yosys + OpenROAD ASIC synthesis driver
│   ├── nir_bridge/         NIR import/export (18/18 primitives, multi-framework interop)
│   ├─�� learning/           STDP, BPTT, TBPTT, EWC, e-prop, R-STDP, MAML, STP, BCM
│   ├── training/           Surrogate gradient training cells
│   ├── hdc/                Hyper-dimensional computing (VSA)
│   ├── quantum/            Quantum circuit stochastic bridge (Qiskit + PennyLane)
│   ├── quantum_cognition/  Fisher-Posner quantum cognition (experimental) + GOTM Brain
│   ├── adapters/           SCPN layer adapters (L1-L16), holonomic (JAX)
│   ├── scpn/               SCPN layer implementations (L1-L16)
│   ├── analysis/           127-function spike train analysis toolkit
│   ├── math/               SC arithmetic (add, mul, div, CORDIV)
│   ├── solvers/            ODE solvers (RK4, Euler)
│   ├── graphs/             Petri nets, graph algorithms
│   ├── model_zoo/          Pre-built network configs + pretrained weights
│   ├── pipeline/           Training/evaluation pipelines
│   ├── datasets/           Dataset loaders (MNIST, SHD, DVS)
│   ├── bio/                Biological circuits (gap junctions, tripartite, cortical column)
│   ├── spike_codec/        WaveformCodec (neural data compression, BCI)
│   ├── conversion/         ANN-to-SNN conversion (convert(), QCFS activation)
│   ├── cli/                Command-line interface (info, deploy, benchmark)
│   └── ...                 Additional research modules
│
├── tests/                  Python test suite (8 598+ tests, 100% core coverage)
│
├── hdl/                    Verilog RTL — 25 modules, 5 455 lines
│   ├── sc_neurocore_top.v  AXI-Lite top-level wrapper (generic SC network)
│   ├── sc_shd_top.v        SHD 3-stage pipelined inference core
│   ├── sc_shd_axi_wrapper.v  AXI4-Lite slave for Zynq PS-PL
│   ├── sc_vmin_lif_neuron.v   Vmin LIF (JIT eval order, Q8.8)
│   ├── sc_axonal_delay.v   Circular buffer axonal delay
│   ├── sc_dense_int8_sparse.v  CSR sparse int8 matvec
│   ├── sc_lif_neuron.v     Q8.8 leaky integrate-and-fire (generic)
│   ├── sc_dense_layer_core.v  Full SC dense pipeline
│   ├── sc_aer_encoder.v    AER spike encoder (event-driven output)
│   ├── sc_event_neuron.v   Event-triggered LIF (power proportional to spike rate)
│   ├── sc_aer_router.v     AER event distribution to target neurons
│   ├── sc_cordiv.v         CORDIV stochastic division
│   ├── sc_stdp_synapse.v   On-chip STDP learning
│   ├── formal/             SymbiYosys formal verification (7 modules, 67+ properties)
│   ├── constraints/        XDC timing constraints (PYNQ-Z2)
│   ├── reports/            Vivado synthesis reports (committed)
│   ├── pynq/               PYNQ-Z2 deployment (driver, demo, Tcl)
│   └── tb_*.v              Co-simulation testbenches (12 files)
│
├── data/masquelier_shd/    SHD training scripts + cloud results + FPGA artifacts
├── examples/               20 runnable demo scripts (SC, MNIST, NIR, JAX, snnTorch)
├── tools/                  Weight extraction, Q8.8 reference, cosim harness
├── benchmarks/             Python benchmark suite
├── scripts/                Co-simulation + utility scripts
├── paper/                  JOSS paper source
└── docs/                   MkDocs site source (122 model doc pages in progress)
```

## Data Flow

```
                    +---------------------------------------------+
                    |           Python API                        |
                    |  (122 neuron models, training, analysis)    |
                    +----------+------------------+---------------+
                               |                  |
                               v                  v
                    +------------------+  +-------------------+
                    |  Rust Engine     |  |  wgpu GPU Backend  |
                    |  (PyO3, SIMD)    |  |  (Vulkan compute)  |
                    +--------+---------+  +-------------------+
                             |
              +--------------+--------------+
              v              v              v
      +----------+  +--------------+  +--------------+
      | SIMD     |  | IR graph     |  | Criterion    |
      | kernels  |  |              |  | benchmarks   |
      +----------+  +------+-------+  +--------------+
                           |
                    +------+-------+
                    |  SV emitter  |
                    +------+-------+
                           |
                    +------+-------+
                    | Verilog RTL  |
                    +------+-------+
                           |
              +------------+------------+
              v            v            v
      +----------+  +----------+  +--------------+
      | iverilog |  | Yosys +  |  | Vivado       |
      | cosim    |  | nextpnr  |  | (Zynq/7-ser) |
      +----------+  | (ice40)  |  +------+-------+
                    +----------+         |
                                  +------+-------+
                                  |  Bitstream   |
                                  |  (.bit+.hwh) |
                                  +--------------+
```

## SHD FPGA Pipeline (proven end-to-end)

The SHD (Spiking Heidelberg Digits) pipeline demonstrates the full
train-to-deploy flow on real Xilinx silicon:

```
SpikingJelly DCLS max training (Vertex AI T4)
    |  75.2% test, cosine sigma 15 -> 0.23, 0% rounding drop
    v
Weight extraction (tools/extract_shd_weights.py)
    |  int8 weights, integer delays, Q16.16 scales
    v
Q8.8 reference simulator (tools/shd_q88_reference.py)
    |  bit-true Python model, 4% gap vs PyTorch
    v
Verilog modules (hdl/sc_shd_top.v + 4 submodules)
    |  3-stage pipeline: AxDelay -> Dense -> Vmin_LIF
    v
iverilog co-simulation (hdl/tb_sc_shd_top.v)
    |  bit-exact match with Q8.8 reference
    v
Vivado v2025.2 synthesis (Zynq XC7Z020, 100 MHz)
    |  1317 LUT (2.5%), 848 FF (0.8%), WNS +4.048 ns
    v
Bitstream generation (system_wrapper.bit + system.hwh)
    |  via Vivado block design (Zynq PS + AXI-Lite)
    v
PYNQ-Z2 deployment package (hdl/pynq/)
    sc_shd_driver.py -> SHDAccelerator class
    demo_shd_fpga.py -> end-to-end demo
```

### Vivado Synthesis Results (Zynq XC7Z020-1CLG400, 100 MHz)

| Resource | Used | Available | % |
|----------|------|-----------|---|
| LUTs | 1,317 | 53,200 | 2.5% |
| Flip-flops | 848 | 106,400 | 0.8% |
| BRAM | 0 | 140 | 0% |
| DSP48 | 0 | 220 | 0% |
| WNS | +4.048 ns | | ~168 MHz achievable |

All weights, delays, scales, and LUT contents are hardcoded in Verilog.
No external memory. The entire SHD inference network fits in 2.5% of a
Zynq-7020.

## Build Targets

| Target | Command |
|--------|---------|
| Python package | `pip install -e ".[dev]"` |
| Rust engine | `cd engine && cargo build --release` |
| Rust + GPU | `cd engine && cargo build --release --features gpu` |
| Python-Rust bridge | `cd bridge && maturin develop --release` |
| Tests (Python) | `pytest tests/ -v` |
| Tests (Rust) | `cargo test --manifest-path engine/Cargo.toml` |
| Docs | `mkdocs serve` |
| Benchmarks (Rust) | `cargo bench --manifest-path engine/Cargo.toml` |
| Benchmarks (Python) | `python benchmarks/benchmark_suite.py` |
| Co-simulation | `python scripts/cosim_gen_and_check.py` |
| Verilog sim | `iverilog -o sim hdl/sc_shd_top.v hdl/tb_sc_shd_top.v && vvp sim` |
