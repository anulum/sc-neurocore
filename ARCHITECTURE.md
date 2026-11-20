# Architecture

> **Canonical map: [`docs/architecture/SYSTEM_MAP.md`](docs/architecture/SYSTEM_MAP.md).**
> This file is retained for the SHD FPGA pipeline detail. The directory map below
> names directories only; the counts (model modules, documentation pages, test
> files and more) are generated in
> [`docs/_generated/capability_snapshot.md`](docs/_generated/capability_snapshot.md).

> Last updated: 2026-11-15

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
│   ├── tests/              Rust unit + integration tests
├── crates/                 Research Rust crates
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
│   ├── neurons/            Model source modules (models/ subdir) + core SC neurons
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
│   ├── formal/             FormalProofEngine → Lean 4 safety_bounds.lean
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
│   ├── analysis/           Spike train analysis toolkit
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
├── tests/                  Python test suite (current file count: docs/_generated/capability_snapshot.md)
│
├── hdl/                    Verilog RTL — top-level modules and their testbenches
│   ├── sc_neurocore_top.v  AXI-Lite top-level wrapper (generic SC network)
│   ├── sc_shd_top.v        SHD 3-stage pipelined inference core (legacy binary demo)
│   ├── sc_shd_axi_wrapper.v  AXI4-Lite slave for legacy sc_shd_top
│   ├── shd_event/          Event-driven integer SHD core + AXI4-Lite loader (sc_shd_event_axi.v)
│   ├── sc_vmin_lif_neuron.v   Vmin LIF (JIT eval order, Q8.8)
│   ├── sc_axonal_delay.v   Circular buffer axonal delay
│   ├── sc_dense_int8_sparse.v  CSR sparse int8 matvec
│   ├── sc_lif_neuron.v     Q8.8 leaky integrate-and-fire (generic)
│   ├── sc_dense_layer_core.v  Full SC dense pipeline
│   ├── sc_aer_encoder.v    AER spike encoder (event-driven output)
│   ├── sc_event_neuron.v   Event-triggered LIF (membrane updates on events and leak ticks)
│   ├── sc_aer_router.v     AER event distribution to target neurons
│   ├── sc_cordiv.v         CORDIV stochastic division
│   ├── sc_stdp_synapse.v   On-chip STDP learning
│   ├── formal/             SymbiYosys formal verification jobs (legacy and catalogue)
│   ├── constraints/        XDC timing constraints (PYNQ-Z2)
│   ├── reports/            Vivado synthesis reports (committed)
│   ├── pynq/               PYNQ-Z2 deployment (driver, demo, Tcl)
│   └── tb_*.v              Co-simulation testbenches
│
├── data/masquelier_shd/    SHD training scripts + cloud results + FPGA artifacts
├── examples/               Runnable demo scripts (SC, MNIST, NIR, JAX, snnTorch)
├── tools/                  Weight extraction, Q8.8 reference, cosim harness
├── benchmarks/             Python benchmark suite
├── scripts/                Co-simulation + utility scripts
├── paper/                  JOSS paper source
└── docs/                   MkDocs site source, the model documentation pages among it
```

## Data Flow

```
                    +---------------------------------------------+
                    |           Python API                        |
                    |  (model modules, training, analysis)        |
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

## SHD FPGA Pipeline

The SHD (Spiking Heidelberg Digits) pipeline runs from training to a
synthesised netlist and a PYNQ-Z2 driver package. Co-simulation checks the
Verilog against its Q8.8 reference; no board run has validated it yet:

```
SpikingJelly DCLS max training (Vertex AI T4)
    |  75.2% test, cosine sigma 15 -> 0.23, 0% rounding drop
    v
Weight extraction (tools/extract_shd_weights.py)
    |  int8 weights, integer delays, Q16.16 scales
    v
Q8.8 reference simulator (tools/shd_q88_reference.py)
    |  bit-true Python model; 76.06% against PyTorch's 75.66% on the
    |  2264-sample test set, as reported in 1616c7628 (2026-04-11);
    |  no committed record holds that run, and it was not measured again
    v
Verilog modules (hdl/sc_shd_top.v + 4 submodules)
    |  3-stage pipeline: AxDelay -> Dense -> Vmin_LIF
    v
iverilog co-simulation (hdl/tb_sc_shd_top.v)
    |  bit-exact match with Q8.8 reference
    v
Vivado v2025.2 out-of-context synthesis (Zynq XC7Z020, 100 MHz),
without weights or delays
    |  990 LUT (1.86%), 670 FF (0.63%), WNS +4.048 ns
    v
Block design script (hdl/pynq/create_block_design.tcl)
    |  Zynq PS + AXI-Lite; placed, routed and written to a bitstream on
    |  2026-04-13 without weights or delays (outputs no longer in the tree)
    v
PYNQ-Z2 driver package (hdl/pynq/), not yet run on a board
    sc_shd_driver.py -> SHDAccelerator class
    demo_shd_fpga.py -> demo script
```

### Vivado Synthesis Results (Zynq XC7Z020-1CLG400, 100 MHz)

| Resource | Used | Available | % |
|----------|------|-----------|---|
| LUTs | 990 | 53,200 | 1.86% |
| Flip-flops | 670 | 106,400 | 0.63% |
| BRAM | 0 | 140 | 0% |
| DSP48 | 0 | 220 | 0% |
| WNS | +4.048 ns | | ~168 MHz achievable |

Source artefacts: `hdl/reports/vivado_util_xc7z020_100mhz.rpt` and
`hdl/reports/vivado_timing_xc7z020_100mhz.rpt` (Vivado v2025.2,
design `sc_shd_top`, design state Synthesized, out-of-context run —
post-implementation counts are typically lower).
`tests/test_hardware_number_provenance.py` keeps this table bound to
those committed reports.

These figures describe a netlist without weights or delays.
`hdl/sc_shd_top.v` and `hdl/sc_dense_int8_sparse.v` load their delay tables
and weight ROM with `$readmemh` only when `SYNTHESIS` is undefined, and
`hdl/sc_shd_axi_wrapper.v` has no weight or delay register, so the
synthesised design carries no model. The figures are therefore not the
resource cost of the SHD network with its trained weights, and they do not
show that the network fits the device. The event-driven integer core and
integrity-checked AXI4-Lite loader in the subsection below provide the
loadable model path and its measured resource and timing cost.

### Event-Driven SHD Core with Loaded Weights and Delays (`sc_shd_event_axi`, Zynq XC7Z020-1CLG400)

`hdl/shd_event/sc_shd_event_axi.v` wraps the 8-lane event-driven integer SHD
core `hdl/shd_event/sc_shd_event_p.v` (`P=8`, `THETA1=72`, `THETA2=51`,
`V1_WIDTH=22`, `V2_WIDTH=17`, `CAP1=62`, `CAP2=39`, `WSHAPE=6`, `QSHAPE=12`,
matching `hdl/fixtures/shd_int_network.json`) in a 9-bit AXI4-Lite slave with
an integrity-checked 64-bit group-word loader (reflected IEEE CRC-32 matching
`zlib.crc32`, detecting accidental corruption without claiming authentication).
All weight banks (`q1`, `q2`, `q3`) and 32-bucket axonal-delay time wheels
(`w1`, `w2`) synthesise to 7-Series block RAM (`tech/xilinx_7series/sc_mem.v`),
while per-synapse delay tables (`d1`, `d2`) map to distributed `RAMD64E`
LUTRAM:

| Stage (`100 MHz` OOC constraint) | Slice LUTs (`53,200`) | Flip-Flops (`106,400`) | Block RAM Tiles (`140`) | DSP48 (`220`) | Setup `WNS` (`10.0 ns`) | Estimated `Fmax` from `100 MHz` `WNS` |
|---|---:|---:|---:|---:|---:|---:|
| Synthesised | 12,974 (24.39%) | 14,073 (13.23%) | 21 (`19 RAMB36E1 + 4 RAMB18E1`, 15.00%) | 0 (0.00%) | -6.887 ns | — |
| Placed and routed | 12,957 (24.36%) | 14,082 (13.23%) | 21 (`19 RAMB36E1 + 4 RAMB18E1`, 15.00%) | 0 (0.00%) | -3.343 ns | 74.95 MHz (`13.343 ns`) |

Source artefacts: `hdl/reports/vivado_sc_shd_event_axi_synth_util_xc7z020_100mhz.rpt`,
`hdl/reports/vivado_sc_shd_event_axi_synth_hier_util_xc7z020_100mhz.rpt`,
`hdl/reports/vivado_sc_shd_event_axi_synth_timing_xc7z020_100mhz.rpt`,
`hdl/reports/vivado_sc_shd_event_axi_routed_util_xc7z020_100mhz.rpt`,
`hdl/reports/vivado_sc_shd_event_axi_routed_hier_util_xc7z020_100mhz.rpt`,
`hdl/reports/vivado_sc_shd_event_axi_routed_timing_xc7z020_100mhz.rpt`, and
`hdl/reports/vivado_sc_shd_event_axi_routed_drc_xc7z020_100mhz.rpt`,
reproducible via `hdl/pynq/synth_shd_event_xc7z020.tcl` and
`hdl/constraints/sc_shd_event_axi_ooc_100mhz.xdc` (Vivado v2025.2, out-of-context
mode on `xc7z020clg400-1`).

In the routed hierarchy (`5,814 / 13,300` slices = `43.71%`, `12,917` logic
LUTs + `40` `RAMD64E` LUTRAMs), the AXI4-Lite wrapper and loader use `439`
LUTs and `906` flip-flops, while `u_core` (`sc_shd_event_p`) uses `12,520`
LUTs, `13,176` flip-flops, `19` `RAMB36E1` (`q1=9`, `q2=8`, `q3=2`), and `4`
`RAMB18E1` (`w1=2`, `w2=2`). The routed critical path (`u_core/pend1_reg[90]/C`
through `d1` `RAMD64E` delay lookup and `CARRY4` address arithmetic to
`u_core/w1/bk[1].col[0].ram_reg/ENBWREN`, `17` logic levels, `12.917 ns` data
path) misses the `100 MHz` (`10.000 ns`) constraint by `-3.343 ns` (`13.343 ns`
implied minimum period: `74.95 MHz` `Fmax` and `+6.657 ns` setup slack at
`50 MHz` / `20.0 ns`, estimated from the 100 MHz run's WNS (not a run at that
frequency); hold not closed out of context). Synthesised hold slack is `+0.173 ns` (`0` violations); routed `WHS` is
`-0.689 ns` (`241 / 38,859` intra-clock endpoints, `THS = -91.934 ns`), whose
reported worst path is input port `S_AXI_WDATA[0]` to `load_wdata_lo_reg_reg[0]/D`
(`0.500 + 0.924 - (1.660 + 0.035 + 0.418) = -0.689 ns` under
`HD.CLK_SRC BUFGCTRL_X0Y0`'s `1.660 ns` destination `BUFG` delay and an ideal
`0.000 ns` source-clock `set_input_delay -min 0.5` on unrouted OOC ports; the
summary report does not break the remaining `240` hold endpoints down). Routed
DRC (`hdl/reports/vivado_sc_shd_event_axi_routed_drc_xc7z020_100mhz.rpt`)
records `0` errors and `44` warnings (`CHECK-3` ×2, `REQP-1839` ×20, and
`REQP-1840` ×20 at their 20-violation rule limit where asynchronously reset
registers drive `RAMB36E1`/`RAMB18E1` address and enable pins — paths not
analysed by static timing — plus `RTSTAT-10` ×1 and expected out-of-context
`ZPS7-1` ×1). Full Zynq PS+PL block-design closure, physical PYNQ-Z2 board
execution, hardware latency, and power or energy consumption are not measured.

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
