# Roadmap

> Last updated: 2026-03-15 (v3.12.0). Priorities may shift based on
> validation results and community feedback.

## v3.8 — Hardening & Edge AI Readiness ✓

### ~~Coverage gate ≥ 98%~~ ✓

Done. 100% enforced (1 451 tests passed; gate at 100).

### ~~NumPy 2.x full compatibility~~ ✓

Audit complete — zero deprecated calls found.

### ~~Enterprise CI/CD & supply chain hardening~~ ✓

13 CI workflows, all SHA-pinned. Bandit SAST, CodeQL, OpenSSF Scorecard.
Preflight gate with pre-push hook. PyPI OIDC trusted publisher. Python
minimum raised to 3.10.

### ~~Python API documentation~~ ✓

Live at GitHub Pages via mkdocstrings. Deploys on push to main.

### ~~Stale issue automation~~ ✓

`.github/workflows/stale.yml` — labels after 60 days, closes after 14 more.

### ~~Rust engine feature parity~~ ✓

Attention kernel: multi-head softmax with SIMD dispatch (475 lines).
Graph layer: CSR sparse backend (461 lines). MLIR emitter: CIRCT
hw/comb dialect output from IR graphs. 110 Rust neuron models with PyO3 bindings
(neurons, synapses, layers, networks, compiler IR).

### ~~Expanded SIMD kernels (issue #28)~~ ✓

AVX-512, AVX2, ARM NEON, ARM SVE, RISC-V RVV backends with
runtime dispatch. Operations: popcount, pack/unpack, fused
AND/XOR+popcount, dot/max/sum/scale f64, hamming distance,
softmax. SVE/RVV use portable fallbacks pending intrinsic
stabilisation in Rust.

## v3.9 — Quantum, SCPN, Benchmarks ✓

### ~~SCPN L1-L16 stack~~ ✓

16-layer SCPN stack complete. `create_full_stack()` returns all 16 layers.
`run_integrated_step()` chains L1→L16 with inter-layer coupling.

### ~~Formal verification~~ ✓

SymbiYosys proofs: LIF neuron (5 properties), bitstream synapse (4),
encoder (2). 11 formal properties total.

### ~~Brunel balanced-network benchmark~~ ✓

20-variant translator suite. Brian2 comparison with honest framing.
NeuroBench-aligned metrics (up to 847 MOP/s).

### ~~Co-simulation parity~~ ✓

Python golden model → Icarus Verilog → bit-exact checker.

### ~~Quantum backend stabilisation~~ ✓

Qiskit Aer + PennyLane backends validated (Python 3.10+).
IBM Heron r2 noise model (depolarizing, amplitude/phase damping,
asymmetric readout). Parameter-shift gradient rule for variational
circuits. Hybrid quantum-classical VQE pipeline with scipy optimizer.
QEC shield noise integration with surface code thresholds.

### ~~Holonomic adapter ecosystem~~ ✓

Per-adapter benchmark suite (`benchmarks/adapter_benchmark.py`):
L1-L16 latency, memory, throughput with/without JAX JIT.
All 16 adapters registered in ComponentRegistry with factory
function. Plugin discovery via `importlib.metadata` entry points.

## v3.10 — JOSS Paper & FPGA Demo ✓

### ~~JOSS paper~~ ✓

`paper/paper.md` — submission-ready. 12 references, MNIST results,
Brian2 comparison, formal verification data.

### ~~MNIST-on-FPGA demo~~ ✓

`examples/mnist_fpga/demo.py` — train → Q8.8 → SC → Verilog export.
Float 94.2%, Q8.8 94.2%, SC 94.0%. 10 synthesisable HDL modules.

### ~~Vivado tooling~~ ✓

`tools/vivado_impl.tcl` + `tools/vivado_report.py`. Non-project flow
targeting Xilinx 7-series.

### ~~Documentation overhaul~~ ✓

README benchmarks, all 10 HDL modules listed, Zenodo DOI corrected,
test counts updated across all docs.

## v3.12 — Competitive Sprint ✓ (current)

113 Python + 110 Rust neuron models, PyO3 bindings for all extended
model categories, JAX training support, CuPy sparse GPU paths, FMEA +
traceability matrix, 1 800 Python tests across 118 files, 209 Rust
tests, 13 CI workflows.

## v4.0 — Physical FPGA & Production (target: Q3 2026)

### FPGA deployment proof (P0 blocker)

- Deploy MNIST classifier on Artix-7 100T or Zynq 7020
- Measure: LUT count, BRAM, DSP, Fmax, dynamic power, latency
- End-to-end: Python → Rust IR → Verilog → bitstream → silicon
- Latency target: < 1 µs neuron update
- Deterministic replay: FPGA output matches Python bit-for-bit

### JOSS submission & review

- Submit via https://joss.theoj.org/papers/new
- Respond to reviewer feedback (estimated 4-8 weeks)

### Wheel trimming

- Remove frontier/speculative tiers from `pip install sc-neurocore`
- Keep generative, world_model, analysis, audio, dashboard, viz, swarm
  as source-only installs
- Fewer modules = stronger signal for core SC+SNN+FPGA story

### ~~Sparse weight matrices~~ ✓

CuPy CSR path added in `vectorized_layer.py` for N>1K networks.

### ~~JAX JIT compilation~~ ✓

`jax_forward_pass` + `jax_surrogate_gradient_step` added. GPU
acceleration benchmarks vs NumPy baseline.

### ~~Tool Qualification Kit (TQK)~~ ✓

FMEA + traceability matrix created in `docs/safety/`. Safety manual
and requirements-to-tests-to-formal-proofs mapping complete.

## v4.1 — Community & Ecosystem (target: Q4 2026)

### Community seeding

- ~~Awesome-neuromorphic listing~~ drafted, PR pending
- Conference lightning talk (NICE, ICONS, or Telluride)
- ~~Lab outreach~~ templates ready in `docs/internal/`
- GitHub Discussions with seeded categories
- ~~Publish `sc_neurocore_engine` wheels (trusted publisher)~~ ✓

### Silicon partnerships

- Intel Loihi 2: LAVA framework backend adapter
- SpiNNaker2: SpiNNTools compilation target
- Target: default middleware layer for neuromorphic silicon

### Industrial applications

- Robotics: SNN-based reactive controllers with formal timing
- Smart grids: stochastic load prediction with bounded latency
- Fusion control: real-time plasma state estimation (SCPN-Fusion-Core bridge)

### Formal SNN verification standard

Position SC-NeuroCore as reference implementation for formal SNN
verification:
- Liveness: every reachable marking can fire at least one transition
- Boundedness: token counts within proven upper bounds
- Deadlock freedom: compiler output is provably deadlock-free
