# Roadmap

> Last updated: 2026-03-05. Priorities may shift based on validation results
> and community feedback.

## v3.8 — Hardening & Edge AI Readiness (target: Q2 2026)

### ~~Coverage gate ≥ 98%~~ ✓

Done. 98.10% (1058 tests, gate raised from 97→98). De-omitted 6 modules:
chaos/rng, analysis/explainability, physics/wolfram_hypergraph,
robotics/swarm, learning/neuroevolution, spatial/*. Remaining omissions
(audio, sleep, swarm, drivers, experiments) kept — hardware deps or demo code.

### ~~NumPy 2.x full compatibility~~ ✓

Audit complete — zero deprecated calls found. Codebase uses modern
`np.float64`/`np.int_`/`np.prod` etc. throughout.

### Rust engine feature parity

- Attention kernel: fused softmax in SIMD
- Graph layer: sparse CSR backend
- MLIR emitter: full operator coverage for L1-L16 adapters

### ~~Python API documentation (issue #6)~~ ✓

Live at GitHub Pages via mkdocstrings. All 44 subpackages wired into
nav. Deploys automatically on push to main.

### Expanded SIMD kernels (issue #5)

Add ARM SVE and RISC-V Vector (RVV) kernel variants to substantiate
sub-10 µs latency on modern automotive and edge ASICs.

### Hardware benchmarking suite (issue #7)

Implement a transparent benchmarking suite (aligned with NeuroBench
methodology) covering Apple Silicon, AMD, and target FPGAs.

### ~~Stale issue automation~~ ✓

`.github/workflows/stale.yml` — labels after 60 days, closes after 14
more. Exempt: `pinned`, `security`, `roadmap`.

## v3.9 — Quantum & Holonomic Expansion (target: Q3 2026)

### Quantum backend stabilisation

- Qiskit Aer + PennyLane backends validated against analytic results
- Noise model calibration for IBM Heron r2 hardware
- Quantum-classical hybrid gradient pipeline

### Holonomic adapter ecosystem

- L1-L16 adapters: add per-adapter benchmark suite
- Cross-layer coupling tests (L5↔L7, L1↔L16)
- Plugin registry for community-contributed adapters

### HDL synthesis flow

- Xilinx Zynq / Intel Cyclone bitstream export
- Co-simulation: Verilog ↔ Python bit-exact parity
- Formal verification coverage for all HDL modules

## v4.0 — Production & Safety-Critical Release (target: Q4 2026)

### FPGA deployment

- End-to-end: Python model → Rust IR → Verilog → bitstream
- Latency target: < 1 µs neuron update on FPGA
- Deterministic replay: FPGA output matches Python bit-for-bit

### JAX JIT compilation

- Full JIT path for UPDE solver + L1-L16 adapters
- GPU acceleration benchmarks vs NumPy baseline
- Gradient-through-solver for differentiable simulation

### Documentation & tutorials

- MkDocs site with API reference for all public modules
- Jupyter notebook tutorials for each SCPN layer
- Hardware setup guide for supported FPGA boards

### Tool Qualification Kit (TQK)

Commercial-tier package for safety-critical deployment:
- FMEA (Failure Mode and Effects Analysis) for SNN compiler pipeline
- Safety manual documenting deterministic execution guarantees
- Traceability matrix: requirements → tests → formal proofs
- Target: ISO 26262 ASIL-B qualification evidence for automotive clients

### Formal SNN verification standard

Position SC-NeuroCore as the reference implementation for formal SNN
verification — mathematical proofs for liveness and boundedness in
safety-critical Petri net → SNN compilation:
- Liveness: every reachable marking can fire at least one transition
- Boundedness: token counts remain within proven upper bounds
- Deadlock freedom: compiler output is provably deadlock-free

## v4.1 — Industrial Ecosystem (target: Q1 2027)

### Silicon partnerships

Direct integration paths with neuromorphic hardware:
- Intel Loihi 2: LAVA framework backend adapter
- SpiNNaker2: SpiNNTools compilation target
- Target: default middleware layer for neuromorphic silicon

### Industrial digital twin applications

Compiler as the algorithmic core for:
- Robotics: SNN-based reactive controllers with formal timing guarantees
- Smart grids: stochastic load prediction with bounded inference latency
- Autonomous fusion control: real-time plasma state estimation (SCPN-Fusion-Core bridge)
