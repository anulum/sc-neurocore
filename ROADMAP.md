# Roadmap

> Last updated: 2026-03-09 (v3.10.0). Priorities may shift based on
> validation results and community feedback.

## v3.8 — Hardening & Edge AI Readiness ✓

### ~~Coverage gate ≥ 98%~~ ✓

Done. 98% enforced (1 100+ tests passed; gate at 98).

### ~~NumPy 2.x full compatibility~~ ✓

Audit complete — zero deprecated calls found.

### ~~Enterprise CI/CD & supply chain hardening~~ ✓

11 CI workflows, all SHA-pinned. Bandit SAST, CodeQL, OpenSSF Scorecard.
Preflight gate with pre-push hook. PyPI OIDC trusted publisher. Python
minimum raised to 3.10.

### ~~Python API documentation~~ ✓

Live at GitHub Pages via mkdocstrings. Deploys on push to main.

### ~~Stale issue automation~~ ✓

`.github/workflows/stale.yml` — labels after 60 days, closes after 14 more.

### Rust engine feature parity

- Attention kernel: fused softmax in SIMD
- Graph layer: sparse CSR backend
- MLIR emitter: full operator coverage for L1-L16 adapters

### Expanded SIMD kernels (issue #28)

Add ARM SVE and RISC-V Vector (RVV) kernel variants for
sub-10 µs latency on automotive and edge ASICs.

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

### Quantum backend stabilisation

- Qiskit Aer + PennyLane backends validated (Python 3.10+)
- Noise model calibration for IBM Heron r2 hardware
- Quantum-classical hybrid gradient pipeline

### Holonomic adapter ecosystem

- Per-adapter benchmark suite
- Plugin registry for community-contributed adapters

## v3.10 — JOSS Paper & FPGA Demo ✓ (current)

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

### Sparse weight matrices

- scipy.sparse CSR backend for N>1K networks
- Closes Brian2 performance gap at 10K+ neurons
- Dense N×N (800 MB at 10K) → sparse (O(N×C_E))

### JAX JIT compilation

- Full JIT path for UPDE solver + L1-L16 adapters
- GPU acceleration benchmarks vs NumPy baseline
- Gradient-through-solver for differentiable simulation

### Tool Qualification Kit (TQK)

Commercial-tier package for safety-critical deployment:
- FMEA for SNN compiler pipeline
- Safety manual: deterministic execution guarantees
- Traceability matrix: requirements → tests → formal proofs
- Target: ISO 26262 ASIL-B qualification evidence

## v4.1 — Community & Ecosystem (target: Q4 2026)

### Community seeding

- Awesome-neuromorphic listing (PR to GitHub lists)
- Conference lightning talk (NICE, ICONS, or Telluride)
- Lab outreach: 5 neuromorphic hardware labs
- GitHub Discussions with seeded categories
- Publish `sc_neurocore_engine` wheels (trusted publisher)

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
