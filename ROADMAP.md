# Roadmap

> Last updated: 2026-03-04. Priorities may shift based on validation results
> and community feedback.

## v3.8 — Hardening & Coverage (target: Q2 2026)

### Coverage gate ≥ 98%

Current: 97.76%. Remaining gaps are in swarm, audio, and sleep modules
(currently omitted from coverage). Target: bring omitted modules under test
or document why they remain excluded.

### NumPy 2.x full compatibility

All deprecated NumPy calls removed (ptp, etc.). Verify CI passes on
numpy>=2.0 across all test suites.

### Rust engine feature parity

- Attention kernel: fused softmax in SIMD
- Graph layer: sparse CSR backend
- MLIR emitter: full operator coverage for L1-L16 adapters

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

## v4.0 — Production Release (target: Q4 2026)

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
