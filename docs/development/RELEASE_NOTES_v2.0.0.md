# SC-NeuroCore Release Notes v2.0.0

**Release Date:** January 12, 2026

## Executive Summary
This release marks the transition of `sc-neurocore` from an experimental library to a **Universal Stochastic Computing Platform**. It now supports the entire lifecycle of neuromorphic engineering: from quantum-inspired algorithm design to hardware-aware verification and deployment.

## Major Highlights

### 1. Massive Performance Boost
The introduction of `VectorizedSCLayer` and JIT compilation (via Numba) delivers **100x-1000x speedups** for dense layer simulations compared to the previous version.

### 2. Multi-Domain Integration
We have successfully integrated distinct scientific domains into a unified bitstream logic:
- **Quantum:** Variational Quantum Circuits.
- **Biology:** Genetic Regulation and DNA Storage.
- **Physics:** Stochastic PDE Solvers.
- **Robotics:** Swarm Intelligence and CPGs.

### 3. Hardware Realism
New verification tools ensure that software models match hardware reality:
- **Bit-True Verification:** `FixedPointLIFNeuron` matches RTL exactly.
- **Analog Defects:** `MemristiveDenseLayer` simulates stuck-at faults and conductance noise.
- **Synthesis Ready:** Automated Verilog and SPICE generation.

### 4. Enterprise Features
- **Security:** Model Watermarking and ZKP concepts protect IP.
- **Sustainability:** Granular Energy and Carbon profiling.
- **Interoperability:** ONNX-Schema export allows integration with broader ML pipelines.

## Upgrade Guide
Existing users of v1.0.0 should note:
- `StochasticLIFNeuron` signature has changed (added `refractory_period`).
- `BitstreamAverager` is significantly faster but maintains API compatibility.
- New dependencies: `numba` (optional but recommended), `docker` (for deployment).

## Future Outlook
With v2.0.0, `sc-neurocore` is positioned to be the standard-bearer for the 2026 Neuromorphic Computing wave, capable of driving everything from ultra-low power implants (BCI) to massive-scale photonic AI accelerators.
