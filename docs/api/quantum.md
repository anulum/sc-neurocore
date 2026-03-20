# Quantum

Quantum-classical hybrid spiking layers: hardware backend bridge,
hybrid quantum-spiking circuits, quantum error correction, noise
modelling, and parameter-shift gradient optimization.

## Hardware Bridge

::: sc_neurocore.quantum.hardware_bridge

## Hybrid Layer

::: sc_neurocore.quantum.hybrid

## Noise Models

IBM Heron r2 noise model with depolarizing, amplitude damping,
phase damping channels, and asymmetric readout error.

::: sc_neurocore.quantum.noise_models

## Parameter-Shift Gradient

Exact gradient computation for parameterized quantum circuits.

::: sc_neurocore.quantum.param_shift

## Hybrid Pipeline

VQE-style quantum-classical optimization pipeline.

::: sc_neurocore.quantum.hybrid_pipeline

## QEC

::: sc_neurocore.quantum.qec

## SC→Quantum Compiler (Conjecture C1+C4)

Compiles SC operations to quantum circuits. SC probability p encodes as
Ry(2·arcsin(√p)) rotation; AND gate maps to joint measurement; Born rule
recovers P(|1⟩) = p exactly. Includes noisy simulation via HeronR2NoiseModel.

::: sc_neurocore.quantum.sc_quantum_compiler.sc_prob_to_statevector

::: sc_neurocore.quantum.sc_quantum_compiler.compile_sc_multiply

::: sc_neurocore.quantum.sc_quantum_compiler.compile_sc_layer

::: sc_neurocore.quantum.sc_quantum_compiler.SCQuantumCircuit
