# Quantum

Quantum-classical hybrid spiking layers: hardware backend bridge,
hybrid quantum-spiking circuits, quantum error correction, noise
modelling, and parameter-shift gradient optimisation.

## Research-grade optional backend boundary

The quantum stack is an opt-in research-grade backend. The default
`pip install sc-neurocore` path does not install Qiskit, PennyLane, Aer, IBM
Runtime, or quantum hardware credentials. Use `pip install
"sc-neurocore[quantum]"` only when a local experiment explicitly needs those
packages.

Simulator results, noisy emulation, and QPU dispatch artefacts are different
evidence classes. Do not promote simulator or PennyLane results as hardware
claims unless the run cites committed hardware provenance, backend name,
credentials-free metadata, and result artefacts.

## Hardware Bridge

::: sc_neurocore.quantum.hardware_bridge

## Hybrid Layer

::: sc_neurocore.quantum.hybrid

## Noise Models

IBM Heron r2 noise model with depolarising, amplitude damping,
phase damping channels, and asymmetric readout error.

::: sc_neurocore.quantum.noise_models

## Parameter-Shift Gradient

Exact gradient computation for parameterised quantum circuits.

::: sc_neurocore.quantum.param_shift

## Hybrid Pipeline

VQE-style quantum-classical optimisation pipeline.

::: sc_neurocore.quantum.hybrid_pipeline

## QEC

::: sc_neurocore.quantum.qec

## SC→Quantum Compiler (Conjecture C1+C4)

Compiles SC operations to quantum circuits. SC probability p encodes as
Ry(2·arcsin(√p)) rotation; AND gate maps to joint measurement; Born rule
recovers P(|1⟩) = p exactly. Includes noisy simulation via HeronR2NoiseModel.
The maintained compiler surface is exported from `sc_neurocore.quantum` for
direct selection:

```python
from sc_neurocore.quantum import compile_sc_layer, compile_sc_multiply

circuit = compile_sc_multiply(0.6, 0.7)
compiled_layer = compile_sc_layer(weights, inputs)
```

Existing submodule imports from `sc_neurocore.quantum.sc_quantum_compiler`
remain compatible. This wiring does not change quantum simulation algorithms,
polyglot safety mirrors, or benchmark-dispatched paths.

::: sc_neurocore.quantum.sc_quantum_compiler.sc_prob_to_statevector

::: sc_neurocore.quantum.sc_quantum_compiler.compile_sc_multiply

::: sc_neurocore.quantum.sc_quantum_compiler.compile_sc_layer

::: sc_neurocore.quantum.sc_quantum_compiler.SCQuantumCircuit
