<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 25: SC→Quantum Bridge

SC-NeuroCore includes a compiler that transpiles stochastic computing
operations to quantum circuits. The mapping is exact:

- SC probability $p$ ↔ quantum state $|\psi\rangle = \sqrt{1-p}|0\rangle + \sqrt{p}|1\rangle$
- SC AND gate ↔ joint measurement $P(q_0=1 \wedge q_1=1)$
- Born rule $P(|1\rangle) = |\beta|^2 = p$ (exact recovery)

## Why SC Maps to Quantum

Both SC and quantum computing encode information as probabilities.
An SC bitstream with $\Pr(\text{bit}=1) = p$ and a qubit with
$P(|1\rangle) = p$ carry the same information. The difference is that
quantum circuits use unitary rotations instead of random bit generation —
they're deterministic in the amplitude domain.

This means any SC circuit has a quantum equivalent, and the quantum version
computes the exact probability (no stochastic noise from finite bitstream length).

## 1. Encode a Probability

```python
from sc_neurocore.quantum import (
    sc_prob_to_statevector, statevector_to_prob, prob_to_ry_angle, ry_gate,
)

# SC probability → quantum statevector
sv = sc_prob_to_statevector(0.7)
print(f"|ψ⟩ = {sv}")  # [√0.3, √0.7]
print(f"P(|1⟩) = {statevector_to_prob(sv)}")  # 0.7 exactly

# The rotation angle
angle = prob_to_ry_angle(0.7)
print(f"Ry angle: {angle:.4f} rad")  # 2*arcsin(√0.7)
```

## 2. Compile SC Multiplication

SC multiplication is an AND gate. In quantum, this becomes a controlled
operation that entangles two qubits:

```python
from sc_neurocore.quantum import compile_sc_multiply

circuit = compile_sc_multiply(0.6, 0.7)
print(circuit.summary())

state = circuit.simulate()
p_11 = abs(state[3])**2  # P(q0=1 AND q1=1)
print(f"P(a AND b) = {p_11:.4f}")  # ≈ 0.42 (= 0.6 × 0.7)
```

## 3. Compile an SC Layer

Map a full weight matrix to quantum gates:

```python
import numpy as np
from sc_neurocore.quantum import compile_sc_layer

weights = np.array([[0.5, 0.3, 0.7], [0.8, 0.2, 0.4]])
inputs = np.array([0.6, 0.4, 0.8])
results = compile_sc_layer(weights, inputs)

for r in results:
    print(f"Neuron {r['neuron_idx']}: SC={r['expected_output']:.4f}, "
          f"Quantum={r['quantum_output']:.4f}")
```

The quantum outputs match the SC outputs exactly (no bitstream noise).

## 4. Noisy Simulation (IBM Heron r2)

Real quantum hardware has noise. The Heron r2 noise model simulates
depolarizing errors, amplitude/phase damping, and readout asymmetry:

```python
import numpy as np
from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

noise = HeronR2NoiseModel()

# Simulate noisy SC multiplication
circuit = compile_sc_multiply(0.6, 0.7)
rho = circuit.simulate_noisy(noise)
print(f"Trace(ρ) = {np.trace(rho).real:.6f}")  # ≈ 1.0

# Sample from noisy circuit
prob = circuit.output_probability_noisy(noise, n_shots=1000)
print(f"P(output=1) with noise: {prob:.3f}")  # ≈ 0.42 ± noise
```

### Noise channels

| Channel | Physical origin | Effect |
|---------|----------------|--------|
| Depolarizing | Gate errors | Random Pauli errors |
| Amplitude damping | T1 relaxation | $|1\rangle \to |0\rangle$ decay |
| Phase damping | T2 dephasing | Loss of quantum coherence |
| Readout error | Measurement noise | Bit-flip on readout (asymmetric) |

## 5. Hybrid Quantum-Classical Pipeline (VQE)

Combine variational quantum circuits with classical optimization:

```python
from sc_neurocore.quantum.hybrid_pipeline import HybridQuantumClassicalPipeline

pipeline = HybridQuantumClassicalPipeline(n_qubits=4, n_layers=2)

# Train the variational circuit
result = pipeline.train(n_steps=100, lr=0.01)
print(f"Training complete")

# Evaluate with learned parameters
energy = pipeline.evaluate(result)
print(f"Final energy: {energy:.4f}")
```

## 6. Parameter-Shift Gradients

Compute exact quantum gradients without backpropagation — shift each
parameter by $\pm\pi/2$ and measure the expectation value difference:

```python
from sc_neurocore.quantum.param_shift import (
    parameter_shift_gradient,
    ParameterShiftOptimizer,
)
import numpy as np

# Define a cost function that builds and evaluates a circuit
def cost_fn(params):
    from sc_neurocore.quantum.sc_quantum_compiler import compile_sc_multiply
    circuit = compile_sc_multiply(params[0], params[1])
    return circuit.output_probability()

# Exact gradient via parameter shift
params = np.array([0.5, 0.7])
grad = parameter_shift_gradient(cost_fn, params)
print(f"Gradient: {grad}")

# Gradient-based optimization
optimizer = ParameterShiftOptimizer(cost_fn, n_params=2, lr=0.1)
params = np.array([0.5, 0.7])
for s in range(50):
    params = optimizer.step(params)
    if s % 10 == 0:
        print(f"  Step {s}: cost = {cost_fn(params):.4f}")
```

## 7. Quantum Error Correction

Protect quantum circuits from noise using error correction codes:

```python
from sc_neurocore.quantum.qec import QecShield, SurfaceCodeShield

# Repetition code (distance-3)
shield = QecShield(code_type="repetition", distance=3)

# Surface code (distance-3)
surface = SurfaceCodeShield(distance=3)
```

## The SC↔Quantum Correspondence Table

| SC Operation | Quantum Equivalent | Gate |
|---|---|---|
| Encode $p$ | $R_y(2\arcsin\sqrt{p})$ | Single-qubit rotation |
| AND (multiply) | Toffoli (CCNOT) | 3-qubit gate |
| MUX (scaled add) | Controlled rotation | 2-qubit gate |
| NOT | Pauli-X | Single-qubit gate |
| XNOR (equality) | CNOT + X | 2-qubit gate |
| Popcount | Multi-qubit parity | Ancilla chain |
| Measure → $p$ | Born rule: $|\langle 1|\psi\rangle|^2$ | Measurement |

## Further Reading

- [Tutorial 15: Quantum-SC Hybrid Networks](15_quantum_sc_hybrid.md) — Qiskit/PennyLane integration
- [API: Quantum](../api/quantum.md) — auto-generated API docs
