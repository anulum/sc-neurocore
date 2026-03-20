# Tutorial 25: SC→Quantum Bridge

SC-NeuroCore includes a compiler that transpiles stochastic computing
operations to quantum circuits. The mapping is exact (Conjecture C1+C4):

- SC probability p ↔ quantum state |ψ⟩ = √(1-p)|0⟩ + √p|1⟩
- SC AND gate ↔ joint measurement P(q0=1 ∧ q1=1)
- Born rule P(|1⟩) = |β|² = p (exact recovery)

## Encode a Probability

```python
from sc_neurocore.quantum.sc_quantum_compiler import (
    sc_prob_to_statevector, statevector_to_prob, prob_to_ry_angle, ry_gate,
)

sv = sc_prob_to_statevector(0.7)
print(f"|ψ⟩ = {sv}")
print(f"P(|1⟩) = {statevector_to_prob(sv)}")  # 0.7 exactly
```

## Compile SC Multiplication

```python
from sc_neurocore.quantum.sc_quantum_compiler import compile_sc_multiply

circuit = compile_sc_multiply(0.6, 0.7)
print(circuit.summary())

state = circuit.simulate()
p_11 = abs(state[3])**2  # P(q0=1 AND q1=1)
print(f"P(a AND b) = {p_11:.4f}")  # ≈ 0.42
```

## Compile an SC Layer

```python
import numpy as np
from sc_neurocore.quantum.sc_quantum_compiler import compile_sc_layer

weights = np.array([[0.5, 0.3, 0.7], [0.8, 0.2, 0.4]])
inputs = np.array([0.6, 0.4, 0.8])
results = compile_sc_layer(weights, inputs)

for r in results:
    print(f"Neuron {r['neuron_idx']}: SC={r['expected_output']:.4f}, "
          f"Quantum={r['quantum_output']:.4f}")
```

## Noisy Simulation

```python
from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

noise = HeronR2NoiseModel()
circuit = compile_sc_multiply(0.6, 0.7)
rho = circuit.simulate_noisy(noise)
print(f"Trace(ρ) = {np.trace(rho).real:.6f}")  # ≈ 1.0
prob = circuit.output_probability_noisy(noise, n_shots=1000)
print(f"P(output=1) with noise: {prob:.3f}")
```
