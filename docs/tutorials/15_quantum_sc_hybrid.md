<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Quantum-Stochastic Computing Hybrid Networks

Combine quantum circuits with stochastic computing neurons. Quantum
gates produce measurement probabilities that map naturally to SC
bitstream probabilities — no additional encoding needed.

**Prerequisites**: `pip install sc-neurocore pennylane`

## 1. Why quantum + SC?

Quantum computing and stochastic computing share a fundamental
representation: probabilities. A qubit measurement yields 0 or 1
with some probability p — exactly the semantics of a stochastic
bitstream bit. This makes quantum circuits natural front-ends
for SC neural networks.

| Quantum output | SC interpretation |
|---------------|-------------------|
| P(|1⟩) = 0.7 | Bitstream with 70% ones |
| Measurement sequence | Bitstream directly |
| Multi-qubit state | Multiple correlated bitstreams |

## 2. Quantum feature encoder

Encode classical data into quantum amplitudes, measure, and feed
the measurement probabilities to an SC layer:

```python
import numpy as np

try:
    import pennylane as qml

    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits, shots=256)

    @qml.qnode(dev)
    def quantum_encoder(x):
        """Angle encoding: each feature → RY rotation."""
        for i in range(n_qubits):
            qml.RY(x[i] * np.pi, wires=i)
        # Entangle
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Encode a test vector
    x = np.array([0.3, 0.7, 0.5, 0.9])
    expectations = quantum_encoder(x)
    # Map from [-1, 1] to [0, 1] for SC
    probs = [(e + 1) / 2 for e in expectations]
    print(f"Input: {x}")
    print(f"Quantum probs: {[f'{p:.3f}' for p in probs]}")

except ImportError:
    print("PennyLane not installed — using simulated quantum output")
    probs = [0.35, 0.72, 0.48, 0.91]
```

## 3. Quantum → SC pipeline

```python
from sc_neurocore import VectorizedSCLayer

# Quantum encoder outputs n_qubits probabilities
# SC layer processes them
sc_layer = VectorizedSCLayer(n_inputs=4, n_neurons=8, length=256)

quantum_output = np.array(probs)
quantum_output = np.clip(quantum_output, 0.01, 0.99)
sc_result = sc_layer.forward(quantum_output)
print(f"SC layer output: {sc_result}")
```

## 4. Variational quantum-SC classifier

A hybrid classifier: parameterised quantum circuit (feature extraction)
→ SC dense layer (classification):

```python
try:
    n_qubits = 4
    n_layers_q = 2  # quantum circuit depth

    @qml.qnode(dev)
    def variational_circuit(x, params):
        """Parameterised quantum circuit for feature extraction."""
        # Encode data
        for i in range(n_qubits):
            qml.RY(x[i] * np.pi, wires=i)

        # Variational layers
        for l in range(n_layers_q):
            for i in range(n_qubits):
                qml.RY(params[l, i, 0], wires=i)
                qml.RZ(params[l, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Initialise parameters
    params = np.random.uniform(-np.pi, np.pi, (n_layers_q, n_qubits, 2))

    # Forward pass
    x_test = np.array([0.2, 0.8, 0.4, 0.6])
    q_out = variational_circuit(x_test, params)
    sc_input = np.clip([(e + 1) / 2 for e in q_out], 0.01, 0.99)
    prediction = sc_layer.forward(np.array(sc_input))
    print(f"Hybrid prediction: {prediction.argmax()}")

except NameError:
    print("Skipping variational circuit (PennyLane not available)")
```

## 5. Quantum noise as SC randomness

Quantum measurement is inherently random — instead of using a PRNG
to generate bitstreams, use actual quantum random bits:

```python
try:
    @qml.qnode(dev)
    def quantum_random_bits(p, n_shots_implicit=None):
        """Generate random bits with probability p using a qubit."""
        qml.RY(2 * np.arcsin(np.sqrt(p)), wires=0)
        return qml.sample(qml.PauliZ(0))

    # Generate a bitstream with P(1) = 0.7
    samples = quantum_random_bits(0.7)
    # Map from {-1, +1} to {0, 1}
    bitstream = ((np.array(samples) + 1) / 2).astype(int)
    actual_prob = bitstream.mean()
    print(f"Target: 0.700, Actual: {actual_prob:.3f} ({len(bitstream)} bits)")

except NameError:
    print("Skipping quantum RNG (PennyLane not available)")
```

## 6. Entanglement-correlated bitstreams

Entangled qubits produce correlated measurement outcomes.
SC multiplication of correlated bitstreams gives different results
than independent streams — this is a feature, not a bug:

```python
try:
    dev2 = qml.device("default.qubit", wires=2, shots=1000)

    @qml.qnode(dev2)
    def bell_pair():
        """Create maximally entangled Bell state |Φ+⟩."""
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

    s0, s1 = bell_pair()
    b0 = ((np.array(s0) + 1) / 2).astype(int)
    b1 = ((np.array(s1) + 1) / 2).astype(int)

    # SC multiply (AND gate)
    product = b0 & b1
    independent_expected = b0.mean() * b1.mean()
    actual = product.mean()
    print(f"P(0): {b0.mean():.3f}, P(1): {b1.mean():.3f}")
    print(f"Independent product: {independent_expected:.3f}")
    print(f"Entangled product:   {actual:.3f}")
    print(f"Correlation boost:   {actual - independent_expected:+.3f}")

except NameError:
    print("Skipping Bell pair demo (PennyLane not available)")
```

## 7. Hybrid training loop

Train both quantum parameters and SC weights jointly:

```python
def hybrid_forward(x, q_params, sc_layer_obj):
    """Quantum encoder → SC classifier."""
    try:
        q_out = variational_circuit(x, q_params)
        sc_in = np.clip([(e + 1) / 2 for e in q_out], 0.01, 0.99)
    except NameError:
        # Fallback: simulate quantum encoding with sigmoid
        sc_in = 1 / (1 + np.exp(-x[:4]))
        sc_in = np.clip(sc_in, 0.01, 0.99)

    return sc_layer_obj.forward(np.array(sc_in))

# Training sketch (parameter-shift for quantum, pseudo-gradient for SC)
LR_Q = 0.01   # quantum parameter learning rate
LR_SC = 0.005  # SC weight learning rate
# ... gradient computation omitted for brevity
```

## 8. When to use quantum-SC hybrids

| Scenario | Recommended |
|----------|-------------|
| Small feature space (< 20 dims) | Quantum encoder → SC |
| Large feature space (> 100 dims) | Classical PCA → SC |
| Correlated features | Entangled quantum encoding |
| Low-power edge deployment | SC only (FPGA) |
| Quantum hardware available | Full hybrid |
| Research/exploration | Simulated quantum + SC |

## 9. Resource estimates

```python
# Quantum resources
n_qubits_needed = 4
circuit_depth = 2 * n_qubits  # approximate
shots_per_inference = 256  # matches SC bitstream length

# SC resources (from Tutorial 14)
sc_luts = 128 * 50 + 10 * 128  # rough LUT estimate
print(f"Quantum: {n_qubits_needed} qubits, depth {circuit_depth}, {shots_per_inference} shots")
print(f"SC FPGA: ~{sc_luts} LUTs")
print(f"Total inference time: {shots_per_inference * 100e-9 * 1e6:.1f} μs (quantum) + "
      f"{512 * 10e-9 * 1e6:.1f} μs (SC @ 100 MHz)")
```

## What you learned

- Quantum measurement probabilities map directly to SC bitstream probabilities
- Angle encoding (RY gates) converts classical features to qubit amplitudes
- Variational quantum circuits serve as trainable feature extractors
- Entangled qubits produce correlated bitstreams (useful for structured computation)
- Hybrid training: parameter-shift rule (quantum) + pseudo-gradient (SC)
- Practical only for small feature spaces until quantum hardware scales

## Next steps

- Implement quantum kernel estimation → SC classification
- Compare quantum-SC vs classical-SC on XOR / iris datasets
- Use IBM Qiskit backend instead of PennyLane simulator
- Explore quantum error mitigation for noisy SC bitstreams
