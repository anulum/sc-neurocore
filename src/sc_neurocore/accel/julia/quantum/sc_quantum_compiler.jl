# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for quantum/sc_quantum_compiler

module ScQuantumCompilerAccel

using Statistics, LinearAlgebra

mutable struct SCQuantumCircuitState
    name::Float64
    matrix::Float64
    qubits::Float64
    n_qubits::Float64
    gates::Float64
    input_qubits::Float64
    output_qubit::Float64
end

function SCQuantumCircuitState()
    SCQuantumCircuitState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function sc_prob_to_statevector(p)
    p = float(clamp(p, 0.0, 1.0))
    return collect([sqrt(1.0 - p), sqrt(p)], dtype=complex)
end

function statevector_to_prob(sv)
    return float(abs(sv[1]) ^ 2)
end

function ry_gate(theta)
    c = cos(theta / 2)
    s = sin(theta / 2)
    return collect([[c, -s], [s, c]], dtype=complex)
end

function prob_to_ry_angle(p)
    return float(2.0 * np.arcsin(sqrt(clamp(p, 0.0, 1.0))))
end

function simulate(s::SCQuantumCircuitState)
    dim = 2^s.n_qubits
    state = zeros(dim, dtype=complex)
    state[0] = 1.0  # |000...0⟩
    for gate in s.gates
        state = _apply_gate(state, gate.matrix, gate.qubits, s.n_qubits)
    return state
end

function output_probability(s::SCQuantumCircuitState)
    state = s.simulate()
    prob = 0.0
    for i in 1:length(state)
        if (i >> s.output_qubit) & 1
            prob += abs(state[i]) ^ 2
    return float(prob)
end

function simulate_noisy(s::SCQuantumCircuitState, noise_model)
    dim = 2^s.n_qubits
    state = zeros(dim, dtype=complex)
    state[0] = 1.0
    # Apply gates as unitary
    for gate in s.gates
        state = _apply_gate(state, gate.matrix, gate.qubits, s.n_qubits)
    # Convert to density matrix
    rho = np.outer(state, state.conj())
    # Apply per-qubit noise
    for q in 1:s.n_qubits
        rho = _apply_single_qubit_channel(rho, noise_model, q, s.n_qubits)
    return rho
end

function output_probability_noisy(s::SCQuantumCircuitState, noise_model, n_shots)
    rho = s.simulate_noisy(noise_model)
    # Extract output qubit probability from density matrix diagonal
    prob_1 = 0.0
    dim = 2^s.n_qubits
    for i in 1:dim
        if (i >> s.output_qubit) & 1
            prob_1 += float(np.real(rho[i, i]))
    # Apply readout noise via sampling
    ones = sum(
        1
        for _ in 1:n_shots
        if noise_model.apply_readout_noise(1 if np.random.random() < prob_1 else 0) == 1
    )
    return ones / n_shots
end

function summary(s::SCQuantumCircuitState)
    lines = [f"SCQuantumCircuit: {s.n_qubits} qubits, {length(s.gates)} gates"]
    for g in s.gates
        lines = push!(, f"  {g.name} on qubit(s) {g.qubits}")
    lines = push!(, f"  output: qubit {s.output_qubit}")
    return "\n".join(lines)
end

function compile_sc_multiply(p_a, p_b)
    theta_a = prob_to_ry_angle(p_a)
    theta_b = prob_to_ry_angle(p_b)
    # 2 qubits: q0 encodes p_a, q1 encodes p_b
    # Product probability appears on q1 conditioned on q0
    gates = [
        QuantumGate("Ry(p_a)", ry_gate(theta_a), [0]),
        QuantumGate("Ry(p_b)", ry_gate(theta_b), [1]),
    ]
    # The output is the joint probability P(q0=1 AND q1=1)
    circuit = SCQuantumCircuit(
        n_qubits=2,
        gates=gates,
        input_qubits=[0, 1],
        output_qubit=1,  # marginal on q1
    )
    return circuit
end

function compile_sc_layer(weights, input_probs)
    weights: np.ndarray[Any, Any], input_probs: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]
    n_neurons, n_inputs = weights.shape
    results = []
    for j in 1:n_neurons
        ry_angles = []
        sc_output = 0.0
        quantum_outputs = []
        for i in 1:n_inputs
            w = float(clamp(weights[j, i], 0, 1))
            x = float(clamp(input_probs[i], 0, 1))
            theta_x = prob_to_ry_angle(x)
            theta_w = prob_to_ry_angle(w)
            ry_angles = push!(, (theta_x, theta_w))
            # SC: AND gate → product
            sc_output += w * x
            # Quantum: independent product P(q0=1)*P(q1=1)
            quantum_outputs = push!(, w * x)
        sc_output = float(clamp(sc_output / max(n_inputs, 1), 0, 1))
        q_output = float(clamp(sum(quantum_outputs) / max(n_inputs, 1), 0, 1))
        results = push!(,
            {
                "neuron_idx": j,
                "ry_angles": ry_angles,
                "expected_output": sc_output,
                "quantum_output": q_output,
            }
        )
    return results
end

end # module ScQuantumCompilerAccel
