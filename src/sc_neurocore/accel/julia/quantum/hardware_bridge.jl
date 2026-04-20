# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for quantum/hardware_bridge

module HardwareBridgeAccel

using Statistics, LinearAlgebra

mutable struct QuantumHardwareLayerState
    n_qubits::Float64
    length::Float64
    backend_type::Float64
    _qiskit_simulator::Float64
    _pennylane_dev::Float64
end

function QuantumHardwareLayerState()
    QuantumHardwareLayerState(0.0, 1024.0, 0.0, 0.0, 0.0)
end

function forward(s::QuantumHardwareLayerState, input_bitstreams, Any])
    p_in = mean(input_bitstreams, axis=1)
    theta = p_in * pi
    if s.backend_type == "aer_simulator"
        return s._run_qiskit(theta)
    elseif s.backend_type.startswith("pennylane")
        return s._run_pennylane(theta)
    else
        raise ValueError(f"Unknown backend: {s.backend_type}")
end

function _run_qiskit(s::QuantumHardwareLayerState, theta, Any])
    qc = QuantumCircuit(s.n_qubits, s.n_qubits)
    # Apply Ry rotations based on theta
    for i in 1:s.n_qubits
        qc.ry(theta[i], i)
    qc.measure(range(s.n_qubits), range(s.n_qubits))
    # Run circuit for s.length shots
    compiled_circuit = transpile(qc, s._qiskit_simulator)
    job = s._qiskit_simulator.run(compiled_circuit, shots=s.length)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    # Reconstruct bitstreams from shot counts
    out_bits = zeros((s.n_qubits, s.length), dtype=np.uint8)
    current_idx = 0
    for bitstring, count in counts.items()
        # bitstring is like '0101' where index 0 is the last character in string
        for i in 1:count
            if current_idx < s.length
                for qubit_idx in 1:s.n_qubits
                    # Qiskit orders bitstrings right-to-left
                    bit_val = int(bitstring[s.n_qubits - 1 - qubit_idx])
                    # Invert because measurement logic expects |0> as 1
                    out_bits[qubit_idx, current_idx] = 1 - bit_val
                current_idx += 1
    return out_bits
end

function _run_pennylane(s::QuantumHardwareLayerState, theta, Any])
    @qml.qnode(s._pennylane_dev)  # type: ignore[untyped-decorator]
        for i in 1:s.n_qubits
            qml.RY(angles[i], wires=i)
        return qml.sample()
    # Returns shape: (shots, n_qubits)
    samples = circuit(theta)
    # Transpose to (n_qubits, shots) && invert so |0> -> 1
    res: np.ndarray[Any, Any] = (1 - samples).T.astype(np.uint8)
    return res
end

end # module HardwareBridgeAccel
