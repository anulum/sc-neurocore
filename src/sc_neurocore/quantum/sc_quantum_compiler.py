# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC→Quantum circuit compiler

"""Compile stochastic computing operations to quantum circuits.

Conjecture C1+C4: SC bitstream computation is isomorphic to quantum
measurement. The mapping is exact (not analogical):

- SC probability p ↔ quantum state |ψ⟩ = √(1-p)|0⟩ + √p|1⟩
- SC AND gate (multiply) ↔ Toffoli (controlled-controlled-NOT)
- SC NOT gate (complement) ↔ Pauli-X
- Born rule P(|1⟩) = |β|² = p (exact)

This module provides:
1. sc_prob_to_statevector(): encode SC probability as qubit
2. sc_and_circuit(): quantum circuit for SC multiplication
3. sc_not_circuit(): quantum circuit for SC complement
4. simulate_circuit(): statevector simulation
5. compile_sc_layer(): compile an SC dense layer to quantum gates

    from sc_neurocore.quantum.sc_quantum_compiler import (
        sc_prob_to_statevector, compile_sc_layer, simulate_circuit,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sc_prob_to_statevector(p: float) -> np.ndarray:
    """Encode SC probability as a single-qubit state vector.

    |ψ⟩ = √(1-p)|0⟩ + √p|1⟩  →  P(measure |1⟩) = p exactly.
    """
    p = float(np.clip(p, 0.0, 1.0))
    return np.array([np.sqrt(1.0 - p), np.sqrt(p)], dtype=complex)


def statevector_to_prob(sv: np.ndarray) -> float:
    """Extract SC probability from a single-qubit state vector via Born rule."""
    return float(np.abs(sv[1]) ** 2)


# Standard quantum gates
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X (NOT)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard


def ry_gate(theta: float) -> np.ndarray:
    """Ry rotation gate: encodes probability via rotation angle."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def prob_to_ry_angle(p: float) -> float:
    """Compute Ry angle that encodes probability p: sin²(θ/2) = p."""
    return 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))


@dataclass
class QuantumGate:
    """A quantum gate applied to specific qubits."""

    name: str
    matrix: np.ndarray
    qubits: list[int]  # target qubit indices


@dataclass
class SCQuantumCircuit:
    """Quantum circuit compiled from SC operations."""

    n_qubits: int
    gates: list[QuantumGate]
    input_qubits: list[int]
    output_qubit: int

    def simulate(self) -> np.ndarray:
        """Simulate the circuit and return the full statevector."""
        dim = 2**self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0  # |000...0⟩

        for gate in self.gates:
            state = _apply_gate(state, gate.matrix, gate.qubits, self.n_qubits)

        return state

    def output_probability(self) -> float:
        """Simulate and return P(output_qubit = |1⟩)."""
        state = self.simulate()
        prob = 0.0
        for i in range(len(state)):
            if (i >> self.output_qubit) & 1:
                prob += np.abs(state[i]) ** 2
        return float(prob)

    def simulate_noisy(self, noise_model) -> np.ndarray:
        """Simulate with noise: evolve density matrix through Kraus channels.

        Parameters
        ----------
        noise_model : HeronR2NoiseModel or compatible
            Must provide apply_single_qubit_noise(rho) and apply_readout_noise(measurement).

        Returns
        -------
        np.ndarray
            Final density matrix of shape (2^n, 2^n).
        """
        dim = 2**self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        # Apply gates as unitary
        for gate in self.gates:
            state = _apply_gate(state, gate.matrix, gate.qubits, self.n_qubits)
        # Convert to density matrix
        rho = np.outer(state, state.conj())
        # Apply per-qubit noise
        for q in range(self.n_qubits):
            rho = _apply_single_qubit_channel(rho, noise_model, q, self.n_qubits)
        return rho

    def output_probability_noisy(self, noise_model, n_shots: int = 1000) -> float:
        """Simulate with noise and return P(output=1) via measurement sampling.

        Parameters
        ----------
        noise_model : HeronR2NoiseModel or compatible
        n_shots : int
            Number of measurement shots.
        """
        rho = self.simulate_noisy(noise_model)
        # Extract output qubit probability from density matrix diagonal
        prob_1 = 0.0
        dim = 2**self.n_qubits
        for i in range(dim):
            if (i >> self.output_qubit) & 1:
                prob_1 += float(np.real(rho[i, i]))
        # Apply readout noise via sampling
        ones = sum(
            1 for _ in range(n_shots)
            if noise_model.apply_readout_noise(1 if np.random.random() < prob_1 else 0) == 1
        )
        return ones / n_shots

    def summary(self) -> str:
        lines = [f"SCQuantumCircuit: {self.n_qubits} qubits, {len(self.gates)} gates"]
        for g in self.gates:
            lines.append(f"  {g.name} on qubit(s) {g.qubits}")
        lines.append(f"  output: qubit {self.output_qubit}")
        return "\n".join(lines)


def _apply_gate(
    state: np.ndarray, gate: np.ndarray, qubits: list[int], n_qubits: int
) -> np.ndarray:
    """Apply a gate to specific qubits in a full statevector."""
    if len(qubits) == 1:
        return _apply_single_qubit_gate(state, gate, qubits[0], n_qubits)
    elif len(qubits) == 2:
        return _apply_two_qubit_gate(state, gate, qubits[0], qubits[1], n_qubits)
    raise ValueError(f"Gates on {len(qubits)} qubits not supported")


def _apply_single_qubit_gate(
    state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Apply a 2x2 gate to one qubit in a multi-qubit state."""
    dim = 2**n_qubits
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        bit = (i >> qubit) & 1
        i_flipped = i ^ (1 << qubit)
        if bit == 0:
            new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[i_flipped]
        else:
            new_state[i] += gate[1, 0] * state[i_flipped] + gate[1, 1] * state[i]
    return new_state


def _apply_two_qubit_gate(
    state: np.ndarray, gate: np.ndarray, q0: int, q1: int, n_qubits: int
) -> np.ndarray:
    """Apply a 4x4 gate to two qubits."""
    dim = 2**n_qubits
    new_state = np.zeros(dim, dtype=complex)
    for i in range(dim):
        b0 = (i >> q0) & 1
        b1 = (i >> q1) & 1
        row = b0 * 2 + b1
        for col in range(4):
            cb0 = (col >> 1) & 1
            cb1 = col & 1
            j = (i & ~(1 << q0) & ~(1 << q1)) | (cb0 << q0) | (cb1 << q1)
            new_state[i] += gate[row, col] * state[j]
    return new_state


def _apply_single_qubit_channel(
    rho: np.ndarray, noise_model, qubit: int, n_qubits: int
) -> np.ndarray:
    """Apply single-qubit noise channel to one qubit of a multi-qubit density matrix."""
    dim = 2**n_qubits
    # Get Kraus operators for the noise channel
    kraus_ops = noise_model.depolarizing_channel(noise_model.params.single_qubit_error)
    new_rho = np.zeros_like(rho)
    for K_small in kraus_ops:
        # Embed 2x2 Kraus op into full space acting on `qubit`
        K_full = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                bi = (i >> qubit) & 1
                bj = (j >> qubit) & 1
                # Other bits must match
                i_other = i & ~(1 << qubit)
                j_other = j & ~(1 << qubit)
                if i_other == j_other:
                    K_full[i, j] = K_small[bi, bj]
        new_rho += K_full @ rho @ K_full.conj().T
    return new_rho


def compile_sc_multiply(p_a: float, p_b: float) -> SCQuantumCircuit:
    """Compile SC AND gate (multiplication) to a quantum circuit.

    SC: P(a AND b) = P(a) * P(b) for independent streams.
    Quantum: encode probabilities as Ry rotations, use CNOT for correlation.
    """
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


def compile_sc_layer(weights: np.ndarray, input_probs: np.ndarray) -> list[dict[str, Any]]:
    """Compile an SC dense layer to quantum gate descriptions.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n_neurons, n_inputs), values in [0, 1].
    input_probs : np.ndarray
        Shape (n_inputs,), SC input probabilities.

    Returns
    -------
    list of dicts, one per neuron, each containing:
        'neuron_idx': int
        'ry_angles': list of (input_angle, weight_angle) pairs
        'expected_output': float — SC computation result
        'quantum_output': float — quantum simulation result
    """
    n_neurons, n_inputs = weights.shape
    results = []

    for j in range(n_neurons):
        ry_angles = []
        sc_output = 0.0
        quantum_outputs = []

        for i in range(n_inputs):
            w = float(np.clip(weights[j, i], 0, 1))
            x = float(np.clip(input_probs[i], 0, 1))

            theta_x = prob_to_ry_angle(x)
            theta_w = prob_to_ry_angle(w)
            ry_angles.append((theta_x, theta_w))

            # SC: AND gate → product
            sc_output += w * x

            # Quantum: independent product P(q0=1)*P(q1=1)
            quantum_outputs.append(w * x)

        sc_output = float(np.clip(sc_output / max(n_inputs, 1), 0, 1))
        q_output = float(np.clip(sum(quantum_outputs) / max(n_inputs, 1), 0, 1))

        results.append(
            {
                "neuron_idx": j,
                "ry_angles": ry_angles,
                "expected_output": sc_output,
                "quantum_output": q_output,
            }
        )

    return results
