# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ry rotation gate

from __future__ import annotations

from typing import Any

import numpy as np
from .param_shift import parameter_shift_gradient


def _ry(theta: float) -> np.ndarray[Any, Any]:
    """Ry rotation gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _cnot() -> np.ndarray[Any, Any]:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )


def _kron_gate(gate: np.ndarray[Any, Any], qubit: int, n_qubits: int) -> np.ndarray[Any, Any]:
    """Embed single-qubit gate into n-qubit space."""
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


class HybridQuantumClassicalPipeline:
    def __init__(self, n_qubits: int = 2, n_layers: int = 1, noise_model: Any = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_model = noise_model
        self.n_params = n_qubits * n_layers

    def circuit(self, params: np.ndarray[Any, Any]) -> float:
        """Parameterized Ry-CNOT circuit → ⟨Z⊗Z⟩ expectation."""
        dim = 2**self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0  # |00...0⟩

        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                gate = _kron_gate(_ry(params[idx]), q, self.n_qubits)
                state = gate @ state
                idx += 1
            # CNOT chain
            if self.n_qubits >= 2:
                cnot = _cnot()
                for q in range(self.n_qubits - 1):
                    full = np.eye(dim, dtype=complex)
                    # Build CNOT on qubits q, q+1
                    sub = np.eye(dim, dtype=complex)
                    # Direct 2-qubit CNOT embedding
                    for i in range(dim):
                        for j in range(dim):
                            # Extract bits for qubits q and q+1
                            bq = (i >> (self.n_qubits - 1 - q)) & 1
                            bq1 = (i >> (self.n_qubits - 1 - q - 1)) & 1
                            if bq == 1:  # control set → flip target
                                flipped = i ^ (1 << (self.n_qubits - 1 - q - 1))
                                sub[flipped, i] = 1.0
                                sub[i, i] = 0.0
                    state = sub @ state

        # Measure ⟨Z⊗Z⟩ (product of Z eigenvalues on all qubits)
        z_all = np.array([(-1) ** bin(i).count("1") for i in range(dim)], dtype=float)
        return float(np.real(np.conj(state) @ (z_all * state)))

    def train(
        self, n_steps: int = 100, lr: float = 0.01
    ) -> tuple[list[float], np.ndarray[Any, Any]]:
        """VQE-style optimization: minimize ⟨Z⊗Z⟩."""
        params = np.random.randn(self.n_params) * 0.1
        history = []
        for _ in range(n_steps):
            val = self.circuit(params)
            history.append(val)
            grad = parameter_shift_gradient(self.circuit, params)
            params -= lr * grad
        return history, params

    def evaluate(self, params: np.ndarray[Any, Any]) -> float:
        return self.circuit(params)
