# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTwoQubitGate from former test_sc_quantum_compiler.py

"""Focused suite: TestTwoQubitGate from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403


class TestTwoQubitGate:
    """Generic gate application contracts used by the compiler."""

    def test_cnot_flips(self) -> None:
        """CNOT with control=|1⟩ should flip target."""
        from sc_neurocore.quantum.sc_quantum_compiler import (
            _X,
            QuantumGate,
            SCQuantumCircuit,
        )

        cnot = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
        # |10⟩ → CNOT → |11⟩
        circuit = SCQuantumCircuit(
            n_qubits=2,
            gates=[
                QuantumGate("X", _X, [0]),  # put q0 in |1⟩
                QuantumGate("CNOT", cnot, [0, 1]),  # flip q1
            ],
            input_qubits=[0],
            output_qubit=1,
        )
        state = circuit.simulate()
        # Should be |11⟩ = index 3
        np.testing.assert_allclose(np.abs(state[3]) ** 2, 1.0, atol=1e-10)

    def test_three_qubit_gate_raises(self) -> None:
        from sc_neurocore.quantum.sc_quantum_compiler import _apply_gate
        import pytest

        state = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
        gate = np.eye(8, dtype=complex)
        with pytest.raises(ValueError, match="not supported"):
            _apply_gate(state, gate, [0, 1, 2], 3)
