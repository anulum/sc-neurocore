# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCMultiplyCircuit from former test_sc_quantum_compiler.py

"""Focused suite: TestSCMultiplyCircuit from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403


class TestSCMultiplyCircuit:
    """SC multiply compilation contracts."""

    def test_product_probability(self) -> None:
        """SC AND = multiply: P(a)*P(b) should match quantum simulation."""
        circuit = compile_sc_multiply(0.6, 0.7)
        # For independent qubits: P(q0=1 AND q1=1) = P(q0=1) * P(q1=1) = 0.42
        state = circuit.simulate()
        # |11⟩ is at index 3 (binary 11)
        p_11 = np.abs(state[3]) ** 2
        np.testing.assert_allclose(p_11, 0.6 * 0.7, atol=1e-10)

    def test_zero_times_anything(self) -> None:
        circuit = compile_sc_multiply(0.0, 0.8)
        state = circuit.simulate()
        p_11 = np.abs(state[3]) ** 2
        np.testing.assert_allclose(p_11, 0.0, atol=1e-10)

    def test_one_times_p(self) -> None:
        circuit = compile_sc_multiply(1.0, 0.3)
        state = circuit.simulate()
        p_11 = np.abs(state[3]) ** 2
        np.testing.assert_allclose(p_11, 0.3, atol=1e-10)

    def test_summary(self) -> None:
        circuit = compile_sc_multiply(0.5, 0.5)
        s = circuit.summary()
        assert "SCQuantumCircuit" in s
        assert "Ry" in s
