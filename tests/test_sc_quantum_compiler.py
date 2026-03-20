# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC→quantum circuit compiler

"""Tests for SC-to-quantum compilation (Conjecture C1+C4)."""

import numpy as np
import pytest

from sc_neurocore.quantum.sc_quantum_compiler import (
    sc_prob_to_statevector,
    statevector_to_prob,
    prob_to_ry_angle,
    ry_gate,
    compile_sc_multiply,
    compile_sc_layer,
    SCQuantumCircuit,
    QuantumGate,
)


class TestAmplitudeEncoding:
    def test_zero_prob(self):
        sv = sc_prob_to_statevector(0.0)
        np.testing.assert_allclose(sv, [1.0, 0.0])

    def test_one_prob(self):
        sv = sc_prob_to_statevector(1.0)
        np.testing.assert_allclose(sv, [0.0, 1.0])

    def test_half_prob(self):
        sv = sc_prob_to_statevector(0.5)
        np.testing.assert_allclose(np.abs(sv) ** 2, [0.5, 0.5])

    def test_born_rule_roundtrip(self):
        """Encode probability → Born rule should recover it exactly."""
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            sv = sc_prob_to_statevector(p)
            p_recovered = statevector_to_prob(sv)
            np.testing.assert_allclose(p_recovered, p, atol=1e-12)

    def test_normalization(self):
        """State vector should be normalized."""
        for p in [0.1, 0.5, 0.9]:
            sv = sc_prob_to_statevector(p)
            assert np.abs(np.sum(np.abs(sv) ** 2) - 1.0) < 1e-12


class TestRyGate:
    def test_ry_zero_is_identity(self):
        np.testing.assert_allclose(ry_gate(0.0), np.eye(2), atol=1e-12)

    def test_ry_pi_is_Y_rotation(self):
        """Ry(pi) should flip |0⟩ to |1⟩."""
        result = ry_gate(np.pi) @ np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(np.abs(result) ** 2, [0, 1], atol=1e-12)

    def test_ry_encodes_probability(self):
        """Ry(angle) applied to |0⟩ should give P(|1⟩) = p."""
        for p in [0.2, 0.5, 0.8]:
            theta = prob_to_ry_angle(p)
            sv = ry_gate(theta) @ np.array([1, 0], dtype=complex)
            np.testing.assert_allclose(np.abs(sv[1]) ** 2, p, atol=1e-12)


class TestSCMultiplyCircuit:
    def test_product_probability(self):
        """SC AND = multiply: P(a)*P(b) should match quantum simulation."""
        circuit = compile_sc_multiply(0.6, 0.7)
        # For independent qubits: P(q0=1 AND q1=1) = P(q0=1) * P(q1=1) = 0.42
        state = circuit.simulate()
        # |11⟩ is at index 3 (binary 11)
        p_11 = np.abs(state[3]) ** 2
        np.testing.assert_allclose(p_11, 0.6 * 0.7, atol=1e-10)

    def test_zero_times_anything(self):
        circuit = compile_sc_multiply(0.0, 0.8)
        state = circuit.simulate()
        p_11 = np.abs(state[3]) ** 2
        np.testing.assert_allclose(p_11, 0.0, atol=1e-10)

    def test_one_times_p(self):
        circuit = compile_sc_multiply(1.0, 0.3)
        state = circuit.simulate()
        p_11 = np.abs(state[3]) ** 2
        np.testing.assert_allclose(p_11, 0.3, atol=1e-10)

    def test_summary(self):
        circuit = compile_sc_multiply(0.5, 0.5)
        s = circuit.summary()
        assert "SCQuantumCircuit" in s
        assert "Ry" in s


class TestCompileSCLayer:
    def test_output_format(self):
        weights = np.array([[0.5, 0.3], [0.8, 0.2]])
        inputs = np.array([0.6, 0.4])
        results = compile_sc_layer(weights, inputs)
        assert len(results) == 2
        for r in results:
            assert "neuron_idx" in r
            assert "ry_angles" in r
            assert "expected_output" in r
            assert "quantum_output" in r

    def test_sc_quantum_equivalence(self):
        """SC and quantum outputs should match (both compute weighted mean)."""
        weights = np.array([[0.5, 0.3, 0.7]])
        inputs = np.array([0.6, 0.4, 0.8])
        results = compile_sc_layer(weights, inputs)
        r = results[0]
        np.testing.assert_allclose(
            r["expected_output"], r["quantum_output"], atol=1e-10
        )

    def test_all_outputs_bounded(self):
        rng = np.random.RandomState(42)
        weights = rng.uniform(0, 1, (4, 6))
        inputs = rng.uniform(0, 1, 6)
        results = compile_sc_layer(weights, inputs)
        for r in results:
            assert 0.0 <= r["expected_output"] <= 1.0
            assert 0.0 <= r["quantum_output"] <= 1.0
