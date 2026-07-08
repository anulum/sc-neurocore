# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC→quantum circuit compiler

"""Tests for SC-to-quantum compilation (Conjecture C1+C4)."""

import numpy as np

import sc_neurocore.quantum as quantum
from sc_neurocore.quantum.sc_quantum_compiler import (
    sc_prob_to_statevector,
    statevector_to_prob,
    prob_to_ry_angle,
    ry_gate,
    compile_sc_multiply,
    compile_sc_layer,
)


def test_quantum_package_exports_sc_compiler_surface() -> None:
    """The quantum package facade exposes the documented SC compiler surface."""
    expected_names = {
        "QuantumGate",
        "SCQuantumCircuit",
        "compile_sc_layer",
        "compile_sc_multiply",
        "prob_to_ry_angle",
        "ry_gate",
        "sc_prob_to_statevector",
        "statevector_to_prob",
    }

    assert expected_names <= set(quantum.__all__)
    assert quantum.compile_sc_multiply is compile_sc_multiply
    assert quantum.compile_sc_layer is compile_sc_layer


class TestAmplitudeEncoding:
    """Amplitude encoding contracts for SC probabilities."""

    def test_zero_prob(self) -> None:
        sv = sc_prob_to_statevector(0.0)
        np.testing.assert_allclose(sv, [1.0, 0.0])

    def test_one_prob(self) -> None:
        sv = sc_prob_to_statevector(1.0)
        np.testing.assert_allclose(sv, [0.0, 1.0])

    def test_half_prob(self) -> None:
        sv = sc_prob_to_statevector(0.5)
        np.testing.assert_allclose(np.abs(sv) ** 2, [0.5, 0.5])

    def test_born_rule_roundtrip(self) -> None:
        """Encode probability → Born rule should recover it exactly."""
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            sv = sc_prob_to_statevector(p)
            p_recovered = statevector_to_prob(sv)
            np.testing.assert_allclose(p_recovered, p, atol=1e-12)

    def test_normalization(self) -> None:
        """State vector should be normalized."""
        for p in [0.1, 0.5, 0.9]:
            sv = sc_prob_to_statevector(p)
            assert np.abs(np.sum(np.abs(sv) ** 2) - 1.0) < 1e-12


class TestRyGate:
    """Ry gate construction and probability encoding contracts."""

    def test_ry_zero_is_identity(self) -> None:
        np.testing.assert_allclose(ry_gate(0.0), np.eye(2), atol=1e-12)

    def test_ry_pi_is_Y_rotation(self) -> None:
        """Ry(pi) should flip |0⟩ to |1⟩."""
        result = ry_gate(np.pi) @ np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(np.abs(result) ** 2, [0, 1], atol=1e-12)

    def test_ry_encodes_probability(self) -> None:
        """Ry(angle) applied to |0⟩ should give P(|1⟩) = p."""
        for p in [0.2, 0.5, 0.8]:
            theta = prob_to_ry_angle(p)
            sv = ry_gate(theta) @ np.array([1, 0], dtype=complex)
            np.testing.assert_allclose(np.abs(sv[1]) ** 2, p, atol=1e-12)


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


class TestOutputProbability:
    """Circuit output-probability helper contracts."""

    def test_output_probability_matches_manual(self) -> None:
        """output_probability should match P(output_qubit=|1⟩)."""
        circuit = compile_sc_multiply(0.6, 0.4)
        p = circuit.output_probability()
        # Marginal on q1: P(q1=1) = P(01) + P(11) = 0.4
        np.testing.assert_allclose(p, 0.4, atol=1e-10)


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


class TestNoisySimulation:
    """Noisy quantum simulation contracts."""

    def test_noisy_returns_density_matrix(self) -> None:
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        circuit = compile_sc_multiply(0.6, 0.7)
        noise = HeronR2NoiseModel()
        rho = circuit.simulate_noisy(noise)
        assert rho.shape == (4, 4)
        # Trace should be ~1
        np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-10)

    def test_noisy_probability_bounded(self) -> None:
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        circuit = compile_sc_multiply(0.5, 0.5)
        noise = HeronR2NoiseModel()
        prob = circuit.output_probability_noisy(noise, n_shots=500)
        assert 0.0 <= prob <= 1.0

    def test_noise_degrades_fidelity(self) -> None:
        """Noisy output should differ from ideal."""
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel, HeronR2NoiseParams

        circuit = compile_sc_multiply(0.6, 0.7)
        ideal_prob = circuit.output_probability()
        # High noise
        noisy_params = HeronR2NoiseParams(
            single_qubit_error=0.3, readout_0to1=0.1, readout_1to0=0.1
        )
        noise = HeronR2NoiseModel(noisy_params)
        rho = circuit.simulate_noisy(noise)
        noisy_prob = sum(
            float(np.real(rho[i, i])) for i in range(4) if (i >> circuit.output_qubit) & 1
        )
        # With 30% depolarizing error, noisy density matrix trace should still be ~1
        assert np.isclose(np.trace(rho).real, 1.0, atol=1e-10)
        # Noisy probability should be between 0 and 1
        assert 0.0 <= noisy_prob <= 1.0


class TestCompileSCLayer:
    """Dense-layer compiler output contracts."""

    def test_output_format(self) -> None:
        weights = np.array([[0.5, 0.3], [0.8, 0.2]])
        inputs = np.array([0.6, 0.4])
        results = compile_sc_layer(weights, inputs)
        assert len(results) == 2
        for r in results:
            assert "neuron_idx" in r
            assert "ry_angles" in r
            assert "expected_output" in r
            assert "quantum_output" in r

    def test_sc_quantum_equivalence(self) -> None:
        """SC and quantum outputs should match (both compute weighted mean)."""
        weights = np.array([[0.5, 0.3, 0.7]])
        inputs = np.array([0.6, 0.4, 0.8])
        results = compile_sc_layer(weights, inputs)
        r = results[0]
        np.testing.assert_allclose(r["expected_output"], r["quantum_output"], atol=1e-10)

    def test_all_outputs_bounded(self) -> None:
        rng = np.random.RandomState(42)
        weights = rng.uniform(0, 1, (4, 6))
        inputs = rng.uniform(0, 1, 6)
        results = compile_sc_layer(weights, inputs)
        for r in results:
            assert 0.0 <= r["expected_output"] <= 1.0
            assert 0.0 <= r["quantum_output"] <= 1.0
