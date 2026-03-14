# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import numpy as np


class TestHeronNoiseModel:
    def test_default_params(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseParams

        p = HeronR2NoiseParams()
        assert p.cx_error == 0.005
        assert p.t1_us == 300.0

    def test_depolarizing_preserves_trace(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        model = HeronR2NoiseModel()
        kraus = model.depolarizing_channel(0.01)
        total = sum(K.conj().T @ K for K in kraus)
        np.testing.assert_allclose(total, np.eye(2), atol=1e-12)

    def test_amplitude_damping_kraus(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        model = HeronR2NoiseModel()
        kraus = model.amplitude_damping(0.1)
        total = sum(K.conj().T @ K for K in kraus)
        np.testing.assert_allclose(total, np.eye(2), atol=1e-12)

    def test_phase_damping_kraus(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        model = HeronR2NoiseModel()
        kraus = model.phase_damping(0.05)
        total = sum(K.conj().T @ K for K in kraus)
        np.testing.assert_allclose(total, np.eye(2), atol=1e-12)

    def test_noise_increases_error(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        model = HeronR2NoiseModel()
        pure = np.array([[1, 0], [0, 0]], dtype=complex)
        noisy = model.apply_single_qubit_noise(pure)
        # Pure state: Tr(ρ²) = 1, mixed: < 1
        purity = np.real(np.trace(noisy @ noisy))
        assert purity < 1.0

    def test_gate_fidelities(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        model = HeronR2NoiseModel()
        assert model.gate_fidelity_1q() > 0.999
        assert model.gate_fidelity_2q() > 0.99

    def test_readout_noise_0(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        model = HeronR2NoiseModel()
        results = [model.apply_readout_noise(0) for _ in range(1000)]
        assert 0 in results  # most should stay 0
        assert all(r in (0, 1) for r in results)

    def test_readout_noise_1(self):
        from sc_neurocore.quantum.noise_models import HeronR2NoiseModel

        model = HeronR2NoiseModel()
        results = [model.apply_readout_noise(1) for _ in range(1000)]
        assert 1 in results
        assert all(r in (0, 1) for r in results)


class TestParameterShift:
    def test_sin_gradient(self):
        from sc_neurocore.quantum.param_shift import parameter_shift_gradient

        def f(p):
            return np.sin(p[0])

        params = np.array([0.5])
        grad = parameter_shift_gradient(f, params)
        np.testing.assert_allclose(grad[0], np.cos(0.5), atol=1e-10)

    def test_multivariate(self):
        from sc_neurocore.quantum.param_shift import parameter_shift_gradient

        def f(p):
            return np.sin(p[0]) + np.cos(p[1])

        params = np.array([1.0, 2.0])
        grad = parameter_shift_gradient(f, params)
        np.testing.assert_allclose(grad[0], np.cos(1.0), atol=1e-10)
        np.testing.assert_allclose(grad[1], -np.sin(2.0), atol=1e-10)

    def test_optimizer_converges(self):
        from sc_neurocore.quantum.param_shift import ParameterShiftOptimizer

        def f(p):
            return (p[0] - 1.0) ** 2

        opt = ParameterShiftOptimizer(f, 1, lr=0.1)
        params = np.array([0.0])
        for _ in range(50):
            params = opt.step(params)
        assert abs(params[0] - 1.0) < 0.1


class TestHybridPipeline:
    def test_circuit_returns_scalar(self):
        from sc_neurocore.quantum.hybrid_pipeline import HybridQuantumClassicalPipeline

        pipe = HybridQuantumClassicalPipeline(n_qubits=2, n_layers=1)
        params = np.zeros(pipe.n_params)
        val = pipe.circuit(params)
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_vqe_converges(self):
        from sc_neurocore.quantum.hybrid_pipeline import HybridQuantumClassicalPipeline

        pipe = HybridQuantumClassicalPipeline(n_qubits=2, n_layers=1)
        history, params = pipe.train(n_steps=30, lr=0.05)
        assert history[-1] <= history[0] + 0.5

    def test_evaluate(self):
        from sc_neurocore.quantum.hybrid_pipeline import HybridQuantumClassicalPipeline

        pipe = HybridQuantumClassicalPipeline(n_qubits=2, n_layers=1)
        _, params = pipe.train(n_steps=10, lr=0.05)
        val = pipe.evaluate(params)
        assert isinstance(val, float)
