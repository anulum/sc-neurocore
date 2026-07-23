# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHeronNoiseModel from former test_quantum_stabilisation.py

"""Focused suite: TestHeronNoiseModel from former test_quantum_stabilisation.py."""

from __future__ import annotations

from tests.quantum_stabilisation_support import *  # noqa: F403

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
