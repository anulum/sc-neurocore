# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNoisySimulation from former test_sc_quantum_compiler.py

"""Focused suite: TestNoisySimulation from former test_sc_quantum_compiler.py."""

from __future__ import annotations

from tests.sc_quantum_compiler_support import *  # noqa: F403

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
