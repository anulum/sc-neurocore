# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantumEntropySource from former test_sources_and_profiling.py

"""Focused suite: TestQuantumEntropySource from former test_sources_and_profiling.py."""

from __future__ import annotations

from tests.sources_and_profiling_support import *  # noqa: F403

class TestQuantumEntropySource:
    def test_construction_default(self):
        qes = QuantumEntropySource(n_qubits=1, seed=42)
        assert qes.state.shape == (2,)
        assert qes.state[0] == 1.0 + 0j

    def test_construction_multi_qubit(self):
        qes = QuantumEntropySource(n_qubits=3, seed=0)
        assert qes.state.shape == (8,)  # 2^3

    def test_sample_normal_returns_float(self):
        qes = QuantumEntropySource(n_qubits=1, seed=0)
        val = qes.sample_normal()
        assert isinstance(val, float)

    def test_sample_returns_float(self):
        qes = QuantumEntropySource(n_qubits=1, seed=0)
        assert isinstance(qes.sample(), float)

    def test_hadamard_normalizes_state(self):
        """After Hadamard, state should remain normalised (sum |a|^2 = 1)."""
        qes = QuantumEntropySource(n_qubits=2, seed=42)
        qes._hadamard()
        norm = np.sum(np.abs(qes.state) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_samples_vary(self):
        """Repeated samples should not all be identical."""
        qes = QuantumEntropySource(n_qubits=1, seed=42)
        samples = [qes.sample_normal() for _ in range(50)]
        assert len(set(samples)) > 1

    def test_mean_std_scaling(self):
        """With many samples, mean and spread should roughly match request."""
        qes = QuantumEntropySource(n_qubits=3, seed=0)
        samples = np.array([qes.sample_normal(mean=5.0, std=2.0) for _ in range(2000)])
        # Not Gaussian, but mean should be near 5 and spread should be > 0
        assert abs(samples.mean() - 5.0) < 2.0
        assert samples.std() > 0.1

    def test_reproducible_with_seed(self):
        qes1 = QuantumEntropySource(n_qubits=1, seed=99)
        qes2 = QuantumEntropySource(n_qubits=1, seed=99)
        s1 = [qes1.sample() for _ in range(10)]
        s2 = [qes2.sample() for _ in range(10)]
        assert s1 == s2
