# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProbToQuantum from former test_tensor_stream.py

"""Focused suite: TestProbToQuantum from former test_tensor_stream.py."""

from __future__ import annotations

from tests.tensor_stream_support import *  # noqa: F403


class TestProbToQuantum:
    def test_output_shape(self):
        ts = TensorStream.from_prob(np.array([0.3, 0.7]))
        q = ts.to_quantum()
        assert q.shape == (2, 2)  # 2 qubits × [alpha, beta]

    def test_normalisation(self):
        """Quantum states must be normalised: |α|² + |β|² = 1."""
        ts = TensorStream.from_prob(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
        q = ts.to_quantum()
        norms = np.abs(q[:, 0]) ** 2 + np.abs(q[:, 1]) ** 2
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_born_rule(self):
        """P(|1⟩) = |β|² must equal the original probability."""
        probs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        ts = TensorStream.from_prob(probs)
        q = ts.to_quantum()
        born = np.abs(q[:, 1]) ** 2
        np.testing.assert_allclose(born, probs, atol=1e-10)

    def test_p_zero_gives_ground_state(self):
        """p=0 → |ψ⟩ = |0⟩ → α=1, β=0."""
        ts = TensorStream.from_prob(np.array([0.0]))
        q = ts.to_quantum()
        np.testing.assert_allclose(np.abs(q[0, 0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(q[0, 1]), 0.0, atol=1e-10)

    def test_p_one_gives_excited_state(self):
        """p=1 → |ψ⟩ = |1⟩ → α=0, β=1."""
        ts = TensorStream.from_prob(np.array([1.0]))
        q = ts.to_quantum()
        np.testing.assert_allclose(np.abs(q[0, 0]), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.abs(q[0, 1]), 1.0, atol=1e-10)
