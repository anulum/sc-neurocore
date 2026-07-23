# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFLIFGLCoefficients from former test_model_fractional_lif.py

"""Focused suite: TestFLIFGLCoefficients from former test_model_fractional_lif.py."""

from __future__ import annotations

from tests.model_fractional_lif_support import *  # noqa: F403

class TestFLIFGLCoefficients:
    """Grünwald-Letnikov coefficients: c[0]=1, c[k] = c[k-1]·(k-1-α)/k."""

    def test_first_coefficient_is_one(self):
        n = FractionalLIFNeuron()
        assert n._gl_coeffs[0] == 1.0

    def test_gl_recurrence(self):
        """c[k] = c[k-1] · (k-1-alpha) / k."""
        n = FractionalLIFNeuron(alpha=0.8)
        for k in range(1, 10):
            expected = n._gl_coeffs[k - 1] * (k - 1 - 0.8) / k
            assert abs(n._gl_coeffs[k] - expected) < 1e-12

    def test_alpha_1_reduces_to_lif(self):
        """At α=1: GL coeffs → [1, 0, 0, ...] (standard derivative)."""
        n = FractionalLIFNeuron(alpha=1.0)
        assert n._gl_coeffs[0] == 1.0
        # c[1] = 1 * (0 - 1) / 1 = -1
        assert abs(n._gl_coeffs[1] - (-1.0)) < 1e-12

    def test_alpha_1_step_matches_euler_lif_update(self):
        n = FractionalLIFNeuron(v=0.25, alpha=1.0, dt=0.1)

        spike = n.step(0.5)

        assert spike == 0
        assert n.v == pytest.approx(0.25 + (-0.25 + 0.5) * 0.1)

    def test_alpha_affects_memory_depth(self):
        """Lower α → slower coefficient decay → longer effective memory."""
        n_low = FractionalLIFNeuron(alpha=0.5)
        n_high = FractionalLIFNeuron(alpha=0.9)
        # At k=50: low alpha should have larger |coeff|
        assert abs(n_low._gl_coeffs[50]) > abs(n_high._gl_coeffs[50])

    def test_history_buffer_length(self):
        n = FractionalLIFNeuron()
        for _ in range(200):
            n.step(5.0)
        assert len(n._history) == n._max_history
