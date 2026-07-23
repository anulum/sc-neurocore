# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCProbabilities from former test_quantizer.py

"""Focused suite: TestSCProbabilities from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403

class TestSCProbabilities:
    """Numerical safety of SC probability conversion."""

    def test_zero_maps_to_half(self):
        q = quantize_weights(np.array([0.0]), fmt="Q8.8")
        p = q_weights_to_sc_probabilities(q, fmt="Q8.8")
        np.testing.assert_allclose(p[0], 0.5, atol=0.001)

    def test_range_zero_one(self):
        w = np.linspace(-10, 10, 100)
        q = quantize_weights(w, fmt="Q8.8")
        p = q_weights_to_sc_probabilities(q, fmt="Q8.8")
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_sc_probabilities_are_finite_for_finite_inputs(self):
        w = np.array([0.0, -200.0, 200.0, 1.5, -1.5], dtype=np.float64)
        q88 = quantize_weights(w, fmt="Q8.8", clip=True)
        p88 = q_weights_to_sc_probabilities(q88, fmt="Q8.8")
        q16 = quantize_weights(w, fmt="Q16.16", clip=True)
        p16 = q_weights_to_sc_probabilities(q16, fmt="Q16.16")

        assert np.all(np.isfinite(p88))
        assert np.all(np.isfinite(p16))
        assert np.all((p88 >= 0.0) & (p88 <= 1.0))
        assert np.all((p16 >= 0.0) & (p16 <= 1.0))
