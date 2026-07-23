# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCProbabilityMapping from former test_quantisation_pipeline.py

"""Focused suite: TestSCProbabilityMapping from former test_quantisation_pipeline.py."""

from __future__ import annotations

from tests.quantisation_pipeline_support import *  # noqa: F403

class TestSCProbabilityMapping:
    def test_output_in_zero_one(self):
        rng = np.random.default_rng(42)
        w = rng.uniform(-3.0, 3.0, 50)
        q = quantize_weights(w, fmt="Q8.8")
        sc = q_weights_to_sc_probabilities(q, fmt="Q8.8")
        assert np.all(sc >= 0.0), f"min SC prob {sc.min():.4f} < 0"
        assert np.all(sc <= 1.0), f"max SC prob {sc.max():.4f} > 1"

    def test_preserves_ordering(self):
        """Larger Q8.8 value → larger SC probability."""
        w = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        q = quantize_weights(w, fmt="Q8.8")
        sc = q_weights_to_sc_probabilities(q, fmt="Q8.8")
        for i in range(len(sc) - 1):
            assert sc[i] <= sc[i + 1] + 1e-6, f"ordering violated at {i}"

    def test_shape_preserved(self):
        w = np.random.randn(4, 8)
        q = quantize_weights(w, fmt="Q8.8")
        sc = q_weights_to_sc_probabilities(q, fmt="Q8.8")
        assert sc.shape == w.shape

    def test_zero_maps_to_middle(self):
        """0.0 in Q8.8 = integer 0 should map to ~0.5 SC probability."""
        q = np.array([0])
        sc = q_weights_to_sc_probabilities(q, fmt="Q8.8")
        # Exact midpoint depends on range mapping
        assert 0.3 < sc[0] < 0.7, f"zero mapped to {sc[0]:.3f}, expected ~0.5"
