# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRayonThreshold from former test_engine_v3_dense_kernels.py

"""Focused suite: TestRayonThreshold from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

from tests.engine_v3_dense_kernels_support import *  # noqa: F403


class TestRayonThreshold:
    """Test that rayon threshold does not change forward_fast outputs."""

    def test_forward_fast_determinism(self) -> None:
        """forward_fast with small inputs (below threshold) stays deterministic."""
        layer = v3.DenseLayer(16, 8, 1024)
        inputs = [0.5] * 16
        a = layer.forward_fast(inputs, seed=42)
        b = layer.forward_fast(inputs, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_forward_fast_consistent_across_sizes(self) -> None:
        """forward_fast produces valid outputs for various input sizes."""
        for n_in in [4, 16, 64, 128, 256]:
            layer = v3.DenseLayer(n_in, 8, 1024)
            inputs = [0.5] * n_in
            result = layer.forward_fast(inputs, seed=42)
            assert len(result) == 8
            for val in result:
                assert 0.0 <= val <= float(n_in), f"Out of range: {val}"
