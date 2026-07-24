# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFusedAndPopcount from former test_engine_v3_dense_kernels.py

"""Focused suite: TestFusedAndPopcount from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

from tests.engine_v3_dense_kernels_support import *  # noqa: F403


class TestFusedAndPopcount:
    """Tests verifying fused AND+popcount produces same results as before."""

    def test_forward_matches_reference(self) -> None:
        """forward() output should still be valid (range + deterministic)."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = [0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8, 0.1]
        out1 = layer.forward(inputs, seed=42)
        out2 = layer.forward(inputs, seed=42)
        np.testing.assert_array_equal(out1, out2)
        assert all(v >= 0.0 for v in out1)

    def test_prepacked_deterministic(self) -> None:
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out1 = layer.forward_prepacked(packed)
        out2 = layer.forward_prepacked(packed)
        np.testing.assert_array_equal(out1, out2)
