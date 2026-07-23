# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSIMDFusedAndPopcount from former test_engine_v3_dense_kernels.py

"""Focused suite: TestSIMDFusedAndPopcount from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

from tests.engine_v3_dense_kernels_support import *  # noqa: F403

class TestSIMDFusedAndPopcount:
    """Verify SIMD fused AND+popcount preserves dense behavior."""

    def test_dense_forward_unchanged(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        out1 = layer.forward(inputs, seed=123)
        out2 = layer.forward(inputs, seed=123)
        np.testing.assert_array_equal(out1, out2)
        assert all(0.0 <= x <= 8.0 for x in out1)

    def test_dense_prepacked_unchanged(self) -> None:
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_determinism(self) -> None:
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=77)
        out2 = layer.forward_fast(inputs, seed=77)
        np.testing.assert_array_equal(out1, out2)
