# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFusedKernel from former test_engine_v3_dense_kernels.py

"""Focused suite: TestFusedKernel from former test_engine_v3_dense_kernels.py."""

from __future__ import annotations

from tests.engine_v3_dense_kernels_support import *  # noqa: F403


class TestFusedKernel:
    """Verify fused encode+AND+popcount behavior and determinism."""

    def test_fused_matches_forward_fast(self) -> None:
        """Fused forward_fast output matches prepacked materialized encode path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
        seed = 123

        fused = np.asarray(layer.forward_fast(inputs.tolist(), seed=seed), dtype=np.float64)
        packed = v3.batch_encode_numpy(inputs, length=512, seed=seed)
        materialized = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)

        np.testing.assert_array_equal(fused, materialized)

    def test_fused_determinism(self) -> None:
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=777)
        out2 = layer.forward_fast(inputs, seed=777)
        np.testing.assert_array_equal(out1, out2)

    def test_fused_statistical_correctness(self) -> None:
        layer = v3.DenseLayer(16, 8, 2048, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=42))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=42))
        assert high > low
