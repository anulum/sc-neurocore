# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSIMDBernoulliEncode from former test_engine_v3_bitstream_kernels.py

"""Focused suite: TestSIMDBernoulliEncode from former test_engine_v3_bitstream_kernels.py."""

from __future__ import annotations

from tests.engine_v3_bitstream_kernels_support import *  # noqa: F403


class TestSIMDBernoulliEncode:
    """Verify SIMD Bernoulli encoder statistical correctness and determinism."""

    def test_batch_encode_statistics(self) -> None:
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_batch_encode_determinism(self) -> None:
        probs = np.array([0.15, 0.35, 0.55, 0.75], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        b = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        np.testing.assert_array_equal(a, b)

    def test_dense_fast_correctness(self) -> None:
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=22))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=22))
        assert high > low
