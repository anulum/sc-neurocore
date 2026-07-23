# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFastPRNG from former test_engine_v3_bitstream_kernels.py

"""Focused suite: TestFastPRNG from former test_engine_v3_bitstream_kernels.py."""

from __future__ import annotations

from tests.engine_v3_bitstream_kernels_support import *  # noqa: F403

class TestFastPRNG:
    """Verify xoshiro-backed fast paths remain deterministic and statistically sane."""

    def test_xoshiro_determinism(self) -> None:
        probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        b = v3.batch_encode_numpy(probs, length=1024, seed=2026)
        np.testing.assert_array_equal(a, b)

    def test_xoshiro_statistical_quality(self) -> None:
        probs = np.array([0.35], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=1337)
        count = sum(int(w).bit_count() for w in packed[0])
        measured = count / 10_000
        assert abs(measured - 0.35) < 0.03

    def test_forward_fast_determinism_new(self) -> None:
        layer = v3.DenseLayer(12, 6, 1024, seed=42)
        inputs = np.linspace(0.05, 0.95, 12, dtype=np.float64)
        a = layer.forward_fast(inputs.tolist(), seed=98765)
        b = layer.forward_fast(inputs.tolist(), seed=98765)
        np.testing.assert_array_equal(a, b)
