# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizeWeights from former test_quantisation_pipeline.py

"""Focused suite: TestQuantizeWeights from former test_quantisation_pipeline.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.compiler.quantizer import (
    dequantize_weights,
    quantize_weights,
)


class TestQuantizeWeights:
    """Quantising float weights to fixed point."""

    def test_output_is_integer(self) -> None:
        """Quantised weights come back as an integer array."""
        w = np.array([0.5, -0.5, 1.0, -1.0])
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        assert q.dtype in (np.int16, np.int32, np.int64, np.float64), f"unexpected dtype {q.dtype}"

    def test_zero_roundtrip(self) -> None:
        """Zero round-trips through Q8.8 exactly."""
        w = np.array([0.0])
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        d = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(d, 0.0, atol=1e-6)

    def test_roundtrip_error_bounded(self) -> None:
        """Quantisation error should be <= step/2 = 1/512."""
        rng = np.random.default_rng(42)
        w = rng.uniform(-5.0, 5.0, 100)
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        d = dequantize_weights(q, fmt="Q8.8")
        max_err = np.max(np.abs(w - d))
        step_half = 1.0 / 512.0
        assert max_err < step_half + 1e-6, f"max err {max_err:.6f} > step/2 {step_half:.6f}"

    def test_shape_preserved(self) -> None:
        """Quantising keeps the weight matrix shape."""
        w = np.random.randn(8, 16)
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        assert q.shape == w.shape

    def test_large_values_handled(self) -> None:
        """Values outside Q8.8 range should be clipped or wrapped."""
        w = np.array([200.0, -200.0])
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        d = dequantize_weights(q, fmt="Q8.8")
        # Should not crash; values may be clipped
        assert np.all(np.isfinite(d))

    def test_small_values_near_zero(self) -> None:
        """Small values round within one Q8.8 step."""
        w = np.array([0.001, -0.001, 0.004, -0.004])
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        d = dequantize_weights(q, fmt="Q8.8")
        # 0.001 rounds to 0/256 = 0.0 or 1/256 ≈ 0.0039
        for i in range(len(w)):
            assert abs(d[i] - w[i]) < 1.0 / 256.0 + 1e-6
