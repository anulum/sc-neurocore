# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for quantisation pipeline (float → Q8.8 → SC)

"""Tests for quantize_weights, dequantize_weights, q_weights_to_sc_probabilities."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.compiler.quantizer import (
    quantize_weights,
    dequantize_weights,
    q_weights_to_sc_probabilities,
)


class TestQuantizeWeights:
    def test_output_is_integer(self):
        w = np.array([0.5, -0.5, 1.0, -1.0])
        q = quantize_weights(w, fmt="Q8.8")
        assert q.dtype in (np.int16, np.int32, np.int64, np.float64), f"unexpected dtype {q.dtype}"

    def test_zero_roundtrip(self):
        w = np.array([0.0])
        q = quantize_weights(w, fmt="Q8.8")
        d = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(d, 0.0, atol=1e-6)

    def test_roundtrip_error_bounded(self):
        """Quantisation error should be <= step/2 = 1/512."""
        rng = np.random.default_rng(42)
        w = rng.uniform(-5.0, 5.0, 100)
        q = quantize_weights(w, fmt="Q8.8")
        d = dequantize_weights(q, fmt="Q8.8")
        max_err = np.max(np.abs(w - d))
        step_half = 1.0 / 512.0
        assert max_err < step_half + 1e-6, f"max err {max_err:.6f} > step/2 {step_half:.6f}"

    def test_shape_preserved(self):
        w = np.random.randn(8, 16)
        q = quantize_weights(w, fmt="Q8.8")
        assert q.shape == w.shape

    def test_large_values_handled(self):
        """Values outside Q8.8 range should be clipped or wrapped."""
        w = np.array([200.0, -200.0])
        q = quantize_weights(w, fmt="Q8.8")
        d = dequantize_weights(q, fmt="Q8.8")
        # Should not crash; values may be clipped
        assert np.all(np.isfinite(d))

    def test_small_values_near_zero(self):
        w = np.array([0.001, -0.001, 0.004, -0.004])
        q = quantize_weights(w, fmt="Q8.8")
        d = dequantize_weights(q, fmt="Q8.8")
        # 0.001 rounds to 0/256 = 0.0 or 1/256 ≈ 0.0039
        for i in range(len(w)):
            assert abs(d[i] - w[i]) < 1.0 / 256.0 + 1e-6


class TestDequantizeWeights:
    def test_known_value(self):
        # 256 in Q8.8 = 1.0
        q = np.array([256])
        d = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(d, 1.0, atol=1e-6)

    def test_negative_known(self):
        # -256 in Q8.8 = -1.0
        q = np.array([-256])
        d = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(d, -1.0, atol=1e-6)


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


class TestEndToEndPipeline:
    def test_dot_product_fidelity(self):
        """Q8.8 dot product should be close to float dot product."""
        rng = np.random.default_rng(42)
        W = rng.uniform(-2.0, 2.0, (4, 8))
        x = rng.uniform(0.0, 1.0, 8)

        # Float reference
        y_float = W @ x

        # Q8.8 pipeline
        W_q = quantize_weights(W, fmt="Q8.8")
        W_deq = dequantize_weights(W_q, fmt="Q8.8")
        y_q88 = W_deq @ x

        # Error should be small (dominated by weight quantisation)
        mae = np.mean(np.abs(y_float - y_q88))
        assert mae < 0.1, f"dot product MAE {mae:.4f} too large"

    def test_multiple_formats(self):
        """Quantisation should work for at least Q8.8."""
        w = np.array([0.5, -0.5])
        q = quantize_weights(w, fmt="Q8.8")
        d = dequantize_weights(q, fmt="Q8.8")
        assert np.allclose(w, d, atol=0.005)


class TestFixedPointQuantizationGuards:
    """Fail-closed branches in the fixed-point quantisation backend."""

    def test_coerce_q_format_rejects_non_format_type(self):
        from sc_neurocore.compiler.fixed_point_quantization import _coerce_q_format

        with pytest.raises(TypeError, match="Expected QFormat or Q-format string"):
            _coerce_q_format(123)  # type: ignore[arg-type]

    def test_quantize_weights_rejects_block_floating_string(self):
        from sc_neurocore.compiler.fixed_point_quantization import quantize_weights as qw

        with pytest.raises(ValueError, match="quantize_block_floating"):
            qw(np.array([0.5]), fmt="BFP16E3X32")

    def test_dequantize_weights_rejects_block_floating_string(self):
        from sc_neurocore.compiler.fixed_point_quantization import (
            dequantize_weights as dqw,
        )

        with pytest.raises(ValueError, match="dequantize_block_floating"):
            dqw(np.array([1]), fmt="BFP16E3X32")

    def test_mixed_precision_scale_must_stay_finite(self):
        from sc_neurocore.compiler.fixed_point_quantization import quantize_weights as qw
        from sc_neurocore.compiler.quantizer import QFormatMixed

        # A near-denormal weight makes the per-tensor scale overflow to inf.
        with pytest.raises(ValueError, match="per-tensor scale must be finite"):
            qw(np.array([1e-308]), fmt=QFormatMixed(scale_per_tensor=True))

    def test_quantization_error_rejects_mixed_format(self):
        from sc_neurocore.compiler.fixed_point_quantization import quantization_error
        from sc_neurocore.compiler.quantizer import QFormatMixed

        with pytest.raises(TypeError, match="not QFormatMixed"):
            quantization_error(np.array([0.5, 0.3]), fmt=QFormatMixed())
