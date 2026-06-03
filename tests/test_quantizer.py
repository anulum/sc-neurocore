# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for weight quantizer

"""Tests for quantizer: float weights → Q-format fixed-point → SC probabilities."""

import numpy as np
import pytest

from sc_neurocore.compiler.quantizer import (
    QFormat,
    BlockFloatingMode,
    parse_precision_format,
    quantize_block_floating,
    dequantize_block_floating,
    quantize_weights,
    dequantize_weights,
    q_weights_to_sc_probabilities,
    quantization_error,
)


class TestQFormat:
    def test_parse_q88(self):
        q = QFormat.from_string("Q8.8")
        assert q.integer_bits == 8
        assert q.fraction_bits == 8
        assert q.total_bits == 16
        assert q.scale == 256

    def test_parse_q4_12(self):
        q = QFormat.from_string("Q4.12")
        assert q.total_bits == 16
        assert q.scale == 4096

    def test_range_q88(self):
        q = QFormat.from_string("Q8.8")
        assert q.min_val == -128.0
        assert q.max_val == pytest.approx(127.99609375)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            QFormat.from_string("float32")


class TestPrecisionFormatParser:
    """Test format parsing for fixed and block-floating modes."""

    def test_parse_block_floating_alias(self):
        fmt = parse_precision_format("BFP16E3X32")
        assert isinstance(fmt, BlockFloatingMode)
        assert fmt.mantissa_bits == 16
        assert fmt.exponent_bits == 3
        assert fmt.block_size == 32

    def test_parse_block_floating_flexible_alias(self):
        fmt = parse_precision_format("bfp16_e3")
        assert isinstance(fmt, BlockFloatingMode)
        assert fmt.mantissa_bits == 16
        assert fmt.exponent_bits == 3

    def test_parse_block_floating_dash_alias(self):
        fmt = parse_precision_format("BFP16-3x32")
        assert isinstance(fmt, BlockFloatingMode)
        assert fmt.block_size == 32


class TestPrecisionMetadata:
    """Validate precision-format parse coverage for wide fixed-point formats."""

    def test_parse_q16_16(self):
        q = QFormat.from_string("Q16.16")
        assert q.integer_bits == 16
        assert q.fraction_bits == 16
        assert q.total_bits == 32
        assert q.scale == 65536


class TestBlockFloatingQuantize:
    """Validate block-floating quantization contracts."""

    def test_quantize_block_floating_roundtrip(self):
        w = np.array([0.0, 0.25, -0.5, 0.75, -1.0, 1.0], dtype=np.float64)
        q, exponents = quantize_block_floating(w, fmt="BFP12E4X4", block_size=4, clip=True)
        recovered = dequantize_block_floating(q, exponents, fmt="BFP12E4X4")
        np.testing.assert_allclose(recovered, w, rtol=0.0, atol=0.02)

    def test_quantize_block_floating_block_size_conflict(self):
        w = np.array([1.0, 0.5, -0.25])
        with pytest.raises(ValueError, match="Block size conflict"):
            quantize_block_floating(w, fmt="BFP12E4X8", block_size=4)

    def test_quantize_block_floating_overflow_boundary_is_finite(self):
        fmt = "BFP8E2X2"
        mode = BlockFloatingMode.from_aliases(fmt)
        w = np.array([0.0, 7.0, -7.0, 112.0, -112.0, 120.0, -120.0], dtype=np.float64)
        q, exponents = quantize_block_floating(w, fmt=fmt, block_size=2, clip=True)

        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(exponents))
        assert np.all(np.abs(q) <= mode.mantissa_range)
        assert np.all(exponents >= 0)
        assert np.all(exponents <= (1 << mode.exponent_bits) - 1)

        restored = dequantize_block_floating(q, exponents, fmt=fmt)
        assert np.all(np.isfinite(restored))
        assert restored.shape == w.shape


class TestQuantizeWeights:
    def test_roundtrip_identity(self):
        w = np.array([0.0, 0.5, 1.0, -1.0, -0.5])
        q = quantize_weights(w, fmt="Q8.8")
        r = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(r, w, atol=1 / 256)

    def test_nearest_rounding(self):
        # np.rint uses banker's rounding: 128.5 → 128 (round half to even)
        w = np.array([0.501953125])  # 128.5 / 256
        q = quantize_weights(w, fmt="Q8.8", rounding="nearest")
        assert q[0] == 128  # banker's rounding: 128.5 → 128 (even)

    def test_nearest_rounding_odd(self):
        # 129.5 → 130 (round half to even, 130 is even)
        w = np.array([129.5 / 256])
        q = quantize_weights(w, fmt="Q8.8", rounding="nearest")
        assert q[0] == 130

    def test_floor_rounding(self):
        w = np.array([0.501953125])
        q = quantize_weights(w, fmt="Q8.8", rounding="floor")
        assert q[0] == 128

    def test_stochastic_rounding_average(self):
        np.random.seed(42)
        w = np.array([0.501953125] * 10000)
        q = quantize_weights(w, fmt="Q8.8", rounding="stochastic")
        # Stochastic: half round up, half round down on average
        assert 128 < q.mean() < 129

    def test_clipping(self):
        w = np.array([200.0, -200.0])
        q = quantize_weights(w, fmt="Q8.8", clip=True)
        assert q[0] == 32767  # max Q8.8
        assert q[1] == -32768  # min Q8.8

    def test_invalid_rounding_raises(self):
        with pytest.raises(ValueError, match="Unknown rounding"):
            quantize_weights(np.array([1.0]), rounding="random")


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


class TestQuantizationError:
    def test_error_stats(self):
        w = np.random.randn(100)
        stats = quantization_error(w, fmt="Q8.8")
        assert stats["max_abs_error"] < 1 / 256 + 1e-9
        assert stats["mean_abs_error"] < 1 / 256
        assert stats["rmse"] > 0
        assert stats["snr_db"] > 30  # good SNR for Q8.8

    def test_higher_precision_lower_error(self):
        w = np.random.randn(100)
        e88 = quantization_error(w, fmt="Q8.8")
        e412 = quantization_error(w, fmt="Q4.12")
        assert e412["rmse"] < e88["rmse"]

    def test_q16_16_dominates_q8_8(self):
        w = np.random.RandomState(0).normal(size=1000)
        e88 = quantization_error(w, fmt="Q8.8")
        e1616 = quantization_error(w, fmt="Q16.16")
        assert e1616["rmse"] < e88["rmse"]
        assert e1616["max_abs_error"] < e88["max_abs_error"]
        assert e1616["snr_db"] > e88["snr_db"]
