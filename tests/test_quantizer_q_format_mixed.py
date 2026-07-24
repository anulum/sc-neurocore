# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQFormatMixed from former test_quantizer.py

"""Focused suite: TestQFormatMixed from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403


class TestQFormatMixed:
    """Validate the Q8.8-weight/Q16.16-accumulator contract."""

    def test_default_contract_metadata(self):
        fmt = QFormatMixed()
        assert fmt.weight_fmt == Q8_8
        assert fmt.accum_fmt == Q16_16
        assert fmt.accumulator_guard_bits == 16
        assert fmt.metadata == {
            "kind": "mixed_fixed_point",
            "weight_format": "Q8.8",
            "accumulator_format": "Q16.16",
            "weight_total_bits": 16,
            "accumulator_total_bits": 32,
            "accumulator_guard_bits": 16,
            "scale_per_tensor": True,
            "rounding": "nearest",
        }

    def test_rejects_accumulator_that_cannot_hold_weight_domain(self):
        with pytest.raises(ValueError, match="dynamic range"):
            QFormatMixed(weight_fmt=Q8_8, accum_fmt=QFormat(4, 12))

    def test_rejects_accumulator_with_less_fractional_precision(self):
        with pytest.raises(ValueError, match="fractional precision"):
            QFormatMixed(weight_fmt=QFormat(8, 12), accum_fmt=QFormat(16, 8))

    def test_fixed_qformat_object_matches_string_quantisation(self):
        weights = np.array([-1.25, -0.5, 0.0, 0.5, 1.25], dtype=np.float64)
        by_object = quantize_weights(weights, fmt=Q8_8)
        by_string = quantize_weights(weights, fmt="Q8.8")
        np.testing.assert_array_equal(by_object, by_string)

    def test_mixed_quantisation_uses_full_tensor_dynamic_range(self):
        weights = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float64)
        fmt = QFormatMixed()
        quantised, tensor_scale = quantize_weights(weights, fmt=fmt)

        assert tensor_scale > 1.0
        assert int(np.max(np.abs(quantised))) == 32767

        restored = dequantize_weights(quantised, fmt=fmt, scale=tensor_scale)
        np.testing.assert_allclose(restored, weights, atol=0.5 / (Q8_8.scale * tensor_scale))

    def test_mixed_without_tensor_scale_uses_canonical_q88_codes(self):
        weights = np.array([0.5, -0.5, 1.25], dtype=np.float64)
        fmt = QFormatMixed(scale_per_tensor=False)
        quantised, tensor_scale = quantize_weights(weights, fmt=fmt)

        assert tensor_scale == 1.0
        np.testing.assert_array_equal(quantised, np.array([128, -128, 320]))
        restored = dequantize(quantised, fmt=fmt, scale=tensor_scale)
        np.testing.assert_allclose(restored, weights, atol=1 / Q8_8.scale)

    def test_zero_tensor_round_trips_with_unit_scale(self):
        weights = np.zeros((2, 3), dtype=np.float64)
        fmt = QFormatMixed()
        quantised, tensor_scale = quantize_weights(weights, fmt=fmt)

        assert tensor_scale == 1.0
        np.testing.assert_array_equal(quantised, np.zeros((2, 3), dtype=np.int64))
        np.testing.assert_array_equal(
            dequantize_weights(quantised, fmt=fmt, scale=tensor_scale), weights
        )

    def test_mixed_quantisation_rejects_non_finite_weights(self):
        with pytest.raises(ValueError, match="finite values"):
            quantize_weights(np.array([0.0, np.inf]), fmt=QFormatMixed())

    def test_dequantise_rejects_invalid_scale(self):
        with pytest.raises(ValueError, match="finite and positive"):
            dequantize_weights(np.array([1]), fmt=QFormatMixed(), scale=0.0)

    @given(
        values=st.lists(
            st.floats(
                min_value=-512.0,
                max_value=512.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=1,
            max_size=32,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_mixed_per_tensor_scale_round_trip_is_bounded(self, values):
        weights = np.array(values, dtype=np.float64)
        fmt = QFormatMixed()
        quantised, tensor_scale = quantize_weights(weights, fmt=fmt)
        restored = dequantize_weights(quantised, fmt=fmt, scale=tensor_scale)

        min_int = -(1 << (fmt.weight_fmt.total_bits - 1))
        max_int = (1 << (fmt.weight_fmt.total_bits - 1)) - 1
        assert np.all(quantised >= min_int)
        assert np.all(quantised <= max_int)
        np.testing.assert_allclose(
            restored,
            weights,
            rtol=0.0,
            atol=(0.5 / (fmt.weight_fmt.scale * tensor_scale)) + 1e-12,
        )
