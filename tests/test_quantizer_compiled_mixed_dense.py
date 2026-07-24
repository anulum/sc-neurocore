# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompiledMixedDense from former test_quantizer.py

"""Focused suite: TestCompiledMixedDense from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403


class TestCompiledMixedDense:
    """Validate the bit-true dense Q8.8-weight/Q16.16-accumulator contract."""

    def test_canonical_q88_q1616_integer_mac_matches_manual_codes(self):
        weights = np.array([[0.5, -0.25], [1.0, 0.125]], dtype=np.float64)
        inputs = np.array([0.5, -0.25], dtype=np.float64)
        compiled = compile_dense_mixed_precision(weights, fmt=QFormatMixed(scale_per_tensor=False))

        assert compiled.manifest()["operation"] == "dense_mixed_qformat"
        assert compiled.manifest()["weight_shape"] == [2, 2]

        manual = np.array(
            [
                (128 * 32768 + (-64) * (-16384)) // Q8_8.scale,
                (256 * 32768 + 32 * (-16384)) // Q8_8.scale,
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(compiled.forward_accumulator_codes(inputs), manual)
        np.testing.assert_allclose(
            compiled.forward_float(inputs), weights @ inputs, atol=1 / Q16_16.scale
        )

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is True
        assert envelope.conservative_overflow_free is True
        assert envelope.max_abs_bound_code == 34816
        assert envelope.required_total_bits == 17
        assert envelope.required_integer_bits == 1
        assert envelope.width_headroom_bits == 15

    def test_default_scaled_contract_tracks_float_dense_result(self):
        weights = np.array([[0.375, -0.125, 0.0625], [-0.5, 0.25, 0.125]], dtype=np.float64)
        inputs = np.array([0.5, -0.25, 0.125], dtype=np.float64)
        compiled = compile_dense_mixed_precision(weights)

        assert compiled.tensor_scale > 1.0
        np.testing.assert_allclose(compiled.forward_float(inputs), weights @ inputs, atol=2e-4)

    def test_canonical_contract_saturates_and_reports_accumulator_overflow(self):
        weights = np.array([[127.0, 127.0]], dtype=np.float64)
        inputs = np.array([20_000.0, 20_000.0], dtype=np.float64)
        compiled = compile_dense_mixed_precision(weights, fmt=QFormatMixed(scale_per_tensor=False))

        codes, overflow = compiled.forward_with_overflow(inputs)

        assert overflow.tolist() == [True]
        assert int(codes[0]) == (1 << (Q16_16.total_bits - 1)) - 1

        report = compiled.precision_trap_report(inputs)
        assert report.manifest()["operation"] == "dense_mixed_qformat"
        assert report.overflow_count == 1
        assert report.saturated_max_count == 1
        assert report.saturated_min_count == 0

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is False
        assert envelope.conservative_overflow_free is False
        assert envelope.max_abs_bound_code > envelope.conservative_safe_bound_code

    def test_canonical_contract_reports_sub_lsb_underflow(self):
        weights = np.array([[1 / Q8_8.scale]], dtype=np.float64)
        inputs = np.array([1 / Q16_16.scale], dtype=np.float64)
        compiled = compile_dense_mixed_precision(weights, fmt=QFormatMixed(scale_per_tensor=False))

        codes, overflow = compiled.forward_with_overflow(inputs)

        assert int(codes[0]) == 0
        assert overflow.tolist() == [False]

        report = compiled.precision_trap_report(inputs)
        assert report.overflow_count == 0
        assert report.underflow_count == 1
        assert report.manifest()["has_underflow"] is True

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is True
        assert envelope.observed_underflow_free is False

    def test_negative_raw_products_use_hardware_arithmetic_shift(self):
        weights = np.array([[0.5]], dtype=np.float64)
        inputs = np.array([-1 / Q16_16.scale], dtype=np.float64)
        compiled = compile_dense_mixed_precision(weights, fmt=QFormatMixed(scale_per_tensor=False))

        assert int(compiled.forward_accumulator_codes(inputs)[0]) == -1

    def test_rejects_non_dense_or_non_finite_inputs_before_accumulation(self):
        with pytest.raises(ValueError, match="2-D dense"):
            compile_dense_mixed_precision(np.array([0.5, -0.25]))
        compiled = compile_dense_mixed_precision(np.array([[0.5, -0.25]], dtype=np.float64))
        with pytest.raises(ValueError, match="input length mismatch"):
            compiled.forward_float(np.array([0.5]))
        with pytest.raises(ValueError, match="finite values"):
            compiled.forward_float(np.array([0.5, np.nan]))
