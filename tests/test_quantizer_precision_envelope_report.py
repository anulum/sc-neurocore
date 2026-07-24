# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionEnvelopeReport from former test_quantizer.py

"""Focused suite: TestPrecisionEnvelopeReport from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403


class TestPrecisionEnvelopeReport:
    """Validate precision envelope invariants for predeployment range gates."""

    def test_manifest_distinguishes_observed_and_conservative_safety(self):
        report = PrecisionEnvelopeReport(
            operation="dense_mixed_qformat",
            output_codes=np.array([1024, -512], dtype=np.int64),
            overflow_mask=np.array([False, False], dtype=bool),
            abs_bound_codes=np.array([2048, 4096], dtype=np.int64),
            output_fmt=Q16_16,
        )

        assert report.output_count == 2
        assert report.overflow_count == 0
        assert report.observed_overflow_free is True
        assert report.underflow_count == 0
        assert report.observed_underflow_free is True
        assert report.conservative_overflow_free is True
        assert report.max_abs_output_code == 1024
        assert report.max_abs_bound_code == 4096
        assert report.min_headroom_code == ((1 << 31) - 1) - 4096
        assert report.required_total_bits == 14
        assert report.required_integer_bits == 1
        assert report.width_headroom_bits == 18
        assert report.saturation_required is False
        assert report.static_overflow_proven_safe is True
        assert report.manifest()["conservative_overflow_free"] is True
        assert report.manifest()["proof_kind"] == "signed_symmetric_fixed_point_width"

    def test_envelope_manifest_reuses_static_analysis_width_proof(self):
        """Quantizer envelope proof fields must match the static-analysis proof."""
        report = PrecisionEnvelopeReport(
            operation="dense_mixed_qformat",
            output_codes=np.array([1024, -512], dtype=np.int64),
            overflow_mask=np.array([False, False], dtype=bool),
            abs_bound_codes=np.array([2048, 4096], dtype=np.int64),
            output_fmt=Q16_16,
        )
        proof = prove_fixed_point_envelope([2048, 4096])
        manifest = report.manifest()
        proof_manifest = proof.manifest()

        for key in (
            "proof_kind",
            "conservative_safe_bound_code",
            "max_abs_bound_code",
            "min_headroom_code",
            "required_total_bits",
            "required_integer_bits",
            "width_headroom_bits",
            "saturation_required",
            "static_overflow_proven_safe",
        ):
            assert manifest[key] == proof_manifest[key]

    def test_manifest_distinguishes_underflow_from_overflow(self):
        report = PrecisionEnvelopeReport(
            operation="dense_mixed_qformat",
            output_codes=np.array([0], dtype=np.int64),
            overflow_mask=np.array([False], dtype=bool),
            underflow_mask=np.array([True], dtype=bool),
            abs_bound_codes=np.array([1], dtype=np.int64),
            output_fmt=Q16_16,
        )

        assert report.observed_overflow_free is True
        assert report.observed_underflow_free is False
        assert report.manifest()["underflow_count"] == 1

    def test_conservative_envelope_can_reject_cancelling_outputs(self):
        report = PrecisionEnvelopeReport(
            operation="dense_mixed_qformat",
            output_codes=np.array([0], dtype=np.int64),
            overflow_mask=np.array([False], dtype=bool),
            abs_bound_codes=np.array([(1 << 31) + 1], dtype=np.int64),
            output_fmt=Q16_16,
        )

        assert report.observed_overflow_free is True
        assert report.conservative_overflow_free is False
        assert report.min_headroom_code < 0
        assert report.required_total_bits == 33
        assert report.required_integer_bits == 17
        assert report.width_headroom_bits == -1
        assert report.saturation_required is True
        assert report.static_overflow_proven_safe is False

    def test_rejects_ambiguous_envelope_shapes_and_negative_bounds(self):
        with pytest.raises(ValueError, match="identical shape"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1, 2], dtype=np.int64),
                overflow_mask=np.array([False, False], dtype=bool),
                abs_bound_codes=np.array([2], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="identical shape"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1, 2], dtype=np.int64),
                overflow_mask=np.array([False, False], dtype=bool),
                underflow_mask=np.array([True], dtype=bool),
                abs_bound_codes=np.array([1, 2], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="non-negative"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1], dtype=np.int64),
                overflow_mask=np.array([False], dtype=bool),
                abs_bound_codes=np.array([-1], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(TypeError, match="integer"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1], dtype=np.int64),
                overflow_mask=np.array([False], dtype=bool),
                abs_bound_codes=np.array([1.5]),
                output_fmt=Q16_16,
            )

    def test_rejects_blank_operation_wrong_format_and_non_vector_arrays(self):
        with pytest.raises(ValueError, match="non-empty string"):
            PrecisionEnvelopeReport(
                operation="",
                output_codes=np.array([0], dtype=np.int64),
                overflow_mask=np.array([False], dtype=bool),
                abs_bound_codes=np.array([0], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(TypeError, match="must be a QFormat"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0], dtype=np.int64),
                overflow_mask=np.array([False], dtype=bool),
                abs_bound_codes=np.array([0], dtype=np.int64),
                output_fmt="Q16.16",
            )
        with pytest.raises(TypeError, match="output_codes must contain integer"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0.0]),
                overflow_mask=np.array([False], dtype=bool),
                abs_bound_codes=np.array([0], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="output_codes must be a 1-D vector"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([[0, 1]], dtype=np.int64),
                overflow_mask=np.array([False, True], dtype=bool),
                abs_bound_codes=np.array([0, 1], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="overflow_mask must be a 1-D vector"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0, 1], dtype=np.int64),
                overflow_mask=np.array([[False, True]], dtype=bool),
                abs_bound_codes=np.array([0, 1], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="underflow_mask must be a 1-D vector"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0, 1], dtype=np.int64),
                overflow_mask=np.array([False, True], dtype=bool),
                underflow_mask=np.array([[False, True]], dtype=bool),
                abs_bound_codes=np.array([0, 1], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="abs_bound_codes must be a 1-D vector"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0, 1], dtype=np.int64),
                overflow_mask=np.array([False, True], dtype=bool),
                abs_bound_codes=np.array([[0, 1]], dtype=np.int64),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="exceed"):
            PrecisionEnvelopeReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1 << 31], dtype=np.int64),
                overflow_mask=np.array([False], dtype=bool),
                abs_bound_codes=np.array([0], dtype=np.int64),
                output_fmt=Q16_16,
            )

    def test_empty_envelope_reports_zero_extents_and_zero_bound_proof(self):
        report = PrecisionEnvelopeReport(
            operation="dense_mixed_qformat",
            output_codes=np.array([], dtype=np.int64),
            overflow_mask=np.array([], dtype=bool),
            abs_bound_codes=np.array([], dtype=np.int64),
            output_fmt=Q16_16,
        )

        assert report.output_count == 0
        assert report.max_abs_output_code == 0
        assert report.max_abs_bound_code == 0
        # Empty bounds fall back to [0] before the static width proof, which is
        # trivially overflow-free.
        assert report.conservative_overflow_free is True
        assert report.static_overflow_proven_safe is True
