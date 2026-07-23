# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionTrapReport from former test_quantizer.py

"""Focused suite: TestPrecisionTrapReport from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403

class TestPrecisionTrapReport:
    """Validate precision trap report invariants for fixed-point deployment telemetry."""

    def test_manifest_counts_overflow_and_saturation_side(self):
        report = PrecisionTrapReport(
            operation="dense_mixed_qformat",
            output_codes=np.array([0, (1 << 31) - 1, -(1 << 31)], dtype=np.int64),
            overflow_mask=np.array([False, True, True], dtype=bool),
            output_fmt=Q16_16,
        )

        assert report.output_count == 3
        assert report.overflow_count == 2
        assert report.saturated_max_count == 1
        assert report.saturated_min_count == 1
        assert report.manifest() == {
            "operation": "dense_mixed_qformat",
            "output_format": "Q16.16",
            "output_count": 3,
            "overflow_count": 2,
            "underflow_count": 0,
            "saturated_min_count": 1,
            "saturated_max_count": 1,
            "has_overflow": True,
            "has_underflow": False,
        }

    def test_manifest_counts_sub_lsb_underflow(self):
        report = PrecisionTrapReport(
            operation="dense_mixed_qformat",
            output_codes=np.array([0, 1], dtype=np.int64),
            overflow_mask=np.array([False, False], dtype=bool),
            underflow_mask=np.array([True, False], dtype=bool),
            output_fmt=Q16_16,
        )

        assert report.underflow_count == 1
        assert report.has_underflow is True
        assert report.manifest()["underflow_count"] == 1

    def test_rejects_ambiguous_trap_shapes_and_out_of_range_codes(self):
        with pytest.raises(ValueError, match="identical shape"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1, 2], dtype=np.int64),
                overflow_mask=np.array([True], dtype=bool),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="identical shape"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1, 2], dtype=np.int64),
                overflow_mask=np.array([False, False], dtype=bool),
                underflow_mask=np.array([True], dtype=bool),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="exceed"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([1 << 31], dtype=np.int64),
                overflow_mask=np.array([True], dtype=bool),
                output_fmt=Q16_16,
            )
        with pytest.raises(TypeError, match="integer"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0.0]),
                overflow_mask=np.array([False], dtype=bool),
                output_fmt=Q16_16,
            )

    def test_rejects_blank_operation_wrong_format_and_non_vector_masks(self):
        with pytest.raises(ValueError, match="non-empty string"):
            PrecisionTrapReport(
                operation="",
                output_codes=np.array([0], dtype=np.int64),
                overflow_mask=np.array([False], dtype=bool),
                output_fmt=Q16_16,
            )
        with pytest.raises(TypeError, match="must be a QFormat"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0], dtype=np.int64),
                overflow_mask=np.array([False], dtype=bool),
                output_fmt="Q16.16",
            )
        with pytest.raises(ValueError, match="output_codes must be a 1-D vector"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([[0, 1]], dtype=np.int64),
                overflow_mask=np.array([False, True], dtype=bool),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="overflow_mask must be a 1-D vector"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0, 1], dtype=np.int64),
                overflow_mask=np.array([[False, True]], dtype=bool),
                output_fmt=Q16_16,
            )
        with pytest.raises(ValueError, match="underflow_mask must be a 1-D vector"):
            PrecisionTrapReport(
                operation="dense_mixed_qformat",
                output_codes=np.array([0, 1], dtype=np.int64),
                overflow_mask=np.array([False, True], dtype=bool),
                underflow_mask=np.array([[False, True]], dtype=bool),
                output_fmt=Q16_16,
            )
