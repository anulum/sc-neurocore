# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantization reports

"""Structured reports for fixed-point quantization anomalies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .q_format import QFormat
from .static_analysis import (
    FixedPointEnvelopeProof,
    prove_fixed_point_envelope,
)


def _fixed_integer_bounds(q: QFormat) -> tuple[int, int]:
    return -(1 << (q.total_bits - 1)), (1 << (q.total_bits - 1)) - 1


@dataclass(frozen=True)
class PrecisionTrapReport:
    """Per-output fixed-point saturation report for deployment trap wiring."""

    operation: str
    output_codes: np.ndarray[Any, Any]
    overflow_mask: np.ndarray[Any, Any]
    output_fmt: QFormat
    underflow_mask: np.ndarray[Any, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.output_fmt, QFormat):
            raise TypeError("output_fmt must be a QFormat")

        raw_codes = np.asarray(self.output_codes)
        if not np.issubdtype(raw_codes.dtype, np.integer):
            raise TypeError("output_codes must contain integer fixed-point codes")
        codes = raw_codes.astype(np.int64, copy=True)
        mask = np.asarray(self.overflow_mask, dtype=bool)
        if self.underflow_mask is None:
            underflow = np.zeros(mask.shape, dtype=bool)
        else:
            underflow = np.asarray(self.underflow_mask, dtype=bool)

        if codes.ndim != 1:
            raise ValueError("output_codes must be a 1-D vector")
        if mask.ndim != 1:
            raise ValueError("overflow_mask must be a 1-D vector")
        if underflow.ndim != 1:
            raise ValueError("underflow_mask must be a 1-D vector")
        if codes.shape != mask.shape or codes.shape != underflow.shape:
            raise ValueError(
                "output_codes, overflow_mask, and underflow_mask must have identical shape"
            )

        min_code, max_code = _fixed_integer_bounds(self.output_fmt)
        if np.any(codes < min_code) or np.any(codes > max_code):
            raise ValueError("output_codes exceed the configured output format")

        object.__setattr__(self, "output_codes", codes)
        object.__setattr__(self, "overflow_mask", mask.astype(bool, copy=True))
        object.__setattr__(self, "underflow_mask", underflow.astype(bool, copy=True))

    @property
    def output_count(self) -> int:
        """Number of output channels covered by this report."""
        return int(self.output_codes.size)

    @property
    def overflow_count(self) -> int:
        """Number of outputs that saturated during the producing operation."""
        return int(np.count_nonzero(self.overflow_mask))

    @property
    def has_overflow(self) -> bool:
        """Whether any output channel saturated."""
        return self.overflow_count > 0

    @property
    def underflow_count(self) -> int:
        """Number of nonzero outputs that collapsed below one output LSB."""
        underflow_mask = self.underflow_mask
        assert underflow_mask is not None
        return int(np.count_nonzero(underflow_mask))

    @property
    def has_underflow(self) -> bool:
        """Whether any nonzero output collapsed to the zero code."""
        return self.underflow_count > 0

    @property
    def saturated_min_count(self) -> int:
        """Number of outputs clamped to the minimum representable code."""
        min_code, _ = _fixed_integer_bounds(self.output_fmt)
        return int(np.count_nonzero(self.output_codes == min_code))

    @property
    def saturated_max_count(self) -> int:
        """Number of outputs clamped to the maximum representable code."""
        _, max_code = _fixed_integer_bounds(self.output_fmt)
        return int(np.count_nonzero(self.output_codes == max_code))

    def manifest(self) -> dict[str, bool | int | str]:
        """Deterministic trap metadata for host and hardware telemetry."""
        return {
            "operation": self.operation,
            "output_format": self.output_fmt.q_label,
            "output_count": self.output_count,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count,
            "saturated_min_count": self.saturated_min_count,
            "saturated_max_count": self.saturated_max_count,
            "has_overflow": self.has_overflow,
            "has_underflow": self.has_underflow,
        }


@dataclass(frozen=True)
class PrecisionEnvelopeReport:
    """Conservative fixed-point output-envelope report for deployment checks."""

    operation: str
    output_codes: np.ndarray[Any, Any]
    overflow_mask: np.ndarray[Any, Any]
    abs_bound_codes: np.ndarray[Any, Any]
    output_fmt: QFormat
    underflow_mask: np.ndarray[Any, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.output_fmt, QFormat):
            raise TypeError("output_fmt must be a QFormat")

        raw_codes = np.asarray(self.output_codes)
        raw_bounds = np.asarray(self.abs_bound_codes)
        if not np.issubdtype(raw_codes.dtype, np.integer):
            raise TypeError("output_codes must contain integer fixed-point codes")
        if not np.issubdtype(raw_bounds.dtype, np.integer):
            raise TypeError("abs_bound_codes must contain integer fixed-point codes")

        codes = raw_codes.astype(np.int64, copy=True)
        bounds = raw_bounds.astype(np.int64, copy=True)
        mask = np.asarray(self.overflow_mask, dtype=bool)
        if self.underflow_mask is None:
            underflow = np.zeros(mask.shape, dtype=bool)
        else:
            underflow = np.asarray(self.underflow_mask, dtype=bool)

        if codes.ndim != 1:
            raise ValueError("output_codes must be a 1-D vector")
        if mask.ndim != 1:
            raise ValueError("overflow_mask must be a 1-D vector")
        if underflow.ndim != 1:
            raise ValueError("underflow_mask must be a 1-D vector")
        if bounds.ndim != 1:
            raise ValueError("abs_bound_codes must be a 1-D vector")
        if (
            codes.shape != mask.shape
            or codes.shape != underflow.shape
            or codes.shape != bounds.shape
        ):
            raise ValueError(
                "output_codes, overflow_mask, underflow_mask, and abs_bound_codes must have identical shape"
            )
        if np.any(bounds < 0):
            raise ValueError("abs_bound_codes must be non-negative")

        min_code, max_code = _fixed_integer_bounds(self.output_fmt)
        if np.any(codes < min_code) or np.any(codes > max_code):
            raise ValueError("output_codes exceed the configured output format")

        object.__setattr__(self, "output_codes", codes)
        object.__setattr__(self, "overflow_mask", mask.astype(bool, copy=True))
        object.__setattr__(self, "underflow_mask", underflow.astype(bool, copy=True))
        object.__setattr__(self, "abs_bound_codes", bounds)

    @property
    def output_count(self) -> int:
        """Number of output channels covered by this report."""
        return int(self.output_codes.size)

    @property
    def overflow_count(self) -> int:
        """Number of outputs that saturated during the producing operation."""
        return int(np.count_nonzero(self.overflow_mask))

    @property
    def observed_overflow_free(self) -> bool:
        """Whether the realised workload avoided fixed-point saturation."""
        return self.overflow_count == 0

    @property
    def underflow_count(self) -> int:
        """Number of nonzero outputs that collapsed below one output LSB."""
        underflow_mask = self.underflow_mask
        assert underflow_mask is not None
        return int(np.count_nonzero(underflow_mask))

    @property
    def observed_underflow_free(self) -> bool:
        """Whether the realised workload avoided sub-LSB output collapse."""
        return self.underflow_count == 0

    @property
    def conservative_safe_bound_code(self) -> int:
        """Largest symmetric absolute code accepted as overflow-free."""
        return self.fixed_point_envelope_proof.conservative_safe_bound_code

    @property
    def max_abs_output_code(self) -> int:
        """Maximum absolute saturated output code observed in the workload."""
        if self.output_codes.size == 0:
            return 0
        return int(np.max(np.abs(self.output_codes.astype(object))))

    @property
    def max_abs_bound_code(self) -> int:
        """Maximum conservative absolute output bound for the workload."""
        if self.abs_bound_codes.size == 0:
            return 0
        return int(np.max(self.abs_bound_codes))

    @property
    def min_headroom_code(self) -> int:
        """Smallest conservative headroom, in fixed-point integer codes."""
        return self.fixed_point_envelope_proof.min_headroom_code

    @property
    def conservative_overflow_free(self) -> bool:
        """Whether the absolute envelope proves the workload is in range."""
        return self.fixed_point_envelope_proof.static_overflow_proven_safe

    @property
    def fixed_point_envelope_proof(self) -> FixedPointEnvelopeProof:
        """Static Q-format width proof for the conservative envelope."""
        bound_codes = self.abs_bound_codes.tolist()
        if not bound_codes:
            bound_codes = [0]
        return prove_fixed_point_envelope(
            [int(code) for code in bound_codes],
            total_bits=self.output_fmt.total_bits,
            fractional_bits=self.output_fmt.fraction_bits,
            signed=True,
        )

    @property
    def required_total_bits(self) -> int:
        """Signed fixed-point width required by the conservative envelope."""
        return self.fixed_point_envelope_proof.required_total_bits

    @property
    def required_integer_bits(self) -> int:
        """Q-format integer bits, including sign, required by the envelope."""
        return self.fixed_point_envelope_proof.required_integer_bits

    @property
    def width_headroom_bits(self) -> int:
        """Remaining signed fixed-point width after the conservative proof."""
        return self.fixed_point_envelope_proof.width_headroom_bits

    @property
    def saturation_required(self) -> bool:
        """Whether the conservative proof requires a saturating output clamp."""
        return self.fixed_point_envelope_proof.saturation_required

    @property
    def static_overflow_proven_safe(self) -> bool:
        """Whether static width proof guarantees no Q-format overflow."""
        return self.fixed_point_envelope_proof.static_overflow_proven_safe

    def manifest(self) -> dict[str, bool | int | str]:
        """Deterministic envelope metadata for predeployment gates."""
        proof = self.fixed_point_envelope_proof
        return {
            "operation": self.operation,
            "output_format": self.output_fmt.q_label,
            "output_count": self.output_count,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count,
            "observed_overflow_free": self.observed_overflow_free,
            "observed_underflow_free": self.observed_underflow_free,
            "conservative_overflow_free": self.conservative_overflow_free,
            "max_abs_output_code": self.max_abs_output_code,
            "max_abs_bound_code": self.max_abs_bound_code,
            "conservative_safe_bound_code": self.conservative_safe_bound_code,
            "min_headroom_code": self.min_headroom_code,
            "proof_kind": proof.proof_kind,
            "required_total_bits": proof.required_total_bits,
            "required_integer_bits": proof.required_integer_bits,
            "width_headroom_bits": proof.width_headroom_bits,
            "saturation_required": proof.saturation_required,
            "static_overflow_proven_safe": proof.static_overflow_proven_safe,
        }
