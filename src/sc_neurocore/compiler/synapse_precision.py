# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synapse precision specification

"""Validated per-synapse precision and error-bound rows."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SynapsePrecision:
    """Precision assignment and conservative error bound for one synapse.

    Parameters
    ----------
    layer_index:
        Zero-based layer index in the adaptive-precision plan.
    layer_name:
        Non-empty layer name used in reports and manifests.
    output_index:
        Zero-based output row index for the weight matrix.
    input_index:
        Zero-based input column index for the weight matrix.
    bit_width:
        Positive fixed-point bit width assigned to the synapse.
    bitstream_length:
        Positive stochastic-computing bitstream length assigned to the synapse.
    sensitivity:
        Finite non-negative synapse sensitivity score.
    quantization_error_bound:
        Finite non-negative error bound from fixed-point quantization.
    stochastic_error_bound:
        Finite non-negative Hoeffding-style stochastic error bound.
    total_error_bound:
        Finite non-negative aggregate bound that must cover both components.
    """

    layer_index: int
    layer_name: str
    output_index: int
    input_index: int
    bit_width: int
    bitstream_length: int
    sensitivity: float
    quantization_error_bound: float
    stochastic_error_bound: float
    total_error_bound: float

    def __post_init__(self) -> None:
        """Validate per-synapse precision-row invariants."""
        _validate_non_negative_int(self.layer_index, "layer_index")
        if not isinstance(self.layer_name, str) or not self.layer_name:
            raise ValueError("layer_name must be a non-empty string")
        _validate_non_negative_int(self.output_index, "output_index")
        _validate_non_negative_int(self.input_index, "input_index")
        _validate_positive_int(self.bit_width, "bit_width")
        _validate_positive_int(self.bitstream_length, "bitstream_length")
        _validate_non_negative_float(self.sensitivity, "sensitivity")
        _validate_non_negative_float(
            self.quantization_error_bound,
            "quantization_error_bound",
        )
        _validate_non_negative_float(
            self.stochastic_error_bound,
            "stochastic_error_bound",
        )
        _validate_non_negative_float(self.total_error_bound, "total_error_bound")
        component_sum = self.quantization_error_bound + self.stochastic_error_bound
        if self.total_error_bound + 1e-15 < component_sum:
            raise ValueError("total_error_bound must cover quantization and stochastic bounds")

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-serialisable precision-plan row."""
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "output_index": self.output_index,
            "input_index": self.input_index,
            "bit_width": self.bit_width,
            "bitstream_length": self.bitstream_length,
            "sensitivity": self.sensitivity,
            "quantization_error_bound": self.quantization_error_bound,
            "stochastic_error_bound": self.stochastic_error_bound,
            "total_error_bound": self.total_error_bound,
        }


def _validate_non_negative_int(value: int, name: str) -> None:
    """Reject non-integer or negative manifest index fields."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-integer or non-positive precision fields."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_float(value: float, name: str) -> None:
    """Reject non-numeric, non-finite, or negative error-bound fields."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
