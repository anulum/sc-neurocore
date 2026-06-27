# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Layer precision specification

"""Validated per-layer bitstream length assignment rows."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LayerPrecision:
    """Bitstream length assignment for one layer.

    Parameters
    ----------
    layer_index:
        Zero-based layer index in the adaptive-precision plan.
    name:
        Non-empty layer name used in reports and manifests.
    bitstream_length:
        Positive stochastic-computing bitstream length. Layer-level planners
        round this value to a power of two.
    error_bound:
        Finite non-negative per-layer stochastic error bound.
    sensitivity:
        Finite non-negative sensitivity score used for budget allocation.
    """

    layer_index: int
    name: str
    bitstream_length: int
    error_bound: float
    sensitivity: float

    def __post_init__(self) -> None:
        """Validate adaptive-precision row invariants."""
        _validate_non_negative_int(self.layer_index, "layer_index")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        _validate_positive_int(self.bitstream_length, "bitstream_length")
        if self.bitstream_length & (self.bitstream_length - 1) != 0:
            raise ValueError("bitstream_length must be a power of two")
        _validate_non_negative_float(self.error_bound, "error_bound")
        _validate_non_negative_float(self.sensitivity, "sensitivity")

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-serializable adaptive-precision manifest row."""
        return {
            "layer_index": self.layer_index,
            "name": self.name,
            "bitstream_length": self.bitstream_length,
            "error_bound": self.error_bound,
            "sensitivity": self.sensitivity,
        }


def _validate_non_negative_int(value: int, name: str) -> None:
    """Reject non-integer or negative manifest index fields."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-integer or non-positive manifest length fields."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_float(value: float, name: str) -> None:
    """Reject non-numeric, non-finite, or negative manifest scalar fields."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
