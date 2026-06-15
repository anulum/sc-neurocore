# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Precision configuration

"""Per-variable precision configuration specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .block_floating import BlockFloatingMode, BlockExponentLayout


@dataclass(frozen=True)
class BlockFloatingPrecisionConfig:
    """Block-floating specification for a single variable."""

    mantissa_bits: int
    exponent_bits: int
    block_size: int
    signed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mantissa_bits, int) or isinstance(self.mantissa_bits, bool):
            raise TypeError("mantissa_bits must be an integer")
        if not isinstance(self.exponent_bits, int) or isinstance(self.exponent_bits, bool):
            raise TypeError("exponent_bits must be an integer")
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool):
            raise TypeError("block_size must be an integer")
        if type(self.signed) is not bool:
            raise TypeError("signed must be a boolean")
        if self.mantissa_bits < 2:
            raise ValueError("mantissa_bits must be at least 2")
        if self.exponent_bits < 1:
            raise ValueError("exponent_bits must be at least 1")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

    @property
    def data_width(self) -> int:
        return self.mantissa_bits

    @property
    def fraction(self) -> int:
        return max(1, self.mantissa_bits - 1)

    @property
    def emit_fraction(self) -> int:
        return self.fraction

    @property
    def kind(self) -> str:
        return "block_floating"

    @property
    def int_bits(self) -> int:
        return self.mantissa_bits - 1

    @property
    def exponent_bias(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def exponent_code_min(self) -> int:
        return 0

    @property
    def exponent_code_max(self) -> int:
        return (1 << self.exponent_bits) - 1

    @property
    def mantissa_abs_max(self) -> int:
        return (1 << (self.mantissa_bits - 1)) - 1

    @property
    def max_exponent(self) -> int:
        return self.exponent_code_max - self.exponent_bias

    @property
    def min_exponent(self) -> int:
        return -self.exponent_bias

    @property
    def max_value(self) -> float:
        return float(self.mantissa_abs_max) * (2.0**self.max_exponent)

    @property
    def min_value(self) -> float:
        return -self.max_value

    @property
    def resolution(self) -> float:
        return 2.0**self.min_exponent

    @property
    def q_label(self) -> str:
        return f"BFP{self.mantissa_bits}E{self.exponent_bits}X{self.block_size}"

    @property
    def is_block_floating(self) -> bool:
        return True

    def can_represent(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value

    def encode(self, value: float) -> int:
        del value
        raise NotImplementedError("Block-floating encoding requires per-block exponent metadata.")

    def manifest(self) -> dict[str, object]:
        return self.manifest_for_parameter_count()

    def block_exponent_count(self, parameter_count: int) -> int:
        return BlockFloatingMode(
            self.mantissa_bits,
            self.exponent_bits,
            self.block_size,
        ).block_exponent_count(parameter_count)

    def block_exponent_layout(self, parameter_count: int) -> BlockExponentLayout:
        return BlockFloatingMode(
            self.mantissa_bits,
            self.exponent_bits,
            self.block_size,
        ).block_exponent_layout(parameter_count)

    def validate_exponents(
        self,
        exponents: np.ndarray[Any, Any],
        *,
        parameter_count: int,
    ) -> np.ndarray[Any, Any]:
        return BlockFloatingMode(
            self.mantissa_bits,
            self.exponent_bits,
            self.block_size,
        ).validate_exponents(exponents, parameter_count=parameter_count)

    def manifest_for_parameter_count(
        self,
        parameter_count: int | None = None,
    ) -> dict[str, object]:
        layout = (
            self.block_exponent_layout(parameter_count) if parameter_count is not None else None
        )
        payload: dict[str, object] = {
            "kind": self.kind,
            "label": self.q_label,
            "data_width": self.data_width,
            "fraction": self.emit_fraction,
            "mantissa_bits": self.mantissa_bits,
            "exponent_bits": self.exponent_bits,
            "block_size": self.block_size,
            "signed": self.signed,
            "emitted_fraction": self.emit_fraction,
            "emitted_datapath_width": self.data_width,
            "emitted_datapath_fraction": self.emit_fraction,
            "exponent_stream_width": self.exponent_bits,
            "exponent_bias": self.exponent_bias,
            "exponent_code_range": [self.exponent_code_min, self.exponent_code_max],
            "exponent_range": [self.min_exponent, self.max_exponent],
            "mantissa_abs_max": self.mantissa_abs_max,
            "minimum_quantum": self.resolution,
            "max_abs_value": self.max_value,
            "block_exponent_alignment": "contiguous_flattened_block",
            "block_exponent_count": "ceil(parameter_count / block_size)",
            "block_exponent_count_policy": "ceil(parameter_count / block_size)",
            "exponent_vector_width": "exponent_bits * ceil(parameter_count / block_size)",
            "datapath_contract": "fixed_mantissa_with_explicit_shared_exponent_metadata",
        }
        if layout is not None:
            payload["parameter_count"] = parameter_count
            payload["block_exponent_count"] = layout.exponent_count
            payload["block_exponent_layout"] = layout.manifest()
            payload["exponent_vector_width"] = self.exponent_bits * layout.exponent_count
        return payload


@dataclass(frozen=True)
class PrecisionConfig:
    """Fixed-point configuration for a single variable."""

    data_width: int
    fraction: int
    signed: bool = True

    @property
    def int_bits(self) -> int:
        return self.data_width - self.fraction - (1 if self.signed else 0)

    @property
    def max_value(self) -> float:
        if self.signed:
            return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)
        return ((1 << self.data_width) - 1) / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        if self.signed:
            return -(1 << (self.data_width - 1)) / (1 << self.fraction)
        return 0.0

    @property
    def resolution(self) -> float:
        return 1.0 / (1 << self.fraction)

    @property
    def q_label(self) -> str:
        """Standard sign-inclusive Q-format label (``Q8.8`` for 16-bit, ``Q16.16``
        for 32-bit): the integer field spans every non-fractional bit, matching the
        ``q_format`` strings used across the NIR/FPGA pipeline. The integer
        *magnitude* bits (sign excluded) are exposed separately as :attr:`int_bits`.
        """
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.data_width - self.fraction}.{self.fraction}"

    @property
    def emit_fraction(self) -> int:
        return self.fraction

    @property
    def kind(self) -> str:
        return "fixed"

    @property
    def is_block_floating(self) -> bool:
        return False

    def manifest(self) -> dict[str, bool | float | int | str]:
        return {
            "kind": self.kind,
            "data_width": self.data_width,
            "fraction": self.fraction,
            "signed": self.signed,
            "label": self.q_label,
            "emitted_datapath_width": self.data_width,
            "emitted_datapath_fraction": self.emit_fraction,
            "exponent_stream_width": 0,
            "exponent_vector_width": 0,
            "datapath_contract": "fixed_point_twos_complement",
        }

    def can_represent(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value

    def encode(self, value: float) -> int:
        raw = round(value * (1 << self.fraction))
        if self.signed:
            lo = -(1 << (self.data_width - 1))
            hi = (1 << (self.data_width - 1)) - 1
        else:
            lo = 0
            hi = (1 << self.data_width) - 1
        return max(lo, min(hi, raw))


PrecisionSpecLike = str | PrecisionConfig | BlockFloatingPrecisionConfig
