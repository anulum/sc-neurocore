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

from .block_floating import BlockExponentLayout, BlockFloatingMode


@dataclass(frozen=True)
class BlockFloatingPrecisionConfig:
    """Block-floating specification for a single variable.

    Attributes
    ----------
    mantissa_bits:
        Number of signed mantissa bits emitted on the fixed datapath.
    exponent_bits:
        Number of bits stored for each shared block exponent code.
    block_size:
        Number of flattened parameters that share one exponent code.
    signed:
        Whether the mantissa stream uses signed two's-complement values.
    """

    mantissa_bits: int
    exponent_bits: int
    block_size: int
    signed: bool = True

    def __post_init__(self) -> None:
        """Validate block-floating width and layout invariants."""
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
        """Mantissa datapath width emitted for hardware and manifest payloads."""
        return self.mantissa_bits

    @property
    def fraction(self) -> int:
        """Conservative fractional width used by fixed-datapath fallbacks."""
        return max(1, self.mantissa_bits - 1)

    @property
    def emit_fraction(self) -> int:
        """Fractional width advertised to downstream fixed-datapath emitters."""
        return self.fraction

    @property
    def kind(self) -> str:
        """Stable manifest kind for block-floating precision contracts."""
        return "block_floating"

    @property
    def int_bits(self) -> int:
        """Signed mantissa magnitude bits excluding shared exponent metadata."""
        return self.mantissa_bits - 1

    @property
    def exponent_bias(self) -> int:
        """Bias that maps stored exponent codes to unbiased exponents."""
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def exponent_code_min(self) -> int:
        """Smallest encoded shared-exponent code."""
        return 0

    @property
    def exponent_code_max(self) -> int:
        """Largest encoded shared-exponent code."""
        return (1 << self.exponent_bits) - 1

    @property
    def mantissa_abs_max(self) -> int:
        """Largest signed mantissa magnitude before applying the block exponent."""
        return (1 << (self.mantissa_bits - 1)) - 1

    @property
    def max_exponent(self) -> int:
        """Largest unbiased exponent represented by the exponent stream."""
        return self.exponent_code_max - self.exponent_bias

    @property
    def min_exponent(self) -> int:
        """Smallest unbiased exponent represented by the exponent stream."""
        return -self.exponent_bias

    @property
    def max_value(self) -> float:
        """Largest positive value representable by mantissa and exponent fields."""
        return float(self.mantissa_abs_max) * (2.0**self.max_exponent)

    @property
    def min_value(self) -> float:
        """Smallest signed value representable by mantissa and exponent fields."""
        return -self.max_value

    @property
    def resolution(self) -> float:
        """Smallest positive exponent quantum available to the block stream."""
        return 2.0**self.min_exponent

    @property
    def q_label(self) -> str:
        """Canonical block-floating label including mantissa, exponent, and block size."""
        return f"BFP{self.mantissa_bits}E{self.exponent_bits}X{self.block_size}"

    @property
    def is_block_floating(self) -> bool:
        """Whether this precision contract requires shared exponent metadata."""
        return True

    def can_represent(self, value: float) -> bool:
        """Return whether ``value`` lies inside the coarse block-floating range."""
        return self.min_value <= value <= self.max_value

    def encode(self, value: float) -> int:
        """Reject scalar encoding when block exponent metadata is unavailable."""
        del value
        raise NotImplementedError("Block-floating encoding requires per-block exponent metadata.")

    def manifest(self) -> dict[str, object]:
        """Return the parameter-count-independent block-floating manifest."""
        return self.manifest_for_parameter_count()

    def block_exponent_count(self, parameter_count: int) -> int:
        """Return the number of shared exponents needed for ``parameter_count``."""
        return BlockFloatingMode(
            self.mantissa_bits,
            self.exponent_bits,
            self.block_size,
        ).block_exponent_count(parameter_count)

    def block_exponent_layout(self, parameter_count: int) -> BlockExponentLayout:
        """Return the flattened exponent-vector layout for ``parameter_count``."""
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
        """Validate and normalise encoded block exponents for a parameter payload."""
        return BlockFloatingMode(
            self.mantissa_bits,
            self.exponent_bits,
            self.block_size,
        ).validate_exponents(exponents, parameter_count=parameter_count)

    def manifest_for_parameter_count(
        self,
        parameter_count: int | None = None,
    ) -> dict[str, object]:
        """Return a deterministic manifest with optional concrete layout metadata."""
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
    """Fixed-point configuration for a single variable.

    Attributes
    ----------
    data_width:
        Total fixed-point storage width in bits.
    fraction:
        Number of fractional bits below the binary point.
    signed:
        Whether encoded values use signed two's-complement storage.
    """

    data_width: int
    fraction: int
    signed: bool = True

    @property
    def int_bits(self) -> int:
        """Integer magnitude bits available above the configured fraction."""
        return self.data_width - self.fraction - (1 if self.signed else 0)

    @property
    def max_value(self) -> float:
        """Largest fixed-point value representable by this configuration."""
        if self.signed:
            return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)
        return ((1 << self.data_width) - 1) / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        """Smallest fixed-point value representable by this configuration."""
        if self.signed:
            return -(1 << (self.data_width - 1)) / (1 << self.fraction)
        return 0.0

    @property
    def resolution(self) -> float:
        """Quantisation step represented by one least-significant bit."""
        return 1.0 / (1 << self.fraction)

    @property
    def q_label(self) -> str:
        """Standard sign-inclusive Q-format label.

        ``Q8.8`` for 16-bit and ``Q16.16`` for 32-bit expose every
        non-fractional bit, matching the ``q_format`` strings used across the
        NIR/FPGA pipeline. The integer magnitude bits, with sign excluded, are
        exposed separately as :attr:`int_bits`.
        """
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.data_width - self.fraction}.{self.fraction}"

    @property
    def emit_fraction(self) -> int:
        """Fractional width advertised to downstream fixed-point emitters."""
        return self.fraction

    @property
    def kind(self) -> str:
        """Stable manifest kind for fixed-point precision contracts."""
        return "fixed"

    @property
    def is_block_floating(self) -> bool:
        """Whether this precision contract requires shared exponent metadata."""
        return False

    def manifest(self) -> dict[str, bool | float | int | str]:
        """Return a deterministic fixed-point manifest for compilers and telemetry."""
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
        """Return whether ``value`` lies inside the fixed-point dynamic range."""
        return self.min_value <= value <= self.max_value

    def encode(self, value: float) -> int:
        """Quantise ``value`` to the nearest clamped fixed-point integer code."""
        raw = round(value * (1 << self.fraction))
        if self.signed:
            lo = -(1 << (self.data_width - 1))
            hi = (1 << (self.data_width - 1)) - 1
        else:
            lo = 0
            hi = (1 << self.data_width) - 1
        return max(lo, min(hi, raw))


PrecisionSpecLike = str | PrecisionConfig | BlockFloatingPrecisionConfig
