# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Block floating-point specifications

"""Shared-exponent block floating-point specifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlockFloatingMode:
    """Shared-exponent block floating-point specification."""

    mantissa_bits: int
    exponent_bits: int
    block_size: int = 32

    @property
    def label(self) -> str:
        """Human-readable label."""
        return f"BFP{self.mantissa_bits}E{self.exponent_bits}"

    @property
    def exponent_bias(self) -> int:
        """Bias applied to shared exponents."""
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def min_exponent(self) -> int:
        """Minimum unbiased exponent."""
        return -self.exponent_bias

    @property
    def max_exponent(self) -> int:
        """Maximum unbiased exponent."""
        return ((1 << self.exponent_bits) - 1) - self.exponent_bias

    @property
    def mantissa_range(self) -> int:
        """Largest signed mantissa magnitude."""
        return (1 << (self.mantissa_bits - 1)) - 1

    @property
    def emit_fraction(self) -> int:
        """Conservative fixed-point fallback fraction for RTL emission."""
        return max(1, self.mantissa_bits - 1)

    @property
    def metadata(self) -> dict[str, int | str]:
        """Deterministic metadata payload for cross-target telemetry."""
        return {
            "kind": "block_floating",
            "mantissa_bits": self.mantissa_bits,
            "exponent_bits": self.exponent_bits,
            "block_size": self.block_size,
            "exponent_min": self.min_exponent,
            "exponent_max": self.max_exponent,
            "block_exponent_alignment": "contiguous_flattened_block",
            "block_exponent_count_policy": "ceil(parameter_count / block_size)",
        }

    def block_exponent_count(self, parameter_count: int) -> int:
        """Return the exact number of shared exponents for a flat parameter payload."""
        if type(parameter_count) is not int:
            raise TypeError("parameter_count must be an integer")
        if parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if parameter_count == 0:
            return 0
        return (parameter_count + self.block_size - 1) // self.block_size

    def block_exponent_layout(self, parameter_count: int) -> "BlockExponentLayout":
        """Return the explicit exponent-vector layout for downstream emitters."""
        return BlockExponentLayout(
            parameter_count=parameter_count,
            block_size=self.block_size,
            exponent_count=self.block_exponent_count(parameter_count),
        )

    def validate_exponents(
        self,
        exponents: np.ndarray[Any, Any],
        *,
        parameter_count: int,
    ) -> np.ndarray[Any, Any]:
        """Validate exponent vector length and code range for a parameter payload."""
        layout = self.block_exponent_layout(parameter_count)
        return layout.validate_exponents(exponents, exponent_bits=self.exponent_bits)

    @classmethod
    def from_string(cls, fmt: str) -> "BlockFloatingMode":
        """Parse strict canonical format like 'BFP16E3'."""
        text = fmt.strip().upper()
        if not text.startswith("BFP"):
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        body = text[3:]
        if not body:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        m = re.fullmatch(r"(?P<left>\d+)E(?P<right>\d+)(?:X(?P<block>\d+))?$", body)
        if not m:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        mantissa_bits = int(m.group("left"))
        exponent_bits = int(m.group("right"))
        block_size = int(m.group("block") or 32)

        if mantissa_bits < 2:
            raise ValueError(f"Mantissa bits must be at least 2, got {mantissa_bits}")
        if exponent_bits < 1:
            raise ValueError(f"Exponent bits must be at least 1, got {exponent_bits}")
        if block_size < 1:
            raise ValueError(f"Block size must be positive, got {block_size}")

        return cls(mantissa_bits=mantissa_bits, exponent_bits=exponent_bits, block_size=block_size)

    @classmethod
    def from_aliases(cls, fmt: str) -> "BlockFloatingMode":
        """Parse tolerant aliases such as BFP16_E3, BFP16.3, BFP16-3, and BFP16E3X32."""
        if not isinstance(fmt, str):
            raise TypeError(f"Expected BFP format string, got {type(fmt)!r}")

        text = fmt.strip().upper()
        if not text.startswith("BFP"):
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        body = text[3:]
        if "E" not in body:
            body = re.sub(r"(?<=\d)[._-](?=\d)", "E", body)

        parts = body.split("X")
        if len(parts) > 2:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        core = re.sub(r"[._-](?=\d)", "E", parts[0])
        core = re.sub(r"[._-]", "", core)
        if "E" not in core:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        if len(parts) == 1:
            return cls.from_string(f"BFP{core}")

        if not parts[1] or not parts[1].isdigit():
            raise ValueError(f"Expected block size in 'BFP16E3X32', got {fmt!r}")

        return cls.from_string(f"BFP{core}X{int(parts[1])}")


@dataclass(frozen=True)
class BlockExponentLayout:
    """Concrete shared-exponent layout for flattened block-floating parameters."""

    parameter_count: int
    block_size: int
    exponent_count: int
    alignment: str = "contiguous_flattened_block"
    flattened_order: str = "row_major"

    def __post_init__(self) -> None:
        if type(self.parameter_count) is not int:
            raise TypeError("parameter_count must be an integer")
        if type(self.block_size) is not int:
            raise TypeError("block_size must be an integer")
        if type(self.exponent_count) is not int:
            raise TypeError("exponent_count must be an integer")
        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        expected = 0
        if self.parameter_count:
            expected = (self.parameter_count + self.block_size - 1) // self.block_size
        if self.exponent_count != expected:
            raise ValueError(
                f"exponent_count mismatch: expected {expected}, got {self.exponent_count}"
            )
        if self.alignment != "contiguous_flattened_block":
            raise ValueError("alignment must be contiguous_flattened_block")
        if self.flattened_order != "row_major":
            raise ValueError("flattened_order must be row_major")

    @property
    def last_block_size(self) -> int:
        """Number of parameters carried by the final exponent block."""
        if self.parameter_count == 0:
            return 0
        remainder = self.parameter_count % self.block_size
        return remainder or self.block_size

    def manifest(self) -> dict[str, int | str]:
        """Deterministic block-exponent layout manifest."""
        return {
            "alignment": self.alignment,
            "flattened_order": self.flattened_order,
            "parameter_count": self.parameter_count,
            "block_size": self.block_size,
            "exponent_count": self.exponent_count,
            "last_block_size": self.last_block_size,
            "exponent_index_formula": "parameter_index // block_size",
        }

    def validate_exponents(
        self,
        exponents: np.ndarray[Any, Any],
        *,
        exponent_bits: int,
    ) -> np.ndarray[Any, Any]:
        """Validate exponent vector length and encoded range."""
        raw = np.asarray(exponents)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("exponents must contain integer codes")
        codes = raw.astype(np.int64, copy=True).reshape(-1)
        if codes.size != self.exponent_count:
            raise ValueError(
                f"exponent count mismatch: expected {self.exponent_count}, got {codes.size}"
            )
        max_code = (1 << exponent_bits) - 1
        if np.any(codes < 0) or np.any(codes > max_code):
            raise ValueError("exponents exceed the configured block-floating range")
        return codes
