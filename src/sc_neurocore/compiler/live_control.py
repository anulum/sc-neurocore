# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Live-parameter contracts for FPGA control and hardware parameter updates.

This module defines immutable schema objects used by compiler and runner layers to
describe how long-lived parameters are written to a target design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from sc_neurocore.compiler.quantizer import QFormat

BusProtocol = Literal["axi4_lite", "pcie"]
PrecisionMode = Literal["q", "bfp"]
TrapAction = Literal["hold", "saturate", "clip", "halt", "interrupt"]

_VALID_PROTOCOLS = frozenset({"axi4_lite", "axi_lite", "pcie"})


def _normalise_bus_protocol(protocol: str) -> BusProtocol:
    protocol = protocol.strip().lower()
    if protocol == "axi_lite":
        protocol = "axi4_lite"
    if protocol not in _VALID_PROTOCOLS:
        raise ValueError(f"Unsupported MMIO protocol: {protocol!r}")
    return protocol  # type: ignore[return-value]


@dataclass(frozen=True)
class TrapSpec:
    """Contract for overflow and saturation trap signaling.

    Parameters
    ----------
    enabled : bool
        Enable hardware trap signal generation.
    action : {'hold', 'saturate', 'clip', 'halt', 'interrupt'}
        Action taken when an overflow condition is detected.
    flag_register_offset : int
        Optional offset in bytes from the bank base where trap flags are written.
    sticky : bool
        Keep trap status latched until an explicit clear operation.
    max_flags : int
        Number of individual trap sources supported (e.g. one per accumulator bank).
    """

    enabled: bool = True
    action: TrapAction = "interrupt"
    flag_register_offset: int | None = None
    sticky: bool = True
    max_flags: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.max_flags, int) or self.max_flags <= 0:
            raise ValueError("max_flags must be a positive integer")
        if self.max_flags > 256:
            raise ValueError("max_flags cannot exceed 256")
        if self.flag_register_offset is not None:
            if not isinstance(self.flag_register_offset, int):
                raise ValueError("flag_register_offset must be an integer")
            if self.flag_register_offset < 0 or self.flag_register_offset % 4 != 0:
                raise ValueError("flag_register_offset must be non-negative and 4-byte aligned")


@dataclass(frozen=True)
class ParameterBankSpec:
    """Immutable contract describing one writable bank mapped to MMIO."""

    bank_name: str
    start_address_bytes: int
    parameter_count: int
    parameter_names: tuple[str, ...] | list[str]
    precision_mode: PrecisionMode = "q"
    q_format: str = "Q8.8"
    bfp_exponent_bits: int = 5
    bfp_mantissa_bits: int = 11
    writable: bool = True
    reset_value: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.parameter_names, list):
            object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        if any(not isinstance(name, str) or not name.strip() for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if not isinstance(self.bank_name, str) or not self.bank_name.strip():
            raise ValueError("bank_name must be a non-empty string")
        if not isinstance(self.start_address_bytes, int) or self.start_address_bytes < 0:
            raise ValueError("start_address_bytes must be a non-negative integer")
        if self.start_address_bytes % 4 != 0:
            raise ValueError("start_address_bytes must be 4-byte aligned")
        if not isinstance(self.parameter_count, int) or self.parameter_count <= 0:
            raise ValueError("parameter_count must be a positive integer")
        if len(self.parameter_names) != len(set(self.parameter_names)):
            raise ValueError("parameter_names must be unique")
        if not self.parameter_names:
            raise ValueError("parameter_names must contain at least one parameter")
        if len(self.parameter_names) > self.parameter_count:
            raise ValueError("parameter_names must not exceed parameter_count")
        if self.precision_mode not in {"q", "bfp"}:
            raise ValueError("precision_mode must be 'q' or 'bfp'")
        if self.precision_mode == "q":
            # parse to validate the exact syntax and enforce byte-aligned width
            q = QFormat.from_string(self.q_format)
            if q.total_bits % 8 != 0:
                raise ValueError("Q-format width must be byte-aligned")
        else:
            if not (3 <= self.bfp_exponent_bits <= 12):
                raise ValueError("bfp_exponent_bits must be between 3 and 12")
            if not (4 <= self.bfp_mantissa_bits <= 48):
                raise ValueError("bfp_mantissa_bits must be between 4 and 48")

        width_bits = self.entry_width_bits
        if width_bits % 8 != 0:
            raise ValueError("entry width must be byte-aligned")
        if width_bits <= 0:
            raise ValueError("entry width must be positive")
        if width_bits > 64:
            raise ValueError("entry width must not exceed 64 bits")

    @property
    def entry_width_bits(self) -> int:
        """Return storage width in bits for one bank entry."""
        if self.precision_mode == "q":
            return QFormat.from_string(self.q_format).total_bits
        return int(self.bfp_exponent_bits + self.bfp_mantissa_bits)

    @property
    def entry_width_bytes(self) -> int:
        """Return storage width in bytes for one bank entry."""
        return self.entry_width_bits // 8

    @property
    def span_bytes(self) -> int:
        """Return total byte span from start to end for the bank."""
        return self.parameter_count * self.entry_width_bytes

    @property
    def end_address_bytes(self) -> int:
        """Return first invalid byte address beyond the parameter bank."""
        return self.start_address_bytes + self.span_bytes

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible mapping."""
        return {
            "bank_name": self.bank_name,
            "start_address_bytes": self.start_address_bytes,
            "parameter_count": self.parameter_count,
            "parameter_names": list(self.parameter_names),
            "precision_mode": self.precision_mode,
            "q_format": self.q_format,
            "bfp_exponent_bits": self.bfp_exponent_bits,
            "bfp_mantissa_bits": self.bfp_mantissa_bits,
            "writable": self.writable,
            "reset_value": self.reset_value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParameterBankSpec":
        """Rehydrate immutable bank spec from serialized mapping."""
        return cls(
            bank_name=payload["bank_name"],
            start_address_bytes=payload["start_address_bytes"],
            parameter_count=payload["parameter_count"],
            parameter_names=tuple(payload["parameter_names"]),
            precision_mode=payload["precision_mode"],
            q_format=payload["q_format"],
            bfp_exponent_bits=payload.get("bfp_exponent_bits", 5),
            bfp_mantissa_bits=payload.get("bfp_mantissa_bits", 11),
            writable=payload.get("writable", True),
            reset_value=payload.get("reset_value", 0),
        )


@dataclass(frozen=True)
class MMIOUpdateSpec:
    """Contract for dynamic updates through bus-mapped control registers."""

    bus_protocol: str
    banks: tuple[ParameterBankSpec, ...] = field(default_factory=tuple)
    read_data_width: int = 32
    write_data_width: int = 32
    address_width_bits: int = 32
    bank_name_width: int = 32
    supports_burst: bool = True
    supports_partial_write: bool = False
    trap: TrapSpec = field(default_factory=TrapSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bus_protocol", _normalise_bus_protocol(self.bus_protocol))

        if not self.banks:
            raise ValueError("MMIOUpdateSpec requires at least one ParameterBankSpec")
        if any(not isinstance(bank, ParameterBankSpec) for bank in self.banks):
            raise ValueError("banks must contain only ParameterBankSpec instances")
        bank_names = [bank.bank_name for bank in self.banks]
        if len(bank_names) != len(set(bank_names)):
            raise ValueError("bank names must be unique")

        if self.read_data_width not in {8, 16, 32, 64}:
            raise ValueError("read_data_width must be one of 8, 16, 32, 64")
        if self.write_data_width not in {8, 16, 32, 64}:
            raise ValueError("write_data_width must be one of 8, 16, 32, 64")
        if self.address_width_bits < 12 or self.address_width_bits > 64:
            raise ValueError("address_width_bits must be between 12 and 64")
        if self.bank_name_width < 8 or self.bank_name_width > 64:
            raise ValueError("bank_name_width must be between 8 and 64")
        if self.bank_name_width < len(max(self.banks, key=lambda item: len(item.bank_name)).bank_name):
            raise ValueError("bank_name_width too small for longest bank name")

        overlaps = _validate_banks_do_not_overlap(self.banks)
        if overlaps:
            raise ValueError("Parameter banks must not overlap")

    @property
    def has_traps(self) -> bool:
        """Whether the contract requires overflow/saturation signalling."""
        return self.trap.enabled

    @property
    def total_address_space_bytes(self) -> int:
        """Total MMIO span from min bank start to max bank end."""
        starts = [bank.start_address_bytes for bank in self.banks]
        ends = [bank.end_address_bytes for bank in self.banks]
        return max(ends) - min(starts)

    def to_dict(self) -> dict[str, Any]:
        """Serialise contract for manifest persistence."""
        return {
            "bus_protocol": self.bus_protocol,
            "banks": [bank.to_dict() for bank in self.banks],
            "read_data_width": self.read_data_width,
            "write_data_width": self.write_data_width,
            "address_width_bits": self.address_width_bits,
            "bank_name_width": self.bank_name_width,
            "supports_burst": self.supports_burst,
            "supports_partial_write": self.supports_partial_write,
            "trap": {
                "enabled": self.trap.enabled,
                "action": self.trap.action,
                "flag_register_offset": self.trap.flag_register_offset,
                "sticky": self.trap.sticky,
                "max_flags": self.trap.max_flags,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MMIOUpdateSpec":
        """Rehydrate contract from serialized mapping."""
        return cls(
            bus_protocol=payload["bus_protocol"],
            banks=tuple(ParameterBankSpec.from_dict(bank) for bank in payload["banks"]),
            read_data_width=payload["read_data_width"],
            write_data_width=payload["write_data_width"],
            address_width_bits=payload["address_width_bits"],
            bank_name_width=payload["bank_name_width"],
            supports_burst=payload.get("supports_burst", True),
            supports_partial_write=payload.get("supports_partial_write", False),
            trap=TrapSpec(
                enabled=payload["trap"]["enabled"],
                action=payload["trap"]["action"],
                flag_register_offset=payload["trap"]["flag_register_offset"],
                sticky=payload["trap"]["sticky"],
                max_flags=payload["trap"]["max_flags"],
            ),
        )


def _validate_banks_do_not_overlap(
    banks: tuple[ParameterBankSpec, ...],
) -> bool:
    """Return True if any two bank ranges overlap."""
    ranges = sorted((bank.start_address_bytes, bank.end_address_bytes) for bank in banks)
    for (start_a, end_a), (start_b, end_b) in zip(ranges, ranges[1:]):
        if end_a > start_b:
            return True
    return False
