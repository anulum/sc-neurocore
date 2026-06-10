# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live-control specifications

"""Immutable contract specifications for live-control parameter banks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from .q_format import QFormat
from .live_control_types import (
    BusProtocol,
    PrecisionMode,
    TrapAction,
    _VALID_PROTOCOLS,
    CONTROL_REGISTER_OFFSETS,
    CONTROL_REGISTER_SPAN_BYTES,
    CONTROL_UPDATE_VALID,
    CONTROL_COMMIT,
    CONTROL_CLEAR_TRAP,
    CONTROL_ROLLBACK,
    STATUS_READY,
    STATUS_BUSY,
    STATUS_UPDATE_ACK,
    STATUS_TRAP_LATCHED,
    STATUS_SHADOW_LOADED,
    STATUS_APPLIED,
    STATUS_ROLLBACK_ACK,
    STATUS_CHECKSUM_VALID,
    TRAP_STAGED_OVERFLOW,
    TRAP_STAGED_UNDERFLOW,
    TRAP_CHECKSUM_MISMATCH,
    TRAP_INVALID_SELECTION,
    TRAP_READ_ONLY_BANK,
    TRAP_PARTIAL_WRITE,
    UPDATE_CHECKSUM_ALGORITHM,
)
from .live_control_ops import MMIOWrite, MMIORead, _crc32_update_guard


def _normalise_bus_protocol(protocol: str) -> BusProtocol:
    protocol = protocol.strip().lower()
    if protocol == "axi_lite":
        protocol = "axi4_lite"
    if protocol not in _VALID_PROTOCOLS:
        raise ValueError(f"Unsupported MMIO protocol: {protocol!r}")
    return cast(BusProtocol, protocol)


@dataclass(frozen=True)
class TrapSpec:
    """Contract for overflow and saturation trap signalling."""

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
        if not isinstance(self.writable, bool):
            raise ValueError("writable must be a bool")
        self.normalise_encoded_word(self.reset_value)

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

    @property
    def encoded_word_max(self) -> int:
        """Largest unsigned storage word accepted for one entry."""
        return (1 << self.entry_width_bits) - 1

    @property
    def signed_code_min(self) -> int:
        """Smallest signed two's-complement code accepted for convenience."""
        return -(1 << (self.entry_width_bits - 1))

    @property
    def signed_code_max(self) -> int:
        """Largest signed two's-complement code accepted for convenience."""
        return (1 << (self.entry_width_bits - 1)) - 1

    def normalise_encoded_word(self, value: int) -> int:
        """Return an unsigned storage word after validating encoded range."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("encoded parameter value must be an integer")
        if 0 <= value <= self.encoded_word_max:
            return value
        if self.signed_code_min <= value <= -1:
            return value & self.encoded_word_max
        raise ValueError(
            f"encoded parameter value must fit the bank entry width ({self.entry_width_bits} bits)"
        )

    def entry_index(self, parameter: int | str) -> int:
        """Resolve a parameter name or numeric entry into a bank index."""
        if isinstance(parameter, bool):
            raise ValueError("parameter index must not be bool")
        if isinstance(parameter, int):
            index = parameter
        elif isinstance(parameter, str):
            try:
                index = self.parameter_names.index(parameter)
            except ValueError as exc:
                raise ValueError(
                    f"unknown parameter {parameter!r} in bank {self.bank_name!r}"
                ) from exc
        else:
            raise ValueError("parameter must be an integer index or parameter name")
        if index < 0 or index >= self.parameter_count:
            raise ValueError("parameter index out of range")
        return index

    def entry_address(self, parameter: int | str) -> int:
        """Return byte address for one parameter entry in this bank."""
        return self.start_address_bytes + self.entry_index(parameter) * self.entry_width_bytes

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
    control_base_address_bytes: int = 0x0

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
        if self.bank_name_width < len(
            max(self.banks, key=lambda item: len(item.bank_name)).bank_name
        ):
            raise ValueError("bank_name_width too small for longest bank name")
        if not isinstance(self.supports_burst, bool):
            raise ValueError("supports_burst must be a bool")
        if not isinstance(self.supports_partial_write, bool):
            raise ValueError("supports_partial_write must be a bool")
        if not isinstance(self.trap, TrapSpec):
            raise ValueError("trap must be a TrapSpec")
        if (
            not isinstance(self.control_base_address_bytes, int)
            or isinstance(self.control_base_address_bytes, bool)
            or self.control_base_address_bytes < 0
            or self.control_base_address_bytes % 4 != 0
        ):
            raise ValueError("control_base_address_bytes must be non-negative and 4-byte aligned")

        overlaps = _validate_banks_do_not_overlap(self.banks)
        if overlaps:
            raise ValueError("Parameter banks must not overlap")
        control_start = self.control_base_address_bytes
        control_end = control_start + CONTROL_REGISTER_SPAN_BYTES
        if _ranges_overlap(
            ((bank.start_address_bytes, bank.end_address_bytes) for bank in self.banks),
            (control_start, control_end),
        ):
            raise ValueError("Control register window must not overlap parameter banks")
        max_address = max([control_end, *(bank.end_address_bytes for bank in self.banks)])
        if max_address > (1 << self.address_width_bits):
            raise ValueError("address_width_bits too small for control and bank address map")

    @property
    def has_traps(self) -> bool:
        """Whether the contract requires overflow/saturation signalling."""
        return self.trap.enabled

    @property
    def total_address_space_bytes(self) -> int:
        """Total MMIO span from min bank start to max bank end."""
        starts = [
            self.control_base_address_bytes,
            *(bank.start_address_bytes for bank in self.banks),
        ]
        ends = [
            self.control_base_address_bytes + CONTROL_REGISTER_SPAN_BYTES,
            *(bank.end_address_bytes for bank in self.banks),
        ]
        return max(ends) - min(starts)

    @property
    def control_register_addresses(self) -> dict[str, int]:
        """Return absolute addresses for the fixed live-control register map."""
        return {
            name: self.control_base_address_bytes + offset
            for name, offset in CONTROL_REGISTER_OFFSETS.items()
        }

    @property
    def status_bits(self) -> dict[str, int]:
        """Return host-visible status-bit assignments."""
        return {
            "ready": STATUS_READY,
            "busy": STATUS_BUSY,
            "update_ack": STATUS_UPDATE_ACK,
            "trap_latched": STATUS_TRAP_LATCHED,
            "shadow_loaded": STATUS_SHADOW_LOADED,
            "applied": STATUS_APPLIED,
            "rollback_ack": STATUS_ROLLBACK_ACK,
            "checksum_valid": STATUS_CHECKSUM_VALID,
        }

    @property
    def control_bits(self) -> dict[str, int]:
        """Return host-writeable control-bit assignments."""
        return {
            "update_valid": CONTROL_UPDATE_VALID,
            "commit": CONTROL_COMMIT,
            "clear_trap": CONTROL_CLEAR_TRAP,
            "rollback": CONTROL_ROLLBACK,
        }

    @property
    def trap_bits(self) -> dict[str, int]:
        """Return deterministic trap-bit assignments for generated parameter banks."""
        return {
            "staged_overflow": TRAP_STAGED_OVERFLOW,
            "staged_underflow": TRAP_STAGED_UNDERFLOW,
            "checksum_mismatch": TRAP_CHECKSUM_MISMATCH,
            "invalid_selection": TRAP_INVALID_SELECTION,
            "read_only_bank": TRAP_READ_ONLY_BANK,
            "partial_write": TRAP_PARTIAL_WRITE,
        }

    @property
    def effective_trap_width(self) -> int:
        """Return trap-vector width needed by host-visible generated traps."""
        return max(self.trap.max_flags, len(self.trap_bits))

    @property
    def trap_clear_mask(self) -> int:
        """Return the mask that clears all host-visible generated trap bits."""
        return (1 << self.effective_trap_width) - 1

    def update_checksum(self, bank_name: str, parameter: int | str, encoded_value: int) -> int:
        """Return deterministic IEEE CRC32 guard for one staged update."""
        bank = self.bank_by_name(bank_name)
        bank_select = self.bank_index(bank_name)
        entry_index = bank.entry_index(parameter)
        encoded_word = bank.normalise_encoded_word(encoded_value)
        return _crc32_update_guard(
            bank_select,
            entry_index,
            encoded_word & 0xFFFF_FFFF,
            encoded_word >> 32,
        )

    def bank_index(self, bank_name: str) -> int:
        """Return deterministic bank-select index for one bank name."""
        for index, bank in enumerate(self.banks):
            if bank.bank_name == bank_name:
                return index
        raise ValueError(f"unknown parameter bank {bank_name!r}")

    def bank_by_name(self, bank_name: str) -> ParameterBankSpec:
        """Return a bank by name or fail closed."""
        return self.banks[self.bank_index(bank_name)]

    def build_update_sequence(
        self,
        bank_name: str,
        parameter: int | str,
        encoded_value: int,
    ) -> tuple[MMIOWrite, ...]:
        """Build an atomic staged MMIO update sequence."""
        bank = self.bank_by_name(bank_name)
        if not bank.writable:
            raise ValueError(f"parameter bank {bank_name!r} is read-only")
        bank_select = self.bank_index(bank_name)
        entry_index = bank.entry_index(parameter)
        encoded_word = bank.normalise_encoded_word(encoded_value)
        addresses = self.control_register_addresses
        writes = [
            MMIOWrite(addresses["bank_select"], bank_select, 32, "select_bank"),
            MMIOWrite(addresses["entry_index"], entry_index, 32, "select_entry"),
            MMIOWrite(addresses["write_data_lo"], encoded_word & 0xFFFF_FFFF, 32, "write_data_lo"),
            MMIOWrite(
                addresses["write_data_hi"],
                (encoded_word >> 32) & 0xFFFF_FFFF,
                32,
                "write_data_hi",
            ),
        ]
        writes.append(
            MMIOWrite(
                addresses["write_checksum"],
                self.update_checksum(bank_name, parameter, encoded_value),
                32,
                "write_checksum",
            )
        )
        writes.append(MMIOWrite(addresses["control"], CONTROL_UPDATE_VALID, 32, "load_shadow"))
        writes.append(MMIOWrite(addresses["control"], CONTROL_COMMIT, 32, "apply_shadow"))
        return tuple(writes)

    def build_apply_sequence(self) -> tuple[MMIOWrite, ...]:
        """Build the host-side write sequence that applies a loaded shadow word."""
        return (
            MMIOWrite(
                self.control_register_addresses["control"], CONTROL_COMMIT, 32, "apply_shadow"
            ),
        )

    def build_rollback_sequence(self) -> tuple[MMIOWrite, ...]:
        """Build the host-side write sequence that restores shadow from active state."""
        return (
            MMIOWrite(
                self.control_register_addresses["control"],
                CONTROL_ROLLBACK,
                32,
                "rollback_shadow",
            ),
        )

    def build_selective_trap_clear_sequence(self, trap_mask: int) -> tuple[MMIOWrite, ...]:
        """Build the host-side sequence for clearing selected sticky traps."""
        if not self.trap.enabled:
            raise ValueError("trap clear sequence requires enabled traps")
        if not isinstance(trap_mask, int) or isinstance(trap_mask, bool):
            raise ValueError("trap_mask must be an integer")
        if trap_mask < 0 or trap_mask > self.trap_clear_mask:
            raise ValueError("trap_mask must select only host-visible trap bits")
        addresses = self.control_register_addresses
        return (
            MMIOWrite(addresses["trap_clear"], trap_mask, 32, "clear_trap"),
            MMIOWrite(addresses["control"], CONTROL_CLEAR_TRAP, 32, "clear_trap"),
        )

    def build_trap_clear_sequence(self) -> tuple[MMIOWrite, ...]:
        """Build the host-side sequence for clearing all generated sticky traps."""
        return self.build_selective_trap_clear_sequence(self.trap_clear_mask)

    def build_readback_sequence(
        self,
        bank_name: str,
        parameter: int | str,
    ) -> tuple[MMIOWrite | MMIORead, ...]:
        """Build the host-side select/readback sequence for one active entry."""
        bank = self.bank_by_name(bank_name)
        bank_select = self.bank_index(bank_name)
        entry_index = bank.entry_index(parameter)
        addresses = self.control_register_addresses
        sequence: list[MMIOWrite | MMIORead] = [
            MMIOWrite(addresses["bank_select"], bank_select, 32, "select_bank"),
            MMIOWrite(addresses["entry_index"], entry_index, 32, "select_entry"),
            MMIORead(addresses["read_data_lo"], 32, "read_active_data_lo"),
        ]
        if bank.entry_width_bits > 32:
            sequence.append(MMIORead(addresses["read_data_hi"], 32, "read_active_data_hi"))
        return tuple(sequence)

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
            "control_base_address_bytes": self.control_base_address_bytes,
            "checksum_algorithm": UPDATE_CHECKSUM_ALGORITHM,
            "control_registers": self.control_register_addresses,
            "control_bits": self.control_bits,
            "status_bits": self.status_bits,
            "trap_bits": self.trap_bits,
            "effective_trap_width": self.effective_trap_width,
            "trap_clear_mask": self.trap_clear_mask,
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
            control_base_address_bytes=payload.get("control_base_address_bytes", 0x0),
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
    return any(end_a > start_b for (_start_a, end_a), (start_b, _end_b) in zip(ranges, ranges[1:]))


def _ranges_overlap(ranges: Any, candidate: tuple[int, int]) -> bool:
    """Return True when any half-open range overlaps the candidate range."""
    candidate_start, candidate_end = candidate
    for range_start, range_end in ranges:
        if range_start < candidate_end and candidate_start < range_end:
            return True
    return False
