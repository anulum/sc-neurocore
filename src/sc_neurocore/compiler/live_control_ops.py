# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live-control operations

"""Operational classes for memory-mapped I/O transactions."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from .live_control_types import MMIOReadPurpose, MMIOWritePurpose


@dataclass(frozen=True)
class MMIOWrite:
    """One deterministic memory-mapped write in a live-update transaction."""

    address_bytes: int
    value: int
    width_bits: int
    purpose: MMIOWritePurpose

    def __post_init__(self) -> None:
        if not isinstance(self.address_bytes, int) or isinstance(self.address_bytes, bool):
            raise ValueError("address_bytes must be an integer")
        if self.address_bytes < 0 or self.address_bytes % 4 != 0:
            raise ValueError("address_bytes must be non-negative and 4-byte aligned")
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise ValueError("value must be an integer")
        if self.width_bits not in {8, 16, 32, 64}:
            raise ValueError("width_bits must be one of 8, 16, 32, 64")
        if self.value < 0 or self.value >= (1 << self.width_bits):
            raise ValueError("value does not fit the declared write width")


@dataclass(frozen=True)
class MMIORead:
    """One deterministic memory-mapped read in a live-control transaction."""

    address_bytes: int
    width_bits: int
    purpose: MMIOReadPurpose

    def __post_init__(self) -> None:
        if not isinstance(self.address_bytes, int) or isinstance(self.address_bytes, bool):
            raise ValueError("address_bytes must be an integer")
        if self.address_bytes < 0 or self.address_bytes % 4 != 0:
            raise ValueError("address_bytes must be non-negative and 4-byte aligned")
        if self.width_bits not in {8, 16, 32, 64}:
            raise ValueError("width_bits must be one of 8, 16, 32, 64")


def _crc32_update_guard(
    bank_select: int,
    entry_index: int,
    data_lo: int,
    data_hi: int,
) -> int:
    """Return IEEE CRC32 over four little-endian 32-bit update words."""
    payload = struct.pack(
        "<IIII",
        bank_select & 0xFFFF_FFFF,
        entry_index & 0xFFFF_FFFF,
        data_lo & 0xFFFF_FFFF,
        data_hi & 0xFFFF_FFFF,
    )
    return zlib.crc32(payload) & 0xFFFF_FFFF
