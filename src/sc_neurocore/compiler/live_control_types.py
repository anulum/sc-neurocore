# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live-control types

"""Type definitions for live-control contracts."""

from __future__ import annotations

from typing import Literal

BusProtocol = Literal["axi4_lite", "pcie"]
_VALID_PROTOCOLS: set[str] = {"axi4_lite", "pcie"}
PrecisionMode = Literal["q", "bfp"]
TrapAction = Literal["hold", "saturate", "clip", "halt", "interrupt"]

MMIOWritePurpose = Literal[
    "select_bank",
    "select_entry",
    "write_data_lo",
    "write_data_hi",
    "write_checksum",
    "load_shadow",
    "apply_shadow",
    "rollback_shadow",
    "commit_update",
    "clear_trap",
]

MMIOReadPurpose = Literal[
    "read_status",
    "read_trap_status",
    "read_active_data_lo",
    "read_active_data_hi",
]

UPDATE_CHECKSUM_ALGORITHM = "crc32-ieee-le-4x32"

# Control bits
CONTROL_UPDATE_VALID = 0x1
CONTROL_COMMIT = 0x2
CONTROL_CLEAR_TRAP = 0x4
CONTROL_ROLLBACK = 0x8

# Status bits
STATUS_READY = 0x1
STATUS_BUSY = 0x2
STATUS_UPDATE_ACK = 0x4
STATUS_TRAP_LATCHED = 0x8
STATUS_SHADOW_LOADED = 0x10
STATUS_APPLIED = 0x20
STATUS_ROLLBACK_ACK = 0x40
STATUS_CHECKSUM_VALID = 0x80

# Trap bits
TRAP_STAGED_OVERFLOW = 0x1
TRAP_STAGED_UNDERFLOW = 0x2
TRAP_CHECKSUM_MISMATCH = 0x4
TRAP_INVALID_SELECTION = 0x8
TRAP_READ_ONLY_BANK = 0x10
TRAP_PARTIAL_WRITE = 0x20

CONTROL_REGISTER_OFFSETS: dict[str, int] = {
    "control": 0x00,
    "status": 0x04,
    "bank_select": 0x08,
    "entry_index": 0x0C,
    "write_data_lo": 0x10,
    "write_data_hi": 0x14,
    "trap_status": 0x18,
    "trap_clear": 0x1C,
    "write_checksum": 0x20,
    "read_data_lo": 0x24,
    "read_data_hi": 0x28,
}
CONTROL_REGISTER_SPAN_BYTES = 0x2C
