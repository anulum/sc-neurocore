# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live-control contract facade

"""Live-parameter contracts for FPGA control and hardware parameter updates."""

from __future__ import annotations

from .live_control_ops import (
    MMIORead,
    MMIOWrite,
    _crc32_update_guard,
)
from .live_control_specs import (
    MMIOUpdateSpec,
    ParameterBankSpec,
    TrapSpec,
    _normalise_bus_protocol,
)
from .live_control_types import (
    CONTROL_CLEAR_TRAP,
    CONTROL_COMMIT,
    CONTROL_REGISTER_OFFSETS,
    CONTROL_REGISTER_SPAN_BYTES,
    CONTROL_ROLLBACK,
    CONTROL_UPDATE_VALID,
    STATUS_APPLIED,
    STATUS_BUSY,
    STATUS_CHECKSUM_VALID,
    STATUS_READY,
    STATUS_ROLLBACK_ACK,
    STATUS_SHADOW_LOADED,
    STATUS_TRAP_LATCHED,
    STATUS_UPDATE_ACK,
    TRAP_CHECKSUM_MISMATCH,
    TRAP_INVALID_SELECTION,
    TRAP_PARTIAL_WRITE,
    TRAP_READ_ONLY_BANK,
    TRAP_STAGED_OVERFLOW,
    TRAP_STAGED_UNDERFLOW,
    UPDATE_CHECKSUM_ALGORITHM,
    BusProtocol,
    MMIOReadPurpose,
    MMIOWritePurpose,
    PrecisionMode,
    TrapAction,
)

__all__ = [
    "CONTROL_CLEAR_TRAP",
    "CONTROL_COMMIT",
    "CONTROL_REGISTER_OFFSETS",
    "CONTROL_REGISTER_SPAN_BYTES",
    "CONTROL_ROLLBACK",
    "CONTROL_UPDATE_VALID",
    "MMIORead",
    "MMIOReadPurpose",
    "MMIOUpdateSpec",
    "MMIOWrite",
    "MMIOWritePurpose",
    "ParameterBankSpec",
    "PrecisionMode",
    "STATUS_APPLIED",
    "STATUS_BUSY",
    "STATUS_CHECKSUM_VALID",
    "STATUS_READY",
    "STATUS_ROLLBACK_ACK",
    "STATUS_SHADOW_LOADED",
    "STATUS_TRAP_LATCHED",
    "STATUS_UPDATE_ACK",
    "TRAP_CHECKSUM_MISMATCH",
    "TRAP_INVALID_SELECTION",
    "TRAP_PARTIAL_WRITE",
    "TRAP_READ_ONLY_BANK",
    "TRAP_STAGED_OVERFLOW",
    "TRAP_STAGED_UNDERFLOW",
    "TrapAction",
    "TrapSpec",
    "UPDATE_CHECKSUM_ALGORITHM",
    "BusProtocol",
]
