# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_compiler_live_control.py

from __future__ import annotations


"""Contract tests for compiler live-control specifications."""


import pytest


from sc_neurocore.compiler.live_control import (
    CONTROL_COMMIT,
    CONTROL_CLEAR_TRAP,
    CONTROL_REGISTER_SPAN_BYTES,
    CONTROL_ROLLBACK,
    CONTROL_UPDATE_VALID,
    MMIOUpdateSpec,
    ParameterBankSpec,
    STATUS_APPLIED,
    STATUS_CHECKSUM_VALID,
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
    TrapSpec,
    UPDATE_CHECKSUM_ALGORITHM,
)


__all__ = ['pytest', 'CONTROL_COMMIT', 'CONTROL_CLEAR_TRAP', 'CONTROL_REGISTER_SPAN_BYTES', 'CONTROL_ROLLBACK', 'CONTROL_UPDATE_VALID', 'MMIOUpdateSpec', 'ParameterBankSpec', 'STATUS_APPLIED', 'STATUS_CHECKSUM_VALID', 'STATUS_ROLLBACK_ACK', 'STATUS_SHADOW_LOADED', 'STATUS_TRAP_LATCHED', 'STATUS_UPDATE_ACK', 'TRAP_CHECKSUM_MISMATCH', 'TRAP_INVALID_SELECTION', 'TRAP_PARTIAL_WRITE', 'TRAP_READ_ONLY_BANK', 'TRAP_STAGED_OVERFLOW', 'TRAP_STAGED_UNDERFLOW', 'TrapSpec', 'UPDATE_CHECKSUM_ALGORITHM']

