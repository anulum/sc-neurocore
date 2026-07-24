# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_ir_type_checker.py

from __future__ import annotations

"""Tests for Stochastic IR type checking."""
from sc_neurocore.compiler.ir_type_checker import (
    IREdge,
    IRNode,
    SignalType,
    check_ir_types,
    types_compatible,
)

__all__ = ["IREdge", "IRNode", "SignalType", "check_ir_types", "types_compatible"]
