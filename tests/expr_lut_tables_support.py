# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_expr_lut_tables.py

from __future__ import annotations

"""Tests for the target-independent expression LUT tables.

These bind the shared numerics that every lowering backend must agree on and
cross-check that the Verilog emitter still produces byte-identical values after
delegating to this module.
"""
import ast
import math
from sc_neurocore.compiler import expr_lut_tables as tables
from sc_neurocore.compiler.equation_compiler import Q88, _VerilogExprEmitter

__all__ = ["ast", "math", "tables", "Q88", "_VerilogExprEmitter"]
