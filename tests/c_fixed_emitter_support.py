# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_c_fixed_emitter.py

from __future__ import annotations

"""Unit tests for the bit-exact integer C/Rust expression emitter.

These assert on the *shape* of the emitted source (operators, helper calls, LUT
statements, free-variable capture, language differences); the numeric proof that
the emitted C reproduces the Verilog RTL bit-for-bit lives in
``tests/test_bit_true_cosim.py`` (iverilog co-simulation).
"""
import pytest
from sc_neurocore.compiler.c_fixed_emitter import (
    _CFixedExprEmitter,
    emit_c_fixed_expr,
    signed_q,
)
from sc_neurocore.compiler.verilog_compiler_config import Q88

Q = Q88(data_width=16, fraction=8)


def _c(expr, state=None, params=None, **kw):
    return emit_c_fixed_expr(expr, state or {}, params or {}, Q, **kw)


__all__ = ["pytest", "_CFixedExprEmitter", "emit_c_fixed_expr", "signed_q", "Q88", "Q", "_c"]
