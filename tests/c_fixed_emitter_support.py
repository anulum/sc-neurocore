# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_c_fixed_emitter.py

"""Shared imports and the ``_c`` helper of the bit-exact C/Rust expression emitter tests.

Those tests assert on the *shape* of the emitted source (operators, helper calls,
LUT statements, free-variable capture, language differences); the numeric check
that the emitted C reproduces the Verilog RTL bit for bit lives in
``tests/test_bit_true_cosim_bit_true_cosim.py`` (Icarus Verilog co-simulation).
"""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.c_fixed_emitter import (
    _CFixedExprEmitter,
    emit_c_fixed_expr,
    signed_q,
)
from sc_neurocore.compiler.verilog_compiler_config import Q88

Q = Q88(data_width=16, fraction=8)


def _c(
    expr: str,
    state: dict[str, str] | None = None,
    params: dict[str, int] | None = None,
    *,
    lang: str = "c",
    input_ref: str = "I_t",
    lut_start: int = 0,
) -> tuple[str, list[str], dict[str, list[int]], list[str], int, bool]:
    return emit_c_fixed_expr(
        expr, state or {}, params or {}, Q, lang=lang, input_ref=input_ref, lut_start=lut_start
    )


__all__ = ["pytest", "_CFixedExprEmitter", "emit_c_fixed_expr", "signed_q", "Q88", "Q", "_c"]
