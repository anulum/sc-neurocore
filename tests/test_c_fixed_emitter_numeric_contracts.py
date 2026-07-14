# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — C/Rust fixed-point numeric contract tests

"""Focused fail-closed contracts for integer C/Rust expression lowering."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.verilog_compiler_config import Q88

_Q88 = Q88()
_STATE = {"v": "s->v"}


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("v % 0.0", "finite positive numeric literal"),
        ("v % 1e309", "finite positive numeric literal"),
        ("v % 128.0", "exceeds fixed-point maximum"),
        ("v % 0.001", "underflows"),
    ],
)
def test_modulo_rejects_periods_that_cannot_share_the_rtl_contract(
    expression: str,
    message: str,
) -> None:
    """C/Rust lowering rejects non-finite, non-positive, oversized, and zero-code periods."""
    with pytest.raises(ValueError, match=message):
        emit_c_fixed_expr(expression, _STATE, {}, _Q88)
