# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Verilog expression lowering contracts

"""Focused fixed-point Verilog expression lowering contracts."""

from .verilog_expr_emitter_support import *


@pytest.mark.parametrize(
    ("expression", "q", "message"),
    [
        ("v % 128.0", Q88(), "exceeds fixed-point maximum"),
        ("v % 0.001", Q88(), "underflows"),
    ],
)
def test_modulo_rejects_unrepresentable_or_unsigned_contracts(
    expression: str,
    q: Q88,
    message: str,
) -> None:
    """Modulo requires signed state and a positive representable literal period."""
    with pytest.raises(ValueError, match=message):
        _emit_expr(expression, _STATE_VARS, {}, q)


def test_modulo_rejects_unsigned_state_before_divisor_validation() -> None:
    """Unsigned state fails at the modulo contract even for a named period."""
    with pytest.raises(ValueError, match="requires signed"):
        _emit_expr(
            "v % period",
            _STATE_VARS,
            {"period": "P_PERIOD"},
            Q88(signed=False),
        )
