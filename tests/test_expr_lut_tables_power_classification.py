# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared power classification and original literal semantics
"""Check the public classifier used by all maintained expression compilers."""

from __future__ import annotations

import ast
import math

import pytest

from sc_neurocore.compiler.c_expr_emitter import emit_c_expr
from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.expr_lut_tables import classify_power
from sc_neurocore.compiler.verilog_compiler_config import Q88


@pytest.mark.parametrize(
    "source,kind,value",
    [
        ("2.0", "multiply", 2),
        ("3.0", "multiply", 3),
        ("8", "multiply", 8),
        ("0.5", "sqrt", 0.5),
        ("1/3", "cbrt", 1 / 3),
        ("0.5000005", "power", 0.5000005),
        ("pi", "power", math.pi),
        ("+pi", "power", math.pi),
        ("-(1+2)", "power", -3),
        ("-+9007199254740993", "power", -9007199254740993),
        ("w+1", "power", None),
    ],
)
def test_power_classification_retains_original_exponent(
    source: str, kind: str, value: float | None
) -> None:
    """Exact roots, dynamic words, named constants and huge integers stay distinct."""
    assert classify_power(ast.parse(source, mode="eval").body, "v2") == (kind, value)


def test_legacy_root_tolerance_and_shadowed_names_remain_explicit() -> None:
    """Only the legacy path recognizes an approximate root; bound pi stays dynamic."""
    root = ast.parse("0.5000005", mode="eval").body
    assert classify_power(root, "lut-v1")[0] == "sqrt"
    assert classify_power(root, "v2")[0] == "power"
    assert classify_power(ast.Name(id="pi"), "v2", {"pi"}) == ("power", None)
    with pytest.raises(ValueError, match="power lowering"):
        classify_power(root, "unknown")


@pytest.mark.parametrize("source", ["True", "1e309", "-1e309", "1e308*1e308"])
def test_public_expression_compilers_refuse_invalid_literal_exponents(source: str) -> None:
    """No floating or fixed expression frontend silently admits Boolean or infinite powers."""
    with pytest.raises(ValueError, match="exponent"):
        emit_c_expr(f"v**({source})", {"v"})
    for language in ("c", "rust"):
        with pytest.raises(ValueError, match="exponent"):
            emit_c_fixed_expr(f"I**({source})", {}, {}, Q88(function_lowering="v2"), lang=language)
