# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native function domains and source replay

"""Exercise legacy function domains, public call refusals and fractional-root replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.compiler.lut_accuracy import lut_read
from sc_neurocore.compiler.native_equation import EquationArithmetic, EquationExpressions
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine
from tests.test_native_program import _observe

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder


@pytest.mark.parametrize("function", ["log", "sqrt"])
def test_native_legacy_domains_execute_endpoint_clamps(function: str, tmp_path: Path) -> None:
    """Persisted native lookup graphs retain negative, zero, interior and upper endpoints."""
    builder = ScGraphBuilder("native_program")
    arithmetic = EquationArithmetic(builder, Q88(16, 8))
    expression = EquationExpressions(arithmetic, {"I": builder.input("arg", "i64")})
    result = expression.lower(f"{function}(I)")
    builder.output("result", builder.word_resize(result, "i64"))
    codes = [-32768, -1, 0, 1, 16, 256, 1024, 32767]
    expected = [int(value) for value in lut_read(function, np.array(codes), 16, 8)]
    _observe(
        tmp_path,
        f"legacy-domain-{function}",
        builder.build(),
        ["arg"],
        [(value,) for value in codes],
        expected,
    )


@pytest.mark.parametrize("expression", ["unknown(I)", "namespace.sqrt(I)"])
def test_native_expression_call_requires_a_supported_plain_name(expression: str) -> None:
    """The public lowerer refuses unknown functions and attribute calls before dispatch."""
    builder = ScGraphBuilder("invalid_call")
    arithmetic = EquationArithmetic(builder, Q88(16, 8))
    source = builder.input("I", "i16")
    with pytest.raises(ValueError, match="supported named function"):
        EquationExpressions(arithmetic, {"I": source}).lower(expression)
    builder.output("result", source)
    assert len(builder.build()) == 2


@pytest.mark.parametrize("expression", ["I ** 0.5", "I ** (1.0 / 3.0)", "sqrt(I)"])
@pytest.mark.parametrize("lowering", ["lut-v1", "v2"])
def test_fractional_root_source_survives_public_native_replay(
    expression: str, lowering: str
) -> None:
    """Both native fractional-root implementations preserve the authored source on replay."""
    model = EquationNeuron({"v": expression}, method="map")
    compiled = compile_to_ir(model, function_lowering=lowering)
    assert compiled.graph.verify() is None
    assert recompile_ir(compiled.graph.to_text()).graph.to_text() == compiled.graph.to_text()


def test_native_function_without_a_legacy_table_requires_v2() -> None:
    """The public expression lowerer refuses a v2-only function under the legacy contract."""
    builder = ScGraphBuilder("legacy_function_refusal")
    arithmetic = EquationArithmetic(builder, Q88(16, 8))
    source = builder.input("I", "i16")
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        EquationExpressions(arithmetic, {"I": source}).lower("sinh(I)")
