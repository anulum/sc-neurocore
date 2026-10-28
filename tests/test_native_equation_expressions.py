# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation expression RTL parity

"""Execute native expression/state graphs beside the public equation compiler's RTL."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.native_equation import EquationArithmetic, EquationExpressions
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize("rounding", ["truncate", "nearest", "bankers"])
@pytest.mark.parametrize(("width", "fraction"), [(8, 4), (16, 8), (32, 16), (64, 8)])
@pytest.mark.parametrize(
    "expression",
    [
        "v + I",
        "+v + pi",
        "v if I else -v",
        "(v + I) * (v - I)",
        "(v + I) / 2",
        "(v + I) / (abs(I) + 1)",
        "v // 2",
        "(v + I) % 3",
        "-v if v + I > 0 else I - v",
        "I * (v + I > 0)",
        "abs(v + I)",
        "clip(v + I, I, -I)",
        "max(v + I, -I, v - I, 0)",
        "min(v + I, -I, v - I, 0)",
        "(v + I) ** 3",
    ],
)
def test_native_equation_expression_matches_clocked_public_compiler(
    expression: str,
    width: int,
    fraction: int,
    rounding: str,
    tmp_path: Path,
    lowering: str = "lut-v1",
) -> None:
    """Observe real feedback/reset and wide arithmetic on every advertised product rounding mode."""
    q = Q88(width, fraction, overflow="wrap", rounding=rounding, function_lowering=lowering)
    neuron = EquationNeuron({"v": expression}, dt=1.0, method="map", state={"v": -0.5})
    b = ScGraphBuilder("native_expression")
    a = EquationArithmetic(b, q)
    state = b.state(a.literal(-0.5, width), f"i{width}")
    current = b.input("I_t", f"i{width}")
    e = EquationExpressions(a, {"v": state, "I": current})
    b.store(state, e.lower(neuron.equations["v"]))
    b.output("v_out", state)
    graph = b.build()
    assert graph.verify() is None
    restored = parse_ir(graph.to_text())
    assert restored.to_text() == graph.to_text()
    assert restored.emit_sv() == graph.emit_sv()
    assert restored.emit_mlir() == graph.emit_mlir()
    legacy = compile_to_verilog(
        neuron,
        "legacy_expression",
        width,
        fraction,
        overflow="wrap",
        rounding=rounding,
        function_lowering=lowering,
    )
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    samples = [low, -19, -1, 0, 1, 9, high, 3, -3, 0]
    samples = [max(low, min(high, code)) for code in samples]
    stimulus = []
    for code in samples:
        stimulus.append(
            f'I_t={width}\'h{code & ((1 << width) - 1):x}; #2; clk=1; #1; if (observed !== reference) $fatal(1, "native expression mismatch: %h %h", observed, reference); $display("OBS %h", observed); clk=0; #1;'
        )
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg signed [{width - 1}:0] I_t=0; wire [{width - 1}:0] observed,reference; wire spike; "
        "native_expression native_dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(observed)); "
        "legacy_expression legacy_dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(reference),.spike_out(spike)); "
        "initial begin #1; rst_n=0; #1; rst_n=1;\n"
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    source, harness, binary = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    source.write_text(restored.emit_sv() + "\n" + legacy, encoding="utf-8")
    harness.write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(source), str(harness)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    observations = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=90
    ).stdout.splitlines()
    assert len([line for line in observations if line.startswith("OBS ")]) == len(samples)
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = (
            Path(root)
            / f"expression-{expression.encode().hex()}-{width}-{fraction}-{rounding}-{lowering}"
        )
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(restored.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(restored.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench + legacy, encoding="utf-8")
        (case / "expected.txt").write_text(
            "\n".join(line for line in observations if line.startswith("OBS ")) + "\n",
            encoding="utf-8",
        )


@pytest.mark.parametrize("expression", ["I ** 0.5", "I ** (1.0 / 3.0)"])
@pytest.mark.parametrize("lowering", ["lut-v1", "v2"])
def test_fractional_roots_match_the_clocked_compiler(
    expression: str, lowering: str, tmp_path: Path
) -> None:
    """Both root kernels handle signed endpoints, zero and repeated state commits."""
    test_native_equation_expression_matches_clocked_public_compiler(
        expression, 16, 8, "bankers", tmp_path, lowering
    )


def test_chained_comparison_keeps_both_signed_predicates(tmp_path: Path) -> None:
    """A bounded interval predicate must hold at both endpoints before selecting a branch."""
    test_native_equation_expression_matches_clocked_public_compiler(
        "I if -1 < v < 1 else -I", 16, 8, "bankers", tmp_path
    )


@pytest.mark.parametrize(
    "expression,message,fraction",
    [
        ("v == I", "comparisons require", 8),
        ("v and I", "operands must be comparisons", 8),
        ("v < I", "represents 1.0", 15),
        ("missing", "unbound native equation identifier", 8),
        ("~v", "only unary", 8),
        ("[v]", "unsupported native equation AST", 8),
        ("v // 3", "floor divisor", 8),
        ("v // I", "floor divisor", 8),
        ("v % 0", "modulo divisor must", 8),
        ("v % I", "modulo divisor must", 8),
        ("v % 0.00001", "modulo divisor underflows", 8),
        ("v & I", "unsupported native equation operator", 8),
        ("v ** 1.25", "powers require", 8),
    ],
)
def test_public_expression_lowerer_refuses_unsupported_arithmetic(
    expression: str, message: str, fraction: int
) -> None:
    """Invalid public expressions fail explicitly against a genuine typed graph builder."""
    builder = ScGraphBuilder("invalid_expression")
    arithmetic = EquationArithmetic(builder, Q88(16, fraction))
    bindings = {name: builder.input(name, "i16") for name in ("v", "I")}
    with pytest.raises(ValueError, match=message):
        EquationExpressions(arithmetic, bindings).lower(expression)


def test_shared_product_retains_one_collapse_across_context_widths(tmp_path: Path) -> None:
    """Reusing a public AST value widens the rounded word rather than recomputing a product."""
    from tests.test_native_program import _observe

    builder = ScGraphBuilder("native_program")
    arithmetic = EquationArithmetic(builder, Q88(64, 8))
    expressions = EquationExpressions(arithmetic, {"v": builder.input("arg", "i64")})
    node = ast.parse("v * 0.5", mode="eval").body
    narrow = expressions.value(node, 64)
    wide = builder.word_resize(expressions.value(node, 128), "i64")
    builder.output("result", builder.word_binary("sub", wide, narrow, "i64"))
    cases = [(-257,), (-1,), (0,), (1,), (257,)]
    graph = builder.build()
    assert graph.to_text().count(" = sc.word mul ") == 1
    _observe(tmp_path, "shared-product-context", graph, ["arg"], cases, [0] * len(cases))
