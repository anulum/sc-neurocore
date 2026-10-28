# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native wide equation product rounding

"""Execute 128-bit products and independently calculated legacy rounding boundaries."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.native_equation import EquationArithmetic
from sc_neurocore.compiler.verilog_compiler_config import Q88
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize("rounding", ["truncate", "nearest", "bankers"])
@pytest.mark.parametrize("fraction", [0, 8, 64])
def test_q64_product_rounding_preserves_wide_bias_overflow_and_signed_ties(
    rounding: str, fraction: int, tmp_path: Path
) -> None:
    """Real 128-bit intermediate words retain both signed endpoint and half-way behaviour."""
    b = ScGraphBuilder("wide_rounding")
    a = EquationArithmetic(b, Q88(64, fraction, rounding=rounding))
    product = b.input("product", "i128")
    b.output("result", a.round_product(product, 128))
    negative = -((1 << 100) + 17)
    b.output("negative", a.constant(negative, 128))
    graph = b.build()
    assert graph.verify() is None
    graph = parse_ir(graph.to_text())
    half = (1 << (fraction - 1)) if fraction else 0
    values = [
        -(1 << 127),
        -(1 << 127) + 1,
        -(1 << fraction) - half,
        -half - 1,
        -half,
        -1,
        0,
        half - 1,
        half,
        half + 1,
        (1 << fraction) + half,
        (1 << 127) - 1,
    ]
    expected = []
    stimulus = []
    for value in values:
        bias = (
            0
            if not fraction or rounding == "truncate"
            else half - int(rounding == "nearest" and value < 0)
        )
        wrapped = (value + bias) & ((1 << 128) - 1)
        if wrapped >= 1 << 127:
            wrapped -= 1 << 128
        rounded = wrapped >> fraction
        if fraction and rounding == "bankers" and value & ((1 << fraction) - 1) == half:
            rounded &= -2
        expected.append(f"OBS {rounded & ((1 << 64) - 1):016x}")
        stimulus.append(
            f'product=128\'h{value & ((1 << 128) - 1):032x}; #1; $display("OBS %h",result);'
        )
    bench = (
        "module tb; reg clk=0,rst_n=1; reg [127:0] product=0; wire [63:0] result; wire [127:0] negative; wide_rounding dut(.*); initial begin #1; "
        + f'if (negative !== 128\'h{negative & ((1 << 128) - 1):032x}) $fatal(1,"wide negative constant differs");\n'
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    top, tb, binary = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(graph.emit_sv(), encoding="utf-8")
    tb.write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    observed = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=90
    ).stdout.splitlines()
    assert [line for line in observed if line.startswith("OBS ")] == expected
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / f"equation-rounding-{rounding}-{fraction}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(graph.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(graph.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench, encoding="utf-8")
        (case / "expected.txt").write_text("\n".join(expected) + "\n", encoding="utf-8")


def test_native_product_refuses_reserved_stochastic_rounding() -> None:
    """A real builder never accepts a product whose rounding needs an absent RNG."""
    builder = ScGraphBuilder("invalid_rounding")
    arithmetic = EquationArithmetic(builder, Q88(16, 8, rounding="stochastic"))
    product = builder.input("product", "i32")
    with pytest.raises(ValueError, match="require truncate, nearest or bankers"):
        arithmetic.round_product(product, 32)
    builder.output("result", product)
    assert len(builder.build()) == 2
