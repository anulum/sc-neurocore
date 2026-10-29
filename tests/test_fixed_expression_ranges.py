# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed operand ranges checked against generated C, Rust and RTL
"""Check inferred bounds through public emitters and real integer target execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.c_fixed_emitter import emit_c_fixed_expr
from sc_neurocore.compiler.fixed_expression_ranges import CodeRange, expression_code_range
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("I", (-128, 127)),
        ("I/4", (-32, 31)),
        ("clip(I,-1,1)", (-16, 16)),
        ("min(max(I,-1),1)", (-16, 16)),
        ("I+1", (-128, 127)),
        ("I*0", (0, 0)),
        ("abs(clip(I,-1,1))", (0, 16)),
        ("I//4", (-32, 16)),
        ("I % .5", (0, 7)),
        ("8", (-128, -128)),
        ("clip(I,2,-2)", (-32, -32)),
        ("1 if I>0 else -1", (-16, 16)),
    ],
)
def test_encoded_arithmetic_bounds(expression: str, expected: tuple[int, int]) -> None:
    """Narrowing uses encoded reciprocal/product/period arithmetic, never float intervals."""
    bounds = expression_code_range(expression, 8, 4)
    assert (bounds.minimum, bounds.maximum) == expected


def test_fixed_bindings_and_state_shadowing() -> None:
    """A wrapped parameter is fixed; a declared state with the same name is unconstrained."""
    fixed = expression_code_range("p", 8, 4, values={"p": 8.0})
    assert (fixed.minimum, fixed.maximum) == (-128, -128)
    state = expression_code_range("p", 8, 4, values={"p": 8.0}, variables=["p"])
    assert (state.minimum, state.maximum) == (-128, 127)
    pi = expression_code_range("pi", 8, 4)
    assert pi.minimum == pi.maximum == 50
    bound = expression_code_range("pi", 8, 4, variables=["pi"])
    assert (bound.minimum, bound.maximum) == (-128, 127)


def test_base_literals_follow_backend_binary64_conversion() -> None:
    """Base literals follow the existing float conversion; power exponents have separate precision."""
    bounds = expression_code_range("9007199254740993", 8, 0)
    assert bounds.minimum == bounds.maximum == 0


def test_undefined_and_opaque_expressions_are_explicit() -> None:
    """A possibly zero denominator or unknown subtree cannot invent a narrow proof."""
    for expression in ["I/v", "unrecognised(I)", "rand()", "I[0]", "I << 1"]:
        bounds = expression_code_range(expression, 8, 4)
        assert (bounds.minimum, bounds.maximum) == (-128, 127)
        assert bounds.notes


@pytest.mark.parametrize("expression", ["I/0", "I%0", "I%.001", "I%8", "I//3", "I//1.5"])
def test_invalid_literal_arithmetic_is_refused(expression: str) -> None:
    """Invalid or unrepresentable divisors cannot acquire numerical evidence."""
    with pytest.raises(ValueError):
        expression_code_range(expression, 8, 4)


@pytest.mark.parametrize(
    "width,fraction,rounding,lowering",
    [
        (1, 0, "truncate", "v2"),
        (8, 8, "truncate", "v2"),
        (8, 4, "stochastic", "v2"),
        (8, 4, "truncate", "unknown"),
    ],
)
def test_invalid_argument_format_and_policy_are_refused(
    width: int, fraction: int, rounding: str, lowering: str
) -> None:
    """An unsupported geometry or arithmetic policy cannot acquire a range contract."""
    with pytest.raises(ValueError):
        expression_code_range("I", width, fraction, rounding=rounding, function_lowering=lowering)


def test_code_range_rejects_reversed_endpoints() -> None:
    """An empty numeric interval is represented by domain exclusion, never reversed bounds."""
    with pytest.raises(ValueError, match="ordered integer"):
        CodeRange(2, 1)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("(I>0)", (0, 16)),
        (".25**.5", (8, 8)),
        ("2**1.5", (45, 45)),
        ("2**p", (45, 45)),
        ("floor(.75)", (0, 0)),
    ],
)
def test_boolean_and_constant_function_results_are_encoded(
    expression: str, expected: tuple[int, int]
) -> None:
    """Constant units and independently coded powers fold to actual output words."""
    domain = expression_code_range(expression, 8, 4, values={"p": 1.5})
    assert (domain.minimum, domain.maximum) == expected


@pytest.mark.parametrize(
    "expression", ["-(rand())", "rand()+I", "I+rand()", "exp(rand())", "rand() if I>0 else I"]
)
def test_opaque_subexpressions_propagate_conservative_bounds(expression: str) -> None:
    """Arithmetic cannot narrow an opaque result by assuming its unknown value is zero."""
    domain = expression_code_range(expression, 8, 4)
    assert (domain.minimum, domain.maximum) == (-128, 127) and domain.notes


def test_legacy_constant_function_uses_the_existing_table_grid() -> None:
    """Legacy constant folding reads the same table rather than assuming a real square root."""
    from sc_neurocore.compiler.lut_accuracy import lut_read_int

    domain = expression_code_range("sqrt(.25)", 8, 4, function_lowering="lut-v1")
    assert domain.minimum == domain.maximum == lut_read_int("sqrt", 4, 8, 4)


def test_a_new_unit_is_not_folded_as_a_legacy_table() -> None:
    """An unavailable legacy call is opaque and still refused by the real compiler."""
    domain = expression_code_range("arctan(.5)", 8, 4, function_lowering="lut-v1")
    assert (domain.minimum, domain.maximum) == (-128, 127) and domain.notes
    with pytest.raises(ValueError, match="v2"):
        compile_to_verilog(
            EquationNeuron({"v": "arctan(.5)"}, method="map"), data_width=8, fraction=4
        )


@pytest.mark.parametrize("rounding", ["truncate", "nearest"])
@pytest.mark.parametrize("width,fraction", [(6, 3), (8, 2), (32, 29)])
@pytest.mark.parametrize("language", ["c", "rust"])
def test_every_small_input_stays_inside_the_emitted_argument_hull(
    rounding: str, width: int, fraction: int, language: str, tmp_path: Path
) -> None:
    """Execute both targets over complete small words and critical wide operands.

    Parameters
    ----------
    rounding : str
        Admitted bit-true kernel multiply policy.
    width, fraction : int
        Signed argument format; wide formats use critical code pairs.
    language : str
        Real C or Rust emitter and toolchain.
    tmp_path : Path
        Directory for generated sources and executable.
    """
    expressions = [
        "I/4",
        "I*.375",
        "I*y",
        "(I+I)*(y-1)",
        "I/(abs(y)+1)",
        "clip(I,-1,1)",
        "clip(I,1,-1)",
        "abs(I)",
        "I//2",
        "I%.5",
        "min(I,y,.5)",
        "max(I,y,-.5)",
        "I**2",
        "I**3",
        "I**-1",
        "sqrt(abs(I))",
        "I if I>y else y",
        "-(I+y)",
        "+I-y",
        "9007199254740993",
    ]
    q = Q88(data_width=width, fraction=fraction, rounding=rounding, function_lowering="v2")
    foundation = generate_bittrue_kernel_from_neuron(
        EquationNeuron({"v": "I"}, method="map", dt=1.0),
        "base",
        data_width=width,
        fraction=fraction,
        rounding=rounding,
        function_lowering="v2",
        language=language,
    ).split("typedef struct" if language == "c" else "pub struct", 1)[0]
    functions = []
    bounds = []
    for index, expression in enumerate(expressions):
        emitted, statements, tables, free, _, _ = emit_c_fixed_expr(
            expression, {}, {}, q, lang=language
        )
        assert set(free) <= {"y"}
        arrays = "\n".join(
            (
                f"static const int64_t {name}[]={{" + ",".join(map(str, values)) + "};"
                if language == "c"
                else f"const {name}: [i64;{len(values)}]=[" + ",".join(map(str, values)) + "];"
            )
            for name, values in tables.items()
        )
        functions.append(
            (
                f"int64_t f{index}(int64_t I_t,int64_t y){{(void)y;"
                if language == "c"
                else f"fn f{index}(I_t:i64,y:i64)->i64{{let _=y;"
            )
            + arrays
            + "\n"
            + "\n".join(statements)
            + f"return sc_wrap({emitted},{width});}}"
        )
        bounds.append(expression_code_range(expression, width, fraction, rounding=rounding))
    calls = "".join(
        (
            f'printf("{index} %lld\\n",(long long)f{index}(x,y));'
            if language == "c"
            else f'println!("{index} {{}}",f{index}(x,y));'
        )
        for index in range(len(expressions))
    )
    source = tmp_path / ("arguments.c" if language == "c" else "arguments.rs")
    binary = tmp_path / "arguments"
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    scale = 1 << fraction
    axis = (
        list(range(low, high + 1))
        if width <= 8
        else sorted(
            {
                low,
                low + 1,
                low // 2,
                -scale - 1,
                -scale,
                -scale + 1,
                -1,
                0,
                1,
                scale - 1,
                scale,
                scale + 1,
                high // 2,
                high - 1,
                high,
            }
        )
    )
    array = ",".join(map(str, axis))
    main = (
        f"\nint main(void){{int64_t axis[]={{{array}}};for(unsigned a=0;a<{len(axis)};a++)for(unsigned b=0;b<{len(axis)};b++){{int64_t x=axis[a],y=axis[b];"
        + calls
        + "}return 0;}\n"
        if language == "c"
        else f"fn main(){{let axis:[i64;{len(axis)}]=[{array}];for &x in &axis{{for &y in &axis{{"
        + calls
        + "}}}\n"
    )
    source.write_text(
        foundation
        + ("#include <stdio.h>\n" if language == "c" else "")
        + "\n".join(functions)
        + main
    )
    compiler = (
        ["cc", "-std=c11", "-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
        if language == "c"
        else ["rustc", "-C", "overflow-checks=on", "-O"]
    )
    subprocess.run(
        [*compiler, str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
    assert not result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == len(axis) ** 2 * len(expressions)
    for line in lines:
        index, code = map(int, line.split())
        domain = bounds[index]
        assert domain.minimum <= code <= domain.maximum, (expressions[index], code, domain)


def test_bankers_intervals_and_point_bounds_match_clocked_verilog(tmp_path: Path) -> None:
    """The RTL-only banker policy is checked on every signed input pair, including ties."""
    expressions = ["clip(I,-1,1)*.375", "clip(I,-1,1)/4", "clip(I,-1,1)*clip(y,-.5,.5)", "(I+I)*y"]
    model = EquationNeuron(
        {**{f"state_{index}": text for index, text in enumerate(expressions)}, "y": "I"},
        method="map",
    )
    rtl, bench, binary = tmp_path / "arguments.v", tmp_path / "tb.v", tmp_path / "sim"
    rtl.write_text(
        compile_to_verilog(
            model, "arguments", 6, 3, rounding="bankers", overflow="wrap", function_lowering="v2"
        )
    )
    wires = "\n".join(f"wire signed [5:0] state_{index}_out;" for index in range(len(expressions)))
    ports = ",".join(f".state_{index}_out(state_{index}_out)" for index in range(len(expressions)))
    outputs = ",".join(f"$signed(state_{index}_out)" for index in range(len(expressions)))
    bench.write_text(
        "module tb; reg clk=0; always #5 clk=~clk; reg rst_n=0; reg signed [5:0] drive=0; wire spike_out; wire signed [5:0] y_out;"
        + wires
        + "arguments dut(.clk(clk),.rst_n(rst_n),.I_t(drive),.spike_out(spike_out),.y_out(y_out),"
        + ports
        + '); integer x,y; initial begin #23;rst_n=1; for(x=-32;x<32;x=x+1)for(y=-32;y<32;y=y+1)begin @(negedge clk);drive=y;@(posedge clk);#1;@(negedge clk);drive=x;@(posedge clk);#1; $display("%0d %0d %0d %0d %0d %0d",x,y,'
        + outputs
        + "); end $finish;end endmodule\n"
    )
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(binary), str(rtl), str(bench)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        ["vvp", str(binary)], check=True, capture_output=True, text=True, timeout=30
    )
    rows = []
    for line in result.stdout.splitlines():
        if "$finish called" in line:
            continue
        fields = line.split()
        assert len(fields) == 6, line
        rows.append(list(map(int, fields)))
    assert len(rows) == 4096
    bounds = [expression_code_range(text, 6, 3, rounding="bankers") for text in expressions]
    for x, y, *codes in rows:
        for text, interval, code in zip(expressions, bounds, codes, strict=True):
            assert interval.minimum <= code <= interval.maximum
            point = expression_code_range(
                text, 6, 3, rounding="bankers", values={"I": x / 8, "y": y / 8}
            )
            assert point.minimum == point.maximum == code, (text, x, y, point, code)
