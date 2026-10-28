# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native reciprocal root equation execution
"""Run source-bound reciprocal roots through direct and CIRCT-lowered native RTL."""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
import pytest
from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine

require_engine()


@pytest.mark.parametrize(
    "width,fraction,method",
    [
        (8, 4, "map"),
        (32, 16, "map"),
        (32, 24, "map"),
        (16, 8, "euler"),
        (16, 8, "rk4"),
        (16, 8, "gauss_seidel"),
        (16, 8, "exp_euler"),
    ],
)
def test_native_rsqrt_replays_and_runs_against_the_clocked_compiler(
    width: int, fraction: int, method: str, tmp_path: Path
) -> None:
    """Source-bound native graphs and direct RTL agree under changing drive and real reset."""
    drive = "rsqrt(v*v+1)" if method == "map" else "-v+rsqrt(v*v+1)+I"
    model = EquationNeuron(
        {"v": drive},
        state={"v": -0.25},
        method=method,
        dt=1.0 if method == "map" else 0.125,
        threshold="rsqrt(v*v+1)>.5",
        reset={"v": "rsqrt(I*I+1)/8"},
        spike_value="rsqrt(v*v+1)",
    )
    native = compile_to_ir(model, "rsqrt_native", width, fraction, function_lowering="v2")
    assert recompile_ir(native.graph.to_text()).graph.to_text() == native.graph.to_text()
    legacy = compile_to_verilog(model, "rsqrt_legacy", width, fraction, function_lowering="v2")
    outputs = dict(native.output_widths)
    declarations = " ".join(
        f"wire [{bits - 1}:0] n_{name},l_{name};" for name, bits in outputs.items()
    )
    connections = {
        p: ",".join(
            [
                ".clk(clk)",
                ".rst_n(rst_n)",
                ".I_t(I_t)",
                *(f".{name}({p}_{name})" for name in outputs),
            ]
        )
        for p in ("n", "l")
    }
    compare = " || ".join(f"n_{name} !== l_{name}" for name in outputs)
    display = (
        '$display("OBS '
        + " ".join("%h" for _ in outputs)
        + '",'
        + ",".join(f"n_{name}" for name in outputs)
        + ");"
    )
    steps = []
    for i, current in enumerate([-0.75, 0.0, 0.75, 0.5, -0.5, 0.0, 0.75, -0.75]):
        code = round(current * (1 << fraction)) & ((1 << width) - 1)
        reset = "rst_n=0;#1;rst_n=1;#1;" if i == 4 else ""
        steps.append(
            f'{reset} I_t={width}\'h{code:x};#2;clk=1;#1;if ({compare}) $fatal(1,"rsqrt mismatch");{display}clk=0;#1;'
        )
    bench = (
        f"module tb;reg clk=0,rst_n=1;reg [{width - 1}:0] I_t=0;{declarations}"
        f"rsqrt_native n({connections['n']});rsqrt_legacy l({connections['l']});"
        "initial begin #1;rst_n=0;#1;rst_n=1;\n" + "\n".join(steps) + "\n$finish;end endmodule\n"
    )
    top, tb, binary = tmp_path / "rsqrt.sv", tmp_path / "tb.sv", tmp_path / "run"
    top.write_text(native.graph.emit_sv() + "\n" + legacy)
    tb.write_text(bench)
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    actual = subprocess.run(
        ["vvp", str(binary)], check=True, capture_output=True, text=True, timeout=90
    )
    observations = [line for line in actual.stdout.splitlines() if line.startswith("OBS ")]
    assert len(observations) == len(steps)
    if receipt_dir := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(receipt_dir) / f"rsqrt-{width}-{fraction}-{method}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(native.graph.to_text())
        (case / "graph.mlir").write_text(native.graph.emit_mlir())
        (case / "tb.sv").write_text(bench + legacy)
        (case / "expected.txt").write_text("\n".join(observations) + "\n")


@pytest.mark.parametrize("expression", ["rsqrt()", "rsqrt(I,1)", "rsqrt(I,x=1)"])
def test_native_root_refuses_wrong_call_signatures(expression: str) -> None:
    """Source-bound native graphs preserve the unary positional-only contract."""
    with pytest.raises(ValueError):
        compile_to_ir(EquationNeuron({"v": expression}, method="map"), function_lowering="v2")


def test_native_root_requires_v2_opt_in() -> None:
    """Legacy source admission never substitutes an unimplemented reciprocal-root table."""
    with pytest.raises(ValueError, match="needs function_lowering='v2'"):
        compile_to_ir(EquationNeuron({"v": "rsqrt(I)"}, method="map"))


def test_native_root_replays_dimensional_source() -> None:
    """Inverse half-power units and original display quantities survive source replay."""
    from sc_neurocore.neurons._units import UNIT_REGISTRY as u

    model = EquationNeuron(
        {"v": "gain*rsqrt(offset)"},
        state={"v": 0 * u.volt},
        parameters={"gain": 1 * u.volt**1.5 / u.second, "offset": 4000 * u.millivolt},
        dt=0.125 * u.second,
        units="strict",
    )
    compiled = compile_to_ir(model, function_lowering="v2")
    assert recompile_ir(compiled.graph.to_text()).graph.to_text() == compiled.graph.to_text()
