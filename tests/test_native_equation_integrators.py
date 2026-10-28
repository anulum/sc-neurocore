# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation integrator RTL parity

"""Compare every native integration method with real registered equation RTL."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from sc_neurocore.compiler.native_equation import EquationArithmetic
from sc_neurocore.compiler.native_equation.integrators import integrate
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS, EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


def test_native_integrator_refuses_a_changed_unsupported_method() -> None:
    """A mutable public model cannot silently switch an unknown method to RK4."""
    neuron = EquationNeuron({"v": "I"}, method="map")
    neuron.method = "unsupported"
    builder = ScGraphBuilder("invalid_integrator")
    arithmetic = EquationArithmetic(builder, Q88(16, 8))
    source = builder.input("I", "i16")

    def commit(raw: int, width: int) -> int:
        """Narrow an integration candidate using the actual typed builder."""
        return arithmetic.builder.word_resize(raw, "i16")

    with pytest.raises(ValueError, match="unsupported native equation integrator"):
        integrate(neuron, arithmetic, {"v": source}, {"I": source}, commit)
    builder.output("result", source)
    assert len(builder.build()) == 2


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
@pytest.mark.parametrize("overflow", ["wrap", "saturate"])
@pytest.mark.parametrize("rounding", ["truncate", "nearest", "bankers"])
@pytest.mark.parametrize(("width", "fraction"), [(8, 4), (16, 8), (32, 16), (64, 8)])
def test_native_integrator_preserves_coupled_atomic_recurrence(
    method: str, overflow: str, rounding: str, width: int, fraction: int, tmp_path: Path
) -> None:
    """Coupled states expose stage, Gauss-Seidel ordering, dt and saturation differences."""
    neuron = EquationNeuron(
        {"v": "(I - v + w) / 4", "w": "(v - w) / 8"},
        state={"v": -0.5, "w": 0.75},
        dt=0.5,
        method=method,
    )
    q = Q88(width, fraction, overflow=overflow, rounding=rounding)
    b = ScGraphBuilder("native_integrator")
    a = EquationArithmetic(b, q)
    states = {
        name: b.state(a.literal(neuron.initial_state[name], width), f"i{width}")
        for name in neuron.equations
    }
    current = b.input("I_t", f"i{width}")

    def commit(raw: int, raw_width: int) -> int:
        """Commit an actual candidate using the selected signed overflow policy."""
        if overflow == "saturate":
            raw = b.word_clamp(raw, -(1 << (width - 1)), (1 << (width - 1)) - 1, f"i{raw_width}")
        return a.builder.word_resize(raw, f"i{width}")

    candidates = integrate(neuron, a, states, {"I": current}, commit)
    for name in states:
        b.store(states[name], candidates[name])
        b.output(f"{name}_out", states[name])
    graph = b.build()
    assert graph.verify() is None
    graph = parse_ir(graph.to_text())
    legacy = compile_to_verilog(
        neuron, "legacy_integrator", width, fraction, overflow=overflow, rounding=rounding
    )
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    samples = [low, high, 0, 1, -1, 3, -3, high, high, low, 0, 0]
    stimulus = []
    for index, code in enumerate(samples):
        if index == 7:
            stimulus.append(
                '#1; rst_n=0; #1; if (nv !== lv || nw !== lw) $fatal(1, "asynchronous reset mismatch"); rst_n=1;'
            )
        stimulus.append(
            f'I_t={width}\'h{code & ((1 << width) - 1):x}; #2; clk=1; #1; if (nv !== lv || nw !== lw) $fatal(1, "native integrator mismatch: %h %h / %h %h",nv,lv,nw,lw); $display("OBS %h %h",nv,nw); clk=0; #1;'
        )
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg signed [{width - 1}:0] I_t=0; wire [{width - 1}:0] nv,nw,lv,lw; "
        "native_integrator native_dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(nv),.w_out(nw)); "
        "legacy_integrator legacy_dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(lv),.w_out(lw),.spike_out()); "
        "initial begin #1; rst_n=0; #1; rst_n=1;\n"
        + "\n".join(stimulus)
        + "\n$finish; end endmodule\n"
    )
    top, tb, binary = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "run.vvp"
    top.write_text(graph.emit_sv() + "\n" + legacy, encoding="utf-8")
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
    observations = [line for line in observed if line.startswith("OBS ")]
    assert len(observations) == len(samples)
    if root := os.environ.get("SCN_IR_CIRCT_RECEIPT_DIR"):
        case = Path(root) / f"integrator-{method}-{overflow}-{rounding}-{width}-{fraction}"
        case.mkdir(parents=True, exist_ok=True)
        (case / "graph.scir").write_text(graph.to_text(), encoding="utf-8")
        (case / "graph.mlir").write_text(graph.emit_mlir(), encoding="utf-8")
        (case / "tb.sv").write_text(bench + legacy, encoding="utf-8")
        (case / "expected.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
