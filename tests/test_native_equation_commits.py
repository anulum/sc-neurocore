# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public native equation overflow semantics

"""Observe integrated, reset and graded overflow through actual public compiler RTL."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from sc_neurocore.compiler import compile_to_ir, compile_to_verilog
from sc_neurocore.compiler.native_equation import EquationArithmetic
from sc_neurocore.compiler.native_equation.commits import EquationCommits
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder


@pytest.mark.parametrize("width", [0, -1, True])
def test_native_commit_refuses_invalid_physical_width_without_appending_effects(width: int) -> None:
    """The real builder remains usable after a rejected public commit-policy request."""
    builder = ScGraphBuilder("commit_width")
    value = builder.input("sample", "i16")
    policy = EquationCommits(EquationArithmetic(builder, Q88(16, 8, overflow="trap")))
    with pytest.raises(ValueError, match="positive integer"):
        policy.state(value, width)
    builder.output("result", policy.state(value, 16))
    graph = builder.build()
    assert graph.verify() is None
    assert "sc.check" not in graph.to_text()
    assert "module commit_width" in graph.emit_sv()


@pytest.mark.parametrize("overflow", ["saturate", "wrap", "trap"])
@pytest.mark.parametrize("event", [False, True])
@pytest.mark.parametrize("width", [8, 40, 64])
def test_public_equation_integrated_and_event_overflow(
    overflow: str, event: bool, width: int, tmp_path: Path
) -> None:
    """Both emitters preserve state traps and wrapping reset/graded expressions."""
    neuron = EquationNeuron(
        {"v": "I" if event else "I + I"},
        method="map",
        dt=1.0,
        state={"v": 0.0},
        threshold="v > 0" if event else None,
        reset={"v": "v + v"} if event else None,
        spike_value="v + v" if event else None,
    )
    result = compile_to_ir(neuron, "native_overflow", width, 0, overflow=overflow)
    legacy = compile_to_verilog(neuron, "legacy_overflow", width, 0, overflow=overflow)
    sample = (1 << (width - 2)) + 1
    raw = 2 * sample
    expected = (1 << (width - 1)) - 1 if overflow == "saturate" else raw - (1 << width)
    event_check = (
        f'if (ng !== {width}\'h{expected & ((1 << width) - 1):x} || ns !== 1) $fatal(1,"event mismatch");'
        if event
        else ""
    )
    declarations = f"wire signed [{width - 1}:0] nv,lv; wire ns,ls;"
    ports = {
        prefix: f".clk(clk),.rst_n(rst_n),.I_t(sample),.v_out({prefix}v),.spike_out({prefix}s)"
        for prefix in ("n", "l")
    }
    if event:
        declarations += f"wire signed [{width - 1}:0] ng,lg;"
        ports = {prefix: text + f",.spike_value_out({prefix}g)" for prefix, text in ports.items()}
    bench = (
        f"module tb; reg clk=0,rst_n=1; reg signed [{width - 1}:0] sample=0; {declarations} "
        f"native_overflow n({ports['n']}); legacy_overflow l({ports['l']}); "
        f"initial begin #1;rst_n=0;#1;rst_n=1;#1;sample={width}'h{sample:x};#1;clk=1;#1; "
        f'if (nv !== lv || ns !== ls) $fatal(1,"state parity mismatch"); '
        f'if (nv !== {width}\'h{expected & ((1 << width) - 1):x}) $fatal(1,"state mismatch"); '
        + ('if (ng !== lg) $fatal(1,"graded parity mismatch"); ' if event else "")
        + event_check
        + '$display("OBS committed");$finish;end endmodule'
    )
    top, tb, executable = tmp_path / "top.sv", tmp_path / "tb.sv", tmp_path / "sim.vvp"
    top.write_text(result.graph.emit_sv() + "\n" + legacy)
    tb.write_text(bench)
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(executable), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    observed = subprocess.run(["vvp", str(executable)], capture_output=True, text=True, timeout=60)
    if overflow == "trap" and not event:
        assert observed.returncode == 1
        assert "overflow" in observed.stdout.lower()
        assert "OBS committed" not in observed.stdout
    else:
        assert observed.returncode == 0, observed.stdout + observed.stderr
        assert "OBS committed" in observed.stdout
