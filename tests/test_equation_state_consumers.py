# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Held-state testbench, command and adaptive compiler consumers

"""Run the public command and wrapper consumers with real held-state hardware."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from sc_neurocore.compiler.adaptive_runtime_precision import compile_adaptive_precision
from sc_neurocore.compiler.equation_compiler import generate_testbench
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron


def test_generated_testbench_observes_held_state_from_the_same_authored_model(
    tmp_path: Path,
) -> None:
    """The supplied testbench elaborates and reports all declared state outputs."""
    model = EquationNeuron({"v": "held + I"}, state={"held": 0.5}, method="map")
    top, tb, binary = tmp_path / "model.v", tmp_path / "tb.v", tmp_path / "run"
    top.write_text(compile_to_verilog(model, "held_model"))
    tb.write_text(generate_testbench(model, "held_model", n_steps=4, input_current=0.25))
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_held_model", "-o", str(binary), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    result = subprocess.run(
        ["vvp", str(binary)], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60
    )
    assert "Final held = 128 " in result.stdout
    assert "Final v = 192 " in result.stdout
    assert model.get_state() == {"held": 0.5, "v": 0.0}


def test_adaptive_wrapper_exports_the_authoritative_held_state_and_event_reset(
    tmp_path: Path,
) -> None:
    """The HP-authoritative wrapper retains its one-register delay for every state word."""
    model = EquationNeuron(
        {"v": "held + I"},
        state={"held": 0.5},
        method="map",
        dt=1.0,
        threshold="v > 1",
        reset={"held": "held + 0.25", "v": "-0.25"},
    )
    source = compile_adaptive_precision(
        model, "held_adaptive", lp_width=8, lp_frac=4, hp_width=16, hp_frac=8
    )
    top, tb, binary = tmp_path / "adaptive.v", tmp_path / "tb.v", tmp_path / "run"
    top.write_text(source)
    steps = []
    previous = model.get_state()
    previous_spike = 0
    for index, current in enumerate([0.75, 0.25, 0.75, 0.25, 1.0, 0.5]):
        reset = "rst_n=0;#1;rst_n=1;#1;" if index == 3 else ""
        if index == 3:
            model.reset()
            previous, previous_spike = model.get_state(), 0
        steps.append(
            f"{reset}I_t=16'h{round(current * 256) & 65535:x};#2;clk=1;#1;if(v !== 16'h{round(previous['v'] * 256) & 65535:x} || held !== 16'h{round(previous['held'] * 256) & 65535:x} || spike !== 1'b{previous_spike}) $fatal(1,\"adaptive held-state mismatch\");clk=0;#1;"
        )
        previous_spike = model.step(current)
        previous = model.get_state()
    tb.write_text(
        "module tb;reg clk=0,rst_n=1;reg [15:0] I_t=0;wire [15:0] v,held;wire spike,use_hp;held_adaptive dut(.clk(clk),.rst_n(rst_n),.I_t(I_t),.v_out(v),.held_out(held),.spike_out(spike),.use_hp(use_hp));initial begin #1;rst_n=0;#1;rst_n=1;"
        + "\n".join(steps)
        + '$display("OBS adaptive held-state");$finish;end endmodule'
    )
    subprocess.run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(binary), str(top), str(tb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    result = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=60
    )
    assert "OBS adaptive held-state" in result.stdout


def test_public_compile_command_emits_a_runnable_partial_state_model_and_testbench(
    tmp_path: Path,
) -> None:
    """The real command carries an auxiliary initial value into its generated hardware."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sc_neurocore.cli",
            "compile",
            "dv/dt = held + I",
            "--method",
            "map",
            "--init",
            "held=0.5",
            "--module-name",
            "held_command",
            "--testbench",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    binary = tmp_path / "run"
    subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_held_command",
            "-o",
            str(binary),
            str(tmp_path / "held_command.v"),
            str(tmp_path / "tb_held_command.v"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    result = subprocess.run(
        ["vvp", str(binary)], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60
    )
    assert "Final held = 128 " in result.stdout
    assert "Final v = 384 " in result.stdout
