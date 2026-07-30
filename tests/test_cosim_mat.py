# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - source MAT* Q32.32 RTL co-simulation

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import numpy as np

from sc_neurocore.neurons.models.mat import MATNeuron

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_mat.v"
IVERILOG = ROOT / ".venv/bin/iverilog"
VVP = ROOT / ".venv/bin/vvp"
YOSYS = ROOT / ".venv/bin/yosys"
SCALE = 1 << 32
V_MIN = -200 * SCALE
V_MAX = 200 * SCALE
THETA_MAX = 2048 * SCALE
DT = 4_294_967
DT_OVER_TAU_M = 858_993
DECAY_1 = 4_294_537_821
DECAY_2 = 4_294_945_821
OMEGA = 19 * SCALE
ALPHA_1 = 37 * SCALE
ALPHA_2 = 2 * SCALE
REFRACTORY_PERIOD = 2 * SCALE


def _qmul(left: int, right: int) -> int:
    return (left * right) >> 32


def _bound(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def _oracle(currents: list[int]) -> np.ndarray:
    v = theta1 = theta2 = refractory = 0
    rows: list[tuple[int, int, int, int, int]] = []
    for current in currents:
        drive = _bound(-v + current * 50, V_MIN, V_MAX)
        v_candidate = _bound(v + _qmul(drive, DT_OVER_TAU_M), V_MIN, V_MAX)
        theta1_decay = _bound(_qmul(theta1, DECAY_1), 0, THETA_MAX)
        theta2_decay = _bound(_qmul(theta2, DECAY_2), 0, THETA_MAX)
        refractory_decay = 0 if refractory <= DT else refractory - DT
        event = int(refractory_decay == 0 and v_candidate >= OMEGA + theta1_decay + theta2_decay)
        v = v_candidate
        theta1 = _bound(theta1_decay + event * ALPHA_1, 0, THETA_MAX)
        theta2 = _bound(theta2_decay + event * ALPHA_2, 0, THETA_MAX)
        refractory = REFRACTORY_PERIOD if event else refractory_decay
        rows.append((v, theta1, theta2, refractory, event))
    return np.asarray(rows, dtype=np.int64)


def _rtl_trace(tmp_path: Path, currents: list[int], rtl: Path = RTL) -> np.ndarray:
    tmp_path.mkdir(parents=True, exist_ok=True)
    drives = "\n".join(
        f"current_t = 64'sd{current}; @(posedge clk); #1; "
        '$display("MAT_TRACE %0d %0d %0d %0d %0d", '
        "v, theta1, theta2, refractory, event_out);"
        for current in currents
    )
    bench = f"""
`timescale 1ns/1ps
module tb;
reg clk=0; reg rst_n=0; reg signed [63:0] current_t=0;
wire signed [63:0] v, theta1, theta2, refractory; wire event_out;
always #5 clk=~clk;
sc_mat uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.v_out(v),
.theta1_out(theta1),.theta2_out(theta2),.refractory_out(refractory),.event_out(event_out));
initial begin #23; rst_n=1; {drives} $finish; end
endmodule
"""
    tb = tmp_path / "tb.v"
    binary = tmp_path / "tb"
    tb.write_text(bench, encoding="utf-8")
    subprocess.run(
        [str(IVERILOG), "-g2012", "-o", str(binary), str(rtl), str(tb)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = subprocess.run(
        [str(VVP), str(binary)], check=True, capture_output=True, text=True
    ).stdout
    rows = re.findall(r"^MAT_TRACE (-?\d+) (-?\d+) (-?\d+) (-?\d+) ([01])$", output, re.M)
    assert len(rows) == len(currents)
    return np.asarray([[int(value) for value in row] for row in rows], dtype=np.int64)


def test_source_mat_q3232_matches_independent_integer_oracle(tmp_path: Path) -> None:
    currents_float = [0.0] * 32 + [4.0] * 5000 + [0.2, 0.7] * 256
    currents = [round(value * SCALE) for value in currents_float]
    expected = _oracle(currents)
    actual = _rtl_trace(tmp_path, currents)
    np.testing.assert_array_equal(actual, expected)
    hand = MATNeuron()
    events = [hand.step(current) for current in currents_float]
    np.testing.assert_array_equal(actual[:, 4], events)
    assert actual[:, 4].sum() > 1


def test_source_mat_yosys_synthesis_and_bounded_post_opt_equivalence(
    tmp_path: Path,
) -> None:
    synth = f"read_verilog -sv {RTL}; synth -top sc_mat; check"
    subprocess.run([str(YOSYS), "-q", "-p", synth], check=True, cwd=ROOT)
    netlist = tmp_path / "sc_mat_post_opt.v"
    script = (
        f"read_verilog -sv {RTL}; hierarchy -top sc_mat; "
        f"proc; memory; opt; check; write_verilog -noattr {netlist}"
    )
    subprocess.run([str(YOSYS), "-q", "-p", script], check=True, cwd=ROOT)
    currents_float = [0.0] * 32 + [4.0] * 5000 + [0.2, 0.7] * 256
    currents = [round(value * SCALE) for value in currents_float]
    original = _rtl_trace(tmp_path / "original", currents)
    optimized = _rtl_trace(tmp_path / "optimized", currents, netlist)
    np.testing.assert_array_equal(optimized, original)
