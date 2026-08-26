# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - SC resetting-MAT Q32.32 RTL co-simulation

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import numpy as np

from sc_neurocore.neurons.models.sc_resetting_mat import SCResettingMATNeuron
from tests.toolchain_support import require_executable

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_resetting_mat.v"
IVERILOG = require_executable("iverilog")
VVP = require_executable("vvp")
YOSYS = require_executable("yosys")
SCALE = 1 << 32
V_REST = -70 * SCALE
V_RESET = -70 * SCALE
V_THRESHOLD_BASE = -50 * SCALE
V_MIN = -200 * SCALE
V_MAX = 100 * SCALE
THETA_MAX = 2048 * SCALE
RK4_FAST = 3_886_247_471
RK4_SLOW = 4_273_546_057
H1 = 5 * SCALE
H2 = 3 * SCALE


def _qmul(left: int, right: int) -> int:
    return (left * right) >> 32


def _bound(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def _oracle(currents: list[int]) -> np.ndarray:
    v, theta1, theta2 = V_REST, 0, 0
    rows: list[tuple[int, int, int, int]] = []
    for current in currents:
        equilibrium = _bound(V_REST + current, V_MIN, V_MAX)
        v_candidate = _bound(equilibrium + _qmul(v - equilibrium, RK4_FAST), V_MIN, V_MAX)
        theta1_candidate = _bound(_qmul(theta1, RK4_FAST), 0, THETA_MAX)
        theta2_candidate = _bound(_qmul(theta2, RK4_SLOW), 0, THETA_MAX)
        event = int(v_candidate >= V_THRESHOLD_BASE + theta1_candidate + theta2_candidate)
        v = V_RESET if event else v_candidate
        theta1 = _bound(theta1_candidate + event * H1, 0, THETA_MAX)
        theta2 = _bound(theta2_candidate + event * H2, 0, THETA_MAX)
        rows.append((v, theta1, theta2, event))
    return np.asarray(rows, dtype=np.int64)


def _rtl_trace(tmp_path: Path, currents: list[int], rtl: Path = RTL) -> np.ndarray:
    tmp_path.mkdir(parents=True, exist_ok=True)
    drives = "\n".join(
        f"current_t = 64'sd{current}; @(posedge clk); #1; "
        '$display("SCMAT_TRACE %0d %0d %0d %0d", v, theta1, theta2, event_out);'
        for current in currents
    )
    bench = f"""
`timescale 1ns/1ps
module tb;
reg clk=0; reg rst_n=0; reg signed [63:0] current_t=0;
wire signed [63:0] v, theta1, theta2; wire event_out;
always #5 clk=~clk;
sc_resetting_mat uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.v_out(v),
.theta1_out(theta1),.theta2_out(theta2),.event_out(event_out));
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
    rows = re.findall(r"^SCMAT_TRACE (-?\d+) (-?\d+) (-?\d+) ([01])$", output, re.M)
    assert len(rows) == len(currents)
    return np.asarray([[int(value) for value in row] for row in rows], dtype=np.int64)


def test_sc_resetting_mat_q3232_matches_independent_integer_oracle(tmp_path: Path) -> None:
    currents_float = [0.0] * 32 + [50.0] * 96 + [20.0, 60.0] * 64
    currents = [round(value * SCALE) for value in currents_float]
    expected = _oracle(currents)
    actual = _rtl_trace(tmp_path, currents)
    np.testing.assert_array_equal(actual, expected)
    hand = SCResettingMATNeuron()
    events = [hand.step(current) for current in currents_float]
    np.testing.assert_array_equal(actual[:, 3], events)
    assert actual[:, 3].sum() == 13


def test_sc_resetting_mat_yosys_synthesis_and_bounded_post_opt_equivalence(
    tmp_path: Path,
) -> None:
    synth = f"read_verilog -sv {RTL}; synth -top sc_resetting_mat; check"
    subprocess.run([str(YOSYS), "-q", "-p", synth], check=True, cwd=ROOT)
    netlist = tmp_path / "sc_resetting_mat_post_opt.v"
    script = (
        f"read_verilog -sv {RTL}; hierarchy -top sc_resetting_mat; "
        f"proc; memory; opt; check; write_verilog -noattr {netlist}"
    )
    subprocess.run([str(YOSYS), "-q", "-p", script], check=True, cwd=ROOT)
    currents_float = [0.0] * 32 + [50.0] * 96 + [20.0, 60.0] * 64
    currents = [round(value * SCALE) for value in currents_float]
    original = _rtl_trace(tmp_path / "original", currents)
    optimized = _rtl_trace(tmp_path / "optimized", currents, netlist)
    np.testing.assert_array_equal(optimized, original)
