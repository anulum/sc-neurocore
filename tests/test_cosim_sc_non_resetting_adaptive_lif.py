# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - project non-resetting adaptive LIF Q32.32 co-simulation

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import numpy as np

from sc_neurocore.neurons.models.sc_non_resetting_adaptive_lif import (
    SCNonResettingAdaptiveLIFNeuron,
)

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_non_resetting_adaptive_lif.v"
IVERILOG = ROOT / ".venv/bin/iverilog"
VVP = ROOT / ".venv/bin/vvp"
YOSYS = ROOT / ".venv/bin/yosys"
SCALE = 1 << 32
V_REST = -65 * SCALE
THETA_REST = -50 * SCALE
DELTA_THETA = 5 * SCALE
V_DECAY = 4_252_231_657
THETA_DECAY = 4_286_385_946
V_MIN = -200 * SCALE
V_MAX = 200 * SCALE
THETA_MIN = -200 * SCALE
THETA_MAX = 2048 * SCALE


def _qmul(left: int, right: int) -> int:
    return (left * right) >> 32


def _bound(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def _oracle(currents: list[int]) -> np.ndarray:
    v, theta = V_REST, THETA_REST
    rows: list[tuple[int, int, int]] = []
    for current in currents:
        equilibrium = _bound(V_REST + current, V_MIN, V_MAX)
        v_candidate = _bound(equilibrium + _qmul(v - equilibrium, V_DECAY), V_MIN, V_MAX)
        theta_decay = _bound(
            THETA_REST + _qmul(theta - THETA_REST, THETA_DECAY), THETA_MIN, THETA_MAX
        )
        event = int(v_candidate >= theta_decay)
        v = v_candidate
        theta = _bound(theta_decay + event * DELTA_THETA, THETA_MIN, THETA_MAX)
        rows.append((v, theta, event))
    return np.asarray(rows, dtype=np.int64)


def _rtl_trace(tmp_path: Path, currents: list[int], rtl: Path = RTL) -> np.ndarray:
    tmp_path.mkdir(parents=True, exist_ok=True)
    drives = "\n".join(
        f"current_t = 64'sd{current}; @(posedge clk); #1; "
        '$display("SCNRLIF_TRACE %0d %0d %0d", v, theta, event_out);'
        for current in currents
    )
    bench = f"""
`timescale 1ns/1ps
module tb;
reg clk=0; reg rst_n=0; reg signed [63:0] current_t=0;
wire signed [63:0] v, theta; wire event_out;
always #5 clk=~clk;
sc_non_resetting_adaptive_lif uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),
.v_out(v),.theta_out(theta),.event_out(event_out));
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
    rows = re.findall(r"^SCNRLIF_TRACE (-?\d+) (-?\d+) ([01])$", output, re.M)
    assert len(rows) == len(currents)
    return np.asarray([[int(value) for value in row] for row in rows], dtype=np.int64)


def test_sc_project_q3232_matches_integer_oracle_and_events(tmp_path: Path) -> None:
    currents_float = [0.0] * 32 + [20.0] * 96 + [20.0, 60.0] * 64
    currents = [round(value * SCALE) for value in currents_float]
    actual = _rtl_trace(tmp_path, currents)
    np.testing.assert_array_equal(actual, _oracle(currents))
    hand = SCNonResettingAdaptiveLIFNeuron()
    events = [hand.step(current) for current in currents_float]
    np.testing.assert_array_equal(actual[:, 2], events)
    assert actual[:, 2].sum() == 5


def test_sc_project_yosys_and_post_opt_sequence_equivalence(tmp_path: Path) -> None:
    subprocess.run(
        [
            str(YOSYS),
            "-q",
            "-p",
            f"read_verilog -sv {RTL}; synth -top sc_non_resetting_adaptive_lif; check",
        ],
        check=True,
        cwd=ROOT,
    )
    netlist = tmp_path / "sc_non_resetting_adaptive_lif_post_opt.v"
    script = (
        f"read_verilog -sv {RTL}; hierarchy -top sc_non_resetting_adaptive_lif; "
        f"proc; memory; opt; check; write_verilog -noattr {netlist}"
    )
    subprocess.run([str(YOSYS), "-q", "-p", script], check=True, cwd=ROOT)
    currents = [round(value * SCALE) for value in ([0.0] * 32 + [20.0] * 96 + [60.0] * 32)]
    np.testing.assert_array_equal(
        _rtl_trace(tmp_path / "optimized", currents, netlist),
        _rtl_trace(tmp_path / "original", currents),
    )
