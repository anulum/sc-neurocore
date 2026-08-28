# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from pathlib import Path
import re
import subprocess

import numpy as np

from sc_neurocore.neurons.models.mckean import McKeanNeuron
from tests.toolchain_support import require_executable

ROOT = Path(__file__).parents[1]
RTL = ROOT / "hdl/formal/catalogue/mckean.v"
SCALE = 1 << 32


def _literal(value: int) -> str:
    return f"-64'sd{-value}" if value < 0 else f"64'sd{value}"


def _rtl_trace(tmp: Path, currents: list[float]) -> np.ndarray:
    drives = "\n".join(
        f'current_t={_literal(round(x * SCALE))};@(posedge clk);#1;$display("MK %0d %0d %0d",v,w,event_out);'
        for x in currents
    )
    tb = tmp / "tb.v"
    binary = tmp / "tb"
    tmp.mkdir(parents=True, exist_ok=True)
    tb.write_text(
        f"`timescale 1ns/1ps\nmodule tb;reg clk=0;reg rst_n=0;reg signed[63:0]current_t=0;wire signed[63:0]v,w;wire event_out;always #5 clk=~clk;mckean uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.v_out(v),.w_out(w),.event_out(event_out));initial begin #23;rst_n=1;{drives}$finish;end endmodule\n"
    )
    subprocess.run(
        [require_executable("iverilog"), "-g2012", "-o", str(binary), str(RTL), str(tb)],
        check=True,
    )
    output = subprocess.run(
        [require_executable("vvp"), str(binary)], check=True, capture_output=True, text=True
    ).stdout
    return np.asarray(
        [
            [int(v) / SCALE, int(w) / SCALE, int(event)]
            for v, w, event in re.findall(r"^MK (-?\d+) (-?\d+) ([01])$", output, re.M)
        ]
    )


def test_source_mckean_q3232_tracks_python(tmp_path: Path) -> None:
    currents = [0.0, 3.0, 0.0, -0.2] * 32
    neuron = McKeanNeuron()
    expected = []
    for current in currents:
        event = neuron.step(current)
        expected.append((neuron.v, neuron.w, event))
    actual = _rtl_trace(tmp_path, currents)
    np.testing.assert_array_equal(actual[:, 2], np.asarray(expected)[:, 2])
    np.testing.assert_allclose(actual[:, :2], np.asarray(expected)[:, :2], atol=2e-6, rtol=0)


def test_yosys_synthesises_committed_rtl() -> None:
    completed = subprocess.run(
        [
            require_executable("yosys"),
            "-q",
            "-p",
            f"read_verilog {RTL}; synth -top mckean -run begin:coarse; check; stat",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
