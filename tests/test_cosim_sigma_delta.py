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

from tests.toolchain_support import require_executable

ROOT = Path(__file__).parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_sigma_delta.v"
SCALE = 1 << 32
DT = 429496730
DECAY = 4252231657
DELTA = SCALE
HALF = SCALE // 2
LIMIT = 1_000_000 * SCALE


def qmul(a: int, b: int) -> int:
    return a * b >> 32


def bound(v: int) -> int:
    return min(LIMIT, max(-LIMIT, v))


def literal(v: int) -> str:
    return f"-64'sd{-v}" if v < 0 else f"64'sd{v}"


def oracle(currents: list[int]) -> np.ndarray:
    sigma = reconstruction = 0
    rows = []
    for current in currents:
        sigma = bound(sigma + qmul(current, DT))
        rd = bound(qmul(reconstruction, DECAY))
        event = int(sigma - rd >= HALF)
        reconstruction = bound(rd + event * DELTA)
        rows.append((sigma, reconstruction, event))
    return np.asarray(rows, dtype=np.int64)


def rtl_trace(tmp: Path, currents: list[int]) -> np.ndarray:
    drives = "\n".join(
        f'current_t={literal(x)};@(posedge clk);#1;$display("SD %0d %0d %0d",sigma,reconstruction,event_out);'
        for x in currents
    )
    tb = tmp / "tb.v"
    binary = tmp / "tb"
    tmp.mkdir(parents=True, exist_ok=True)
    tb.write_text(
        f"`timescale 1ns/1ps\nmodule tb;reg clk=0;reg rst_n=0;reg signed[63:0]current_t=0;wire signed[63:0]sigma,reconstruction;wire event_out;always #5 clk=~clk;sc_sigma_delta uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.sigma_out(sigma),.reconstruction_out(reconstruction),.event_out(event_out));initial begin #23;rst_n=1;{drives}$finish;end endmodule\n"
    )
    subprocess.run(
        [require_executable("iverilog"), "-g2012", "-o", str(binary), str(RTL), str(tb)],
        check=True,
    )
    out = subprocess.run(
        [require_executable("vvp"), str(binary)], check=True, capture_output=True, text=True
    ).stdout
    rows = re.findall(r"^SD (-?\d+) (-?\d+) ([01])$", out, re.M)
    return np.asarray([[int(v) for v in row] for row in rows], dtype=np.int64)


def test_sampled_apsdm_q3232_matches_integer_oracle(tmp_path: Path) -> None:
    currents = [round(v * SCALE) for v in ([0.0] * 16 + [2.0, -1.0, 4.0] * 64)]
    np.testing.assert_array_equal(rtl_trace(tmp_path, currents), oracle(currents))
