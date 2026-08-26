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
RTL = ROOT / "hdl/formal/catalogue/sc_sigma_delta_accumulator.v"
SCALE = 1 << 32
LIMIT = 1_000_000 * SCALE


def literal(v: int) -> str:
    return f"-64'sd{-v}" if v < 0 else f"64'sd{v}"


def oracle(currents: list[int]) -> np.ndarray:
    sigma = 0
    rows = []
    for current in currents:
        sigma = min(LIMIT, max(-LIMIT, sigma + current))
        event = 0
        if sigma >= SCALE:
            sigma -= SCALE
            event = 1
        elif sigma <= -SCALE:
            sigma += SCALE
            event = -1
        rows.append((sigma, event))
    return np.asarray(rows, dtype=np.int64)


def rtl_trace(tmp: Path, currents: list[int]) -> np.ndarray:
    drives = "\n".join(
        f'current_t={literal(x)};@(posedge clk);#1;$display("SCSD %0d %0d",sigma,event_out);'
        for x in currents
    )
    tb = tmp / "tb.v"
    binary = tmp / "tb"
    tmp.mkdir(parents=True, exist_ok=True)
    tb.write_text(
        f"`timescale 1ns/1ps\nmodule tb;reg clk=0;reg rst_n=0;reg signed[63:0]current_t=0;wire signed[63:0]sigma;wire signed[1:0]event_out;always #5 clk=~clk;sc_sigma_delta_accumulator uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.sigma_out(sigma),.event_out(event_out));initial begin #23;rst_n=1;{drives}$finish;end endmodule\n"
    )
    subprocess.run(
        [require_executable("iverilog"), "-g2012", "-o", str(binary), str(RTL), str(tb)],
        check=True,
    )
    out = subprocess.run(
        [require_executable("vvp"), str(binary)], check=True, capture_output=True, text=True
    ).stdout
    rows = re.findall(r"^SCSD (-?\d+) (-?\d+)$", out, re.M)
    return np.asarray([[int(v) for v in row] for row in rows], dtype=np.int64)


def test_sc_accumulator_q3232_matches_integer_oracle(tmp_path: Path) -> None:
    currents = [round(v * SCALE) for v in ([0.0] * 16 + [3.25, -4.5, 0.2] * 64)]
    np.testing.assert_array_equal(rtl_trace(tmp_path, currents), oracle(currents))
