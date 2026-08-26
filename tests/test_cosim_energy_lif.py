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
from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron
from tests.toolchain_support import require_executable

ROOT = Path(__file__).parents[1]
RTL = ROOT / "hdl/formal/catalogue/energy_lif.v"
SCALE = 1 << 32


def literal(v: int) -> str:
    return f"-64'sd{-v}" if v < 0 else f"64'sd{v}"


def rtl_trace(tmp: Path, currents: list[float]) -> np.ndarray:
    drives = "\n".join(
        f'current_t={literal(round(x * SCALE))};@(posedge clk);#1;$display("EL %0d %0d %0d",v,epsilon,event_out);'
        for x in currents
    )
    tb = tmp / "tb.v"
    binary = tmp / "tb"
    tmp.mkdir(parents=True, exist_ok=True)
    tb.write_text(
        f"`timescale 1ns/1ps\nmodule tb;reg clk=0;reg rst_n=0;reg signed[63:0]current_t=0;wire signed[63:0]v,epsilon;wire event_out;always #5 clk=~clk;energy_lif uut(.clk(clk),.rst_n(rst_n),.current_t(current_t),.v_out(v),.epsilon_out(epsilon),.event_out(event_out));initial begin #23;rst_n=1;{drives}$finish;end endmodule\n"
    )
    subprocess.run(
        [require_executable("iverilog"), "-g2012", "-o", str(binary), str(RTL), str(tb)],
        check=True,
    )
    out = subprocess.run(
        [require_executable("vvp"), str(binary)], check=True, capture_output=True, text=True
    ).stdout
    return np.asarray(
        [
            [int(x) / SCALE, int(y) / SCALE, int(e)]
            for x, y, e in re.findall(r"^EL (-?\d+) (-?\d+) ([01])$", out, re.M)
        ]
    )


def test_fardet_levina_q3232_tracks_python(tmp_path: Path) -> None:
    currents = [80.0, 0.0, 120.0, 20.0] * 32
    n = EnergyLIFNeuron()
    expected = []
    for current in currents:
        expected.append((n.v_reset if (event := n.step(current)) else n.v, n.epsilon, event))
    actual = rtl_trace(tmp_path, currents)
    np.testing.assert_array_equal(actual[:, 2], np.asarray(expected)[:, 2])
    np.testing.assert_allclose(actual[:, :2], np.asarray(expected)[:, :2], atol=2e-6, rtol=0)
