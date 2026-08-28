# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source Mihalas-Niebur Q32.32 co-simulation

"""Execute committed Model 14 RTL against source and schema trajectories."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.toolchain_support import require_executable

ROOT = Path(__file__).parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_mihalasnieburneuron.v"
SCALE = 1 << 32


def _literal(value: int) -> str:
    return f"-64'sd{-value}" if value < 0 else f"64'sd{value}"


def _rtl_trace(tmp_path: Path, currents: tuple[float, ...]) -> NDArray[np.float64]:
    drives = "\n".join(
        f'I_t={_literal(round(current * SCALE))};@(posedge clk);#1;$display("MN %0d %0d %0d %0d %0d",spike_out,v_out,theta_out,i1_out,i2_out);'
        for current in currents
    )
    bench = tmp_path / "tb_mihalas_niebur.v"
    binary = tmp_path / "tb_mihalas_niebur"
    bench.write_text(
        "`timescale 1ns/1ps\n"
        "module tb; reg clk=0; reg rst_n=0; reg signed[63:0] I_t=0; "
        "wire spike_out; wire signed[63:0] v_out,theta_out,i1_out,i2_out; "
        "always #5 clk=~clk; "
        "sc_mihalasnieburneuron #("
        f".P_CURRENT_JUMP_1({_literal(round(0.01 * SCALE))}),"
        f".P_CURRENT_JUMP_2({_literal(round(-0.0006 * SCALE))})"
        ") dut(.*); "
        f"initial begin #23; rst_n=1; {drives} $finish; end endmodule\n",
        encoding="utf-8",
    )
    subprocess.run(
        [require_executable("iverilog"), "-g2012", "-o", str(binary), str(RTL), str(bench)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = subprocess.run(
        [require_executable("vvp"), str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    rows = re.findall(r"^MN ([01]) (-?\d+) (-?\d+) (-?\d+) (-?\d+)$", output, re.M)
    assert len(rows) == len(currents)
    return np.asarray(
        [
            (int(event), int(v) / SCALE, int(theta) / SCALE, int(i1) / SCALE, int(i2) / SCALE)
            for event, v, theta, i1, i2 in rows
        ],
        dtype=np.float64,
    )


def _source_trace(currents: tuple[float, ...]) -> NDArray[np.float64]:
    source = MihalasNieburNeuron(current_jump_1=0.01, current_jump_2=-0.0006)
    toml = UniversalNeuron.from_schema(
        "mihalas_niebur",
        parameter_overrides={"current_jump_1": 0.01, "current_jump_2": -0.0006},
    )
    rows: list[tuple[int, float, float, float, float]] = []
    for current in currents:
        event = source.step(current)
        assert int(bool(toml.step(I=current))) == event
        assert toml.state == {
            "v": source.v,
            "theta": source.theta,
            "i1": source.i1,
            "i2": source.i2,
        }
        rows.append((event, source.v, source.theta, source.i1, source.i2))
    return np.asarray(rows, dtype=np.float64)


def test_q3232_complete_state_and_event_vector(tmp_path: Path) -> None:
    currents = (0.002,) * 2000
    expected = _source_trace(currents)
    actual = _rtl_trace(tmp_path, currents)
    np.testing.assert_array_equal(actual[:, 0], expected[:, 0])
    assert int(actual[:, 0].sum()) == 14
    np.testing.assert_allclose(actual[:, 1:], expected[:, 1:], rtol=0.0, atol=1.3e-6)


def test_yosys_synthesises_committed_rtl() -> None:
    completed = subprocess.run(
        [
            require_executable("yosys"),
            "-q",
            "-p",
            f"read_verilog {RTL}; synth -top sc_mihalasnieburneuron -run begin:coarse; check; stat",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
