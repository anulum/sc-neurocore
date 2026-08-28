# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained scaled-reset Q16.16 co-simulation

"""Execute committed retained-SC RTL against public and schema trajectories."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.neurons.models.sc_scaled_reset_adaptive_if import (
    SCScaledResetAdaptiveIFNeuron,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.toolchain_support import require_executable

ROOT = Path(__file__).parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_scaled_reset_adaptive_if.v"
SCALE = 1 << 16


def _literal(value: int) -> str:
    return f"-32'sd{-value}" if value < 0 else f"32'sd{value}"


def _rtl_trace(tmp_path: Path, currents: tuple[float, ...]) -> NDArray[np.float64]:
    drives = "\n".join(
        f'I_t={_literal(round(current * SCALE))};@(posedge clk);#1;$display("SCMN %0d %0d %0d %0d %0d",spike_out,v_out,theta_out,i1_out,i2_out);'
        for current in currents
    )
    bench = tmp_path / "tb_sc_scaled_reset.v"
    binary = tmp_path / "tb_sc_scaled_reset"
    bench.write_text(
        "`timescale 1ns/1ps\n"
        "module tb; reg clk=0; reg rst_n=0; reg signed[31:0] I_t=0; "
        "wire spike_out; wire signed[31:0] v_out,theta_out,i1_out,i2_out; "
        "always #5 clk=~clk; sc_scaled_reset_adaptive_if dut(.*); "
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
    rows = re.findall(r"^SCMN ([01]) (-?\d+) (-?\d+) (-?\d+) (-?\d+)$", output, re.M)
    assert len(rows) == len(currents)
    return np.asarray(
        [
            (int(event), int(v) / SCALE, int(theta) / SCALE, int(i1) / SCALE, int(i2) / SCALE)
            for event, v, theta, i1, i2 in rows
        ],
        dtype=np.float64,
    )


def _source_trace(currents: tuple[float, ...]) -> NDArray[np.float64]:
    source = SCScaledResetAdaptiveIFNeuron(
        theta_reset=1.3,
        tau_theta=40.0,
        tau_1=15.0,
        tau_2=80.0,
        a=0.1,
        b=0.1,
        r1=0.2,
        r2=-0.15,
    )
    toml = UniversalNeuron.from_schema("sc_scaled_reset_adaptive_if")
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


def test_q1616_complete_state_and_event_vector(tmp_path: Path) -> None:
    currents = (3.0,) * 250
    expected = _source_trace(currents)
    actual = _rtl_trace(tmp_path, currents)
    np.testing.assert_array_equal(actual[:, 0], expected[:, 0])
    assert int(actual[:, 0].sum()) == int(expected[:, 0].sum())
    assert int(actual[:, 0].sum()) == 31
    np.testing.assert_allclose(actual[:, 1:], expected[:, 1:], rtol=0.0, atol=0.001)


def test_yosys_synthesises_committed_rtl() -> None:
    completed = subprocess.run(
        [
            require_executable("yosys"),
            "-q",
            "-p",
            f"read_verilog {RTL}; synth -top sc_scaled_reset_adaptive_if -run begin:coarse; check; stat",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
