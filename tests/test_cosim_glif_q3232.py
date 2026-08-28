# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLIF5 committed Q32.32 co-simulation

"""Execute the committed GLIF5 RTL against integer and source oracles."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.neurons.models.glif import GLIFNeuron


_ROOT = Path(__file__).parents[1]
_RTL = _ROOT / "hdl/formal/catalogue/sc_glif.v"
_SCALE = 1 << 32
_E_L = -300_647_710_720
_THETA_INF = -214_748_364_800
_MEMBRANE_DECAY = 3_886_247_119
_SPIKE_DECAY = 4_252_231_657
_ASC1_DECAY = 3_886_247_119
_ASC2_DECAY = 4_273_546_057
_STEADY_FORCING = 4_273_563_864
_VOLTAGE_CONVOLUTION = 4_066_494_874
_A_VOLTAGE = 429_497


def _qmul(left: int, right: int) -> int:
    return (left * right) >> 32


def _integer_oracle(currents: tuple[float, ...]) -> list[tuple[int, ...]]:
    v = _E_L
    theta_spike = i_asc1 = i_asc2 = theta_voltage = refractory = 0
    states: list[tuple[int, ...]] = []
    for current in currents:
        event = 0
        if refractory:
            refractory -= 1
        else:
            drive = round(current * _SCALE) + i_asc1 + i_asc2
            voltage_offset = v - _E_L
            next_v = _E_L + drive + _qmul(voltage_offset - drive, _MEMBRANE_DECAY)
            next_theta_spike = _qmul(theta_spike, _SPIKE_DECAY)
            next_i_asc1 = _qmul(i_asc1, _ASC1_DECAY)
            next_i_asc2 = _qmul(i_asc2, _ASC2_DECAY)
            forcing = _qmul(drive, _STEADY_FORCING) + _qmul(
                voltage_offset - drive, _VOLTAGE_CONVOLUTION
            )
            next_theta_voltage = _qmul(theta_voltage, _SPIKE_DECAY) + _qmul(_A_VOLTAGE, forcing)
            if next_v > _THETA_INF + next_theta_spike + next_theta_voltage:
                v = _E_L
                theta_spike = next_theta_spike + 2 * _SCALE
                i_asc1 = next_i_asc1 + _SCALE
                i_asc2 = next_i_asc2 + _SCALE // 2
                theta_voltage = next_theta_voltage
                refractory = 2
                event = 1
            else:
                v = next_v
                theta_spike = next_theta_spike
                i_asc1 = next_i_asc1
                i_asc2 = next_i_asc2
                theta_voltage = next_theta_voltage
        states.append((event, v, theta_spike, i_asc1, i_asc2, theta_voltage, refractory))
    return states


def _rtl_trace(currents: tuple[float, ...]) -> list[tuple[int, ...]]:
    testbench = f"""
module tb;
reg clk = 0;
reg rst_n = 0;
reg signed [63:0] I_t = 0;
reg signed [63:0] drive [0:{len(currents) - 1}];
integer index;
wire spike_out;
wire signed [63:0] v_out, theta_spike_out, i_asc1_out, i_asc2_out, theta_voltage_out;
wire [1:0] refractory_out;
sc_glif dut(.*);
always #5 clk = ~clk;
initial begin
  $readmemh("drive.hex", drive);
  @(posedge clk); #1; rst_n = 1;
  for (index = 0; index < {len(currents)}; index = index + 1) begin
    I_t = drive[index];
    @(posedge clk); #1;
    $display("TRACE %0d %0d %0d %0d %0d %0d %0d", spike_out, v_out,
      theta_spike_out, i_asc1_out, i_asc2_out, theta_voltage_out, refractory_out);
  end
  $finish;
end
endmodule
"""
    with tempfile.TemporaryDirectory(prefix="glif5_q3232_") as directory:
        root = Path(directory)
        bench = root / "tb.v"
        drive_path = root / "drive.hex"
        executable = root / "tb.out"
        bench.write_text(testbench, encoding="utf-8")
        drive_path.write_text(
            "\n".join(f"{round(current * _SCALE) & ((1 << 64) - 1):016x}" for current in currents),
            encoding="ascii",
        )
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(executable), str(_RTL), str(bench)],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            ["vvp", str(executable)],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
    return [
        tuple(map(int, line.split()[1:]))
        for line in result.stdout.splitlines()
        if line.startswith("TRACE ")
    ]


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="Icarus Verilog unavailable")
@pytest.mark.parametrize(
    ("current", "expected_events"),
    ((0.0, 0), (15.0, 0), (22.0, 22), (30.0, 49), (45.0, 74), (50.0, 80)),
)
def test_default_profile_event_counts(current: float, expected_events: int) -> None:
    currents = (current,) * 1000
    rtl = _rtl_trace(currents)
    assert rtl == _integer_oracle(currents)
    assert sum(row[0] for row in rtl) == expected_events


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="Icarus Verilog unavailable")
def test_mixed_drive_complete_state_and_event_vector() -> None:
    currents = (30.0, 0.0, 45.0, -5.0, 20.0, 35.0, 0.0, 50.0) * 64
    rtl = _rtl_trace(currents)
    assert rtl == _integer_oracle(currents)

    source = GLIFNeuron()
    source_states = []
    for current in currents:
        event = source.step(current)
        source_states.append(
            (
                event,
                source.v,
                source.theta_spike,
                source.i_asc1,
                source.i_asc2,
                source.theta_voltage,
                source.refractory_remaining,
            )
        )
    assert [row[0] for row in rtl] == [row[0] for row in source_states]
    np.testing.assert_allclose(
        np.asarray([row[1:6] for row in rtl], dtype=np.float64) / _SCALE,
        np.asarray([row[1:6] for row in source_states], dtype=np.float64),
        rtol=0.0,
        atol=2e-7,
    )
    assert [row[6] for row in rtl] == [int(row[6]) for row in source_states]
