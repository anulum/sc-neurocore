# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — seeded EscapeRate Python-to-Verilog co-simulation

"""Bit-true LFSR events and statistical fidelity over one complete RTL period."""

from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
from tests.cosim_support import HAS_IVERILOG

_DATA_WIDTH = 48
_FRACTION = 24
_LFSR_PERIOD = 0xFFFF
_SEED = 0xACE1
_RHO_ZERO = 0.25
_VOLTAGE = -50.0
_VOLTAGE_Q = int(_VOLTAGE * (1 << _FRACTION))


def _constant_probability_schema() -> UniversalNeuron:
    """Return an exact LUT-grid operating point with p = 1 - exp(-0.25)."""
    schema = load_schema("escape_rate")
    schema["state"]["v"] = _VOLTAGE
    schema["parameters"].update(
        v_rest=_VOLTAGE,
        v_reset=_VOLTAGE,
        v_threshold=_VOLTAGE,
        rho_0=_RHO_ZERO,
        delta_u=1.0,
    )
    return UniversalNeuron.from_dict(schema, rng_seed_override=_SEED)


def _run_full_period_rtl(tmp_path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    """Compile and run the production registered RTL for all non-zero LFSR states."""
    model = _constant_probability_schema()
    module_name = "sc_escape_rate_seeded_cosim"
    rtl = model.to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    testbench = f"""`timescale 1ns/1ps
module tb_escape_rate;
reg clk = 1'b0;
reg rst_n = 1'b0;
reg signed [{_DATA_WIDTH - 1}:0] I_t = {_DATA_WIDTH}'sd0;
wire spike_out;
wire signed [{_DATA_WIDTH - 1}:0] v_out;
integer i;
integer spike_count = 0;

{module_name} dut (
    .clk(clk), .rst_n(rst_n), .I_t(I_t),
    .spike_out(spike_out), .v_out(v_out)
);

always #5 clk = ~clk;

initial begin
    repeat (2) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    $write("EVENTS ");
    for (i = 0; i < {_LFSR_PERIOD}; i = i + 1) begin
        @(posedge clk);
        #1;
        $write("%0d", spike_out);
        if (spike_out) spike_count = spike_count + 1;
        if (v_out !== -{_DATA_WIDTH}'sd{-_VOLTAGE_Q}) begin
            $fatal(1, "voltage drift at logical step %0d: %0d", i, v_out);
        end
    end
    $display("");
    $display(
        "DONE %0d %0d %0d %0d %0d",
        spike_count,
        dut._escape_lfsr,
        dut._escape_threshold,
        dut._escape_probability,
        v_out
    );
    $finish;
end
endmodule
"""
    rtl_path = tmp_path / "escape_rate.v"
    testbench_path = tmp_path / "tb_escape_rate.v"
    executable = tmp_path / "escape_rate_cosim"
    rtl_path.write_text(rtl, encoding="utf-8")
    testbench_path.write_text(testbench, encoding="utf-8")
    subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-o",
            str(executable),
            str(rtl_path),
            str(testbench_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    completed = subprocess.run(
        ["vvp", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    event_match = re.search(rf"EVENTS ([01]{{{_LFSR_PERIOD}}})", completed.stdout)
    done_match = re.search(r"DONE (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)", completed.stdout)
    if event_match is None or done_match is None:
        raise AssertionError(f"Could not parse EscapeRate RTL output:\n{completed.stdout}")
    events = np.fromiter((int(bit) for bit in event_match.group(1)), dtype=np.uint8)
    return events, tuple(int(value) for value in done_match.groups())


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
def test_seeded_full_period_event_stream_and_distribution_match_python(
    tmp_path: Path,
) -> None:
    """RTL and Python consume the same 65,535-state stream with geometric ISIs."""
    rtl_events, rtl_state = _run_full_period_rtl(tmp_path)
    python = EscapeRateNeuron(
        v=_VOLTAGE,
        v_rest=_VOLTAGE,
        v_reset=_VOLTAGE,
        v_threshold=_VOLTAGE,
        rho_0=_RHO_ZERO,
        delta_u=1.0,
        dt=1.0,
        seed=_SEED,
    )
    python_events = np.fromiter(
        (python.step(0.0) for _ in range(_LFSR_PERIOD)),
        dtype=np.uint8,
        count=_LFSR_PERIOD,
    )
    probability = -math.expm1(-_RHO_ZERO)
    expected_spikes = math.floor(probability * _LFSR_PERIOD)
    expected_probability_q = (1 << _FRACTION) - round(math.exp(-_RHO_ZERO) * (1 << _FRACTION))

    np.testing.assert_array_equal(rtl_events, python_events)
    spike_count, final_rng, threshold, probability_q, final_voltage_q = rtl_state
    assert spike_count == int(rtl_events.sum()) == expected_spikes == 14_496
    assert final_rng == python.rng_state == _SEED
    assert threshold == expected_spikes + 1 == 14_497
    assert probability_q == expected_probability_q
    assert final_voltage_q == _VOLTAGE_Q

    intervals = np.diff(np.flatnonzero(rtl_events))
    assert float(intervals.mean()) == pytest.approx(1.0 / probability, abs=1.0e-3)
    assert float(intervals.std() / intervals.mean()) == pytest.approx(
        math.sqrt(1.0 - probability), abs=0.01
    )
