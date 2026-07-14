# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — seeded Poisson Python-to-Verilog co-simulation

"""Bit-true registered/folded LFSR events over one complete RTL period."""

from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess

import numpy as np
from numpy.typing import NDArray
import pytest

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath
from sc_neurocore.neurons.models.poisson import PoissonNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

_DATA_WIDTH = 48
_FRACTION = 24
_LFSR_PERIOD = 0xFFFF
_SEED = 0xACE1
_RATE_HZ = 250.0
_DT_MS = 1.0
_RATE_OVERRIDE_MAGNITUDE_Q = 1 << _FRACTION


def _constant_probability_schema() -> UniversalNeuron:
    """Return the stateless exact-law schema at a 0.25 interval hazard."""
    return UniversalNeuron.from_schema(
        "poisson",
        parameter_overrides={"rate_hz": _RATE_HZ, "dt_ms": _DT_MS},
        rng_seed_override=_SEED,
    )


def _compile_and_run(
    tmp_path: Path,
    *,
    stem: str,
    rtl: str,
    testbench: str,
) -> tuple[NDArray[np.uint8], tuple[int, ...]]:
    """Compile one production RTL form and parse its full event stream."""
    rtl_path = tmp_path / f"{stem}.v"
    testbench_path = tmp_path / f"tb_{stem}.v"
    executable = tmp_path / f"{stem}_cosim"
    rtl_path.write_text(rtl, encoding="utf-8")
    testbench_path.write_text(testbench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(executable), str(rtl_path), str(testbench_path)],
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
    done_match = re.search(r"DONE (-?\d+) (-?\d+) (-?\d+) (-?\d+)", completed.stdout)
    if event_match is None or done_match is None:
        raise AssertionError(f"Could not parse Poisson RTL output:\n{completed.stdout}")
    events = np.fromiter((int(bit) for bit in event_match.group(1)), dtype=np.uint8)
    return events, tuple(int(value) for value in done_match.groups())


def _run_registered(tmp_path: Path) -> tuple[NDArray[np.uint8], tuple[int, ...]]:
    """Run the production state-owning module for one complete LFSR period."""
    module_name = "sc_poisson_seeded_cosim"
    rtl = _constant_probability_schema().to_verilog(
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    testbench = f"""`timescale 1ns/1ps
module tb_poisson_registered;
reg clk = 1'b0;
reg rst_n = 1'b0;
reg signed [{_DATA_WIDTH - 1}:0] I_t = -{_DATA_WIDTH}'sd{_RATE_OVERRIDE_MAGNITUDE_Q};
wire spike_out;
integer i;
integer spike_count = 0;

{module_name} dut (
    .clk(clk), .rst_n(rst_n), .I_t(I_t), .spike_out(spike_out)
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
    end
    $display("");
    $display(
        "DONE %0d %0d %0d %0d",
        spike_count, dut._escape_lfsr, dut._escape_threshold, dut._escape_probability
    );
    $finish;
end
endmodule
"""
    return _compile_and_run(tmp_path, stem="poisson_registered", rtl=rtl, testbench=testbench)


def _run_folded(tmp_path: Path) -> tuple[NDArray[np.uint8], tuple[int, ...]]:
    """Run the production folded combinational PE with an external LFSR sample."""
    module_name = "sc_poisson_folded_cosim"
    model = _constant_probability_schema().to_equation_neuron()
    rtl = compile_to_datapath(
        model,
        module_name=module_name,
        data_width=_DATA_WIDTH,
        fraction=_FRACTION,
    )
    testbench = f"""`timescale 1ns/1ps
module tb_poisson_folded;
reg signed [{_DATA_WIDTH - 1}:0] I_t = -{_DATA_WIDTH}'sd{_RATE_OVERRIDE_MAGNITUDE_Q};
reg [15:0] rng_sample = 16'h{_SEED:04x};
reg [15:0] rng_state = 16'h{_SEED:04x};
wire spike_out;
integer i;
integer advance;
integer spike_count = 0;

function [15:0] lfsr_advance;
    input [15:0] value;
    begin
        lfsr_advance = {{value[0] ^ value[2] ^ value[3] ^ value[5], value[15:1]}};
    end
endfunction

{module_name} dut (.I_t(I_t), .rng_sample(rng_sample), .spike_out(spike_out));

initial begin
    $write("EVENTS ");
    for (i = 0; i < {_LFSR_PERIOD}; i = i + 1) begin
        for (advance = 0; advance < 8; advance = advance + 1)
            rng_state = lfsr_advance(rng_state);
        rng_sample = rng_state;
        #1;
        $write("%0d", spike_out);
        if (spike_out) spike_count = spike_count + 1;
    end
    $display("");
    $display(
        "DONE %0d %0d %0d %0d",
        spike_count, rng_state, dut._escape_threshold, dut._escape_probability
    );
    $finish;
end
endmodule
"""
    return _compile_and_run(tmp_path, stem="poisson_folded", rtl=rtl, testbench=testbench)


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
def test_seeded_full_period_registered_and_folded_streams_match_python(
    tmp_path: Path,
) -> None:
    """Both hardware forms consume the same 65,535-state Bernoulli stream."""
    registered_events, registered_state = _run_registered(tmp_path)
    folded_events, folded_state = _run_folded(tmp_path)
    python = PoissonNeuron(rate_hz=_RATE_HZ, dt_ms=_DT_MS, seed=_SEED)
    python_events = np.fromiter(
        (python.step() for _ in range(_LFSR_PERIOD)),
        dtype=np.uint8,
        count=_LFSR_PERIOD,
    )

    np.testing.assert_array_equal(registered_events, python_events)
    np.testing.assert_array_equal(folded_events, python_events)
    probability = -math.expm1(-_RATE_HZ * _DT_MS / 1000.0)
    expected_spikes = math.floor(probability * _LFSR_PERIOD)
    expected_probability_q = (1 << _FRACTION) - round(
        math.exp(-_RATE_HZ * _DT_MS / 1000.0) * (1 << _FRACTION)
    )
    for spike_count, final_rng, threshold, probability_q in (
        registered_state,
        folded_state,
    ):
        assert spike_count == int(python_events.sum()) == expected_spikes == 14_496
        assert final_rng == python.rng_state == _SEED
        assert threshold == expected_spikes + 1 == 14_497
        assert probability_q == expected_probability_q

    intervals = np.diff(np.flatnonzero(registered_events))
    assert float(intervals.mean()) == pytest.approx(1.0 / probability, abs=1.0e-3)
    assert float(intervals.std() / intervals.mean()) == pytest.approx(
        math.sqrt(1.0 - probability), abs=0.01
    )
