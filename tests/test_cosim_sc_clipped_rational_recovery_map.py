# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained rational-recovery Python-to-Verilog co-simulation

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.neurons.models.sc_clipped_rational_recovery_map import (
    SCClippedRationalRecoveryMapNeuron,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import HAS_IVERILOG

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
TraceRow = tuple[int, float, float]


def _rtl_trace(*, steps: int, data_width: int, fraction: int) -> list[TraceRow]:
    assert HAS_IVERILOG
    neuron = UniversalNeuron.from_schema("sc_clipped_rational_recovery_map")
    module = f"sc_rational_recovery_q{data_width - fraction}_{fraction}_trace"
    verilog = neuron.to_verilog(
        module_name=module,
        data_width=data_width,
        fraction=fraction,
    )
    zero = Q88(data_width=data_width, fraction=fraction).encode(0.0)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            f"wire signed [{data_width - 1}:0] x_out;",
            f"wire signed [{data_width - 1}:0] y_out;",
            "always #5 clk = ~clk;",
            f"{module} uut (",
            f"  .clk(clk), .rst_n(rst_n), .I_t({data_width}'sd{zero}),",
            "  .spike_out(spike_out), .x_out(x_out), .y_out(y_out));",
            "integer index;",
            "initial begin",
            "  #23; rst_n = 1'b1;",
            f"  for (index=0; index<{steps}; index=index+1) begin",
            "    @(posedge clk); #1;",
            '    $display("SC_RR %0d %0d %0d", spike_out, uut.x_reg, uut.y_reg);',
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rtl = root / "model.v"
        tb = root / "tb.v"
        executable = root / "sim"
        rtl.write_text(verilog, encoding="utf-8")
        tb.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(executable), str(rtl), str(tb)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = subprocess.run(
            ["vvp", str(executable)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    rows = re.findall(r"^SC_RR (-?\d+) (-?\d+) (-?\d+)$", output, re.MULTILINE)
    scale = float(1 << fraction)
    assert len(rows) == steps
    return [(int(event), int(x) / scale, int(y) / scale) for event, x, y in rows]


def _hand_schema_trace(steps: int) -> list[TraceRow]:
    hand = SCClippedRationalRecoveryMapNeuron()
    toml = UniversalNeuron.from_schema(SCHEMA_DIR / "sc_clipped_rational_recovery_map.toml")
    json_model = UniversalNeuron.from_schema(SCHEMA_DIR / "sc_clipped_rational_recovery_map.json")
    trace: list[TraceRow] = []
    for _step in range(steps):
        event = hand.step(0.0)
        assert int(bool(toml.step(I=0.0))) == event
        assert int(bool(json_model.step(I=0.0))) == event
        assert toml.state == {"x": hand.x, "y": hand.y}
        assert json_model.state == {"x": hand.x, "y": hand.y}
        trace.append((event, hand.x, hand.y))
    return trace


def test_paired_schemas_preserve_the_complete_retained_receipt() -> None:
    trace = _hand_schema_trace(512)
    assert sum(row[0] for row in trace) == 0
    assert trace[-1][1:] == (-1_000_000.0, 304058.694830129)


@pytest.mark.parametrize(
    ("data_width", "fraction", "x_tolerance", "y_tolerance"),
    ((32, 16, 0.0009, 0.00046), (64, 32, 0.00000005, 0.00000003)),
    ids=("q16.16", "q32.32"),
)
def test_bounded_rtl_trajectory(
    data_width: int,
    fraction: int,
    x_tolerance: float,
    y_tolerance: float,
) -> None:
    expected = _hand_schema_trace(128)
    observed = _rtl_trace(steps=128, data_width=data_width, fraction=fraction)
    assert [row[0] for row in observed] == [row[0] for row in expected]
    assert sum(row[0] for row in observed) == 0
    assert (
        max(
            abs(reference[1] - candidate[1])
            for reference, candidate in zip(expected, observed, strict=True)
        )
        < x_tolerance
    )
    assert (
        max(
            abs(reference[2] - candidate[2])
            for reference, candidate in zip(expected, observed, strict=True)
        )
        < y_tolerance
    )
