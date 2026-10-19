# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — graded spikes carry their value across the network compiler

"""A graded spike's value reaches its targets in the direct and the folded network.

A population whose template has ``spike_value`` emits a value on a spike. Its
connections add ``(value · weight) >>> fraction`` (a value of 1.0 is the binary
case), read from the direct path's registered ``spike_value_out`` or from the
folded path's ``val_bus``, which commits with ``spike_bus``. The direct path's
injected current is checked word for word against that arithmetic; the folded
raster must equal the direct one with and without a delay and under a source
threshold; and a graded network must differ from the same network with binary
spikes. Without Icarus the cosimulation cases skip.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from sc_neurocore.nir_bridge.fpga_folded_interconnect import can_fold
from sc_neurocore.nir_bridge.fpga_neuron_rtl import population_graded
from tests.folded_interconnect_support import (
    _DW,
    _FR,
    _HAVE_IVERILOG,
    Q88,
    ConnectionSpec,
    NeuronGraph,
    NeuronSpec,
    _build_neuron_module,
    _build_top_direct,
    _dequantised_pop,
    _parity_rasters,
    quantise_graph,
)

_SOURCE = {
    "equations": ["dv/dt = -v / tau + I"],
    "threshold": "v > 1.0",
    "reset": "v = 0.0",
    "default_params": {"tau": 4.0},
}
_GRADED = dict(_SOURCE, spike_value="0.75 * v")
_TARGET = {
    "equations": ["dv/dt = -v / tau + I"],
    "threshold": "v > 1.6",
    "reset": "v = 0.0",
    "default_params": {"tau": 6.0},
}


def _graph(
    source: dict[str, Any], *, delay: int = 0, source_threshold: float | None = None
) -> NeuronGraph:
    """Return stim → three source neurons → two target neurons, all-to-all at 0.6."""
    src = NeuronSpec(name="src", neuron_type="src_t", n_neurons=3, dt=0.25, template=source)
    dst = NeuronSpec(name="dst", neuron_type="dst_t", n_neurons=2, dt=0.25, template=_TARGET)
    conns = [
        ConnectionSpec(src="stim", dst="src", weights=np.array([[1.0], [0.8], [0.6]], np.float32)),
        ConnectionSpec(
            src="src",
            dst="dst",
            weights=np.full((2, 3), 0.6, dtype=np.float32),
            delay_steps=delay,
            source_threshold=None
            if source_threshold is None
            else np.full(3, source_threshold, dtype=np.float32),
        ),
    ]
    return NeuronGraph(
        populations=[src, dst], connections=conns, input_pop="stim", output_pop="dst", dt=0.25
    )


def _sim(sources: dict[str, str], steps: int) -> list[list[int]]:
    """Run Icarus on ``sources`` and return the integers each ``D`` line printed."""
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, text in sources.items():
            path = Path(d) / f"{name}.v"
            path.write_text(text)
            files.append(str(path))
        subprocess.run(
            ["iverilog", "-g2012", "-o", f"{d}/sim", *files],
            check=True,
            capture_output=True,
            timeout=180,
        )
        out = subprocess.run(
            ["vvp", f"{d}/sim"], capture_output=True, text=True, timeout=180
        ).stdout
    rows = [
        [int(x) for x in line.split()[1:]] for line in out.splitlines() if line.startswith("D ")
    ]
    assert len(rows) == steps
    return rows


def test_a_graded_template_is_graded_and_still_folds() -> None:
    """The template's spike_value marks the population graded; the graph folds."""
    graph = _graph(_GRADED)
    assert population_graded(graph.populations[0])
    assert not population_graded(graph.populations[1])
    assert can_fold(quantise_graph(graph, Q88(data_width=_DW, fraction=_FR)), data_width=_DW)


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
def test_the_direct_path_injects_value_times_weight() -> None:
    """Each step, a target's current is sat(sum((value_j * w) >>> fraction)) of the sources."""
    qgraph = quantise_graph(_graph(_GRADED), Q88(data_width=_DW, fraction=_FR))
    top = _build_top_direct("sc_graded", qgraph, data_width=_DW, fraction=_FR)
    modules = {
        f"mod_{pop.neuron_type}": _build_neuron_module(
            pop.neuron_type, _dequantised_pop(pop, _FR), data_width=_DW, fraction=_FR
        )
        for pop in qgraph.populations
    }
    steps = 40
    stim = Q88(data_width=_DW, fraction=_FR).encode(1.5)
    tb = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb;",
            "reg clk; reg rst_n; reg en; wire [4:0] spike_bus;",
            f"sc_graded uut (.clk(clk), .rst_n(rst_n), .en(en), .I_ext_flat({_DW}'d{stim}),",
            "    .spike_bus(spike_bus));",
            "initial clk = 0; always #5 clk = ~clk;",
            "integer t;",
            "initial begin",
            "  rst_n = 0; en = 1; #23; rst_n = 1;",
            f"  for (t = 0; t < {steps}; t = t + 1) begin @(posedge clk); #1;",
            '    $display("D %0d %0d %0d %0d %0d", $signed(uut.p0_n0_val),'
            " $signed(uut.p0_n1_val), $signed(uut.p0_n2_val), $signed(uut.p1_n0_I),"
            " $signed(uut.p1_n1_I));",
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )
    rows = _sim({**modules, "top": top, "tb": tb}, steps)
    weight = round(0.6 * (1 << _FR))
    limit = (1 << (_DW - 1)) - 1
    graded_steps = 0
    for values_and_currents in rows:
        values, currents = values_and_currents[:3], values_and_currents[3:]
        total = sum((value * weight) >> _FR for value in values)
        expected = max(-limit - 1, min(limit, total))
        assert currents == [expected, expected]
        graded_steps += any(value not in (0, 1 << _FR) for value in values)
    assert graded_steps > 0, "some spike carried a value other than 0 or 1"


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
@pytest.mark.parametrize(
    ("delay", "threshold"), [(0, None), (2, None), (0, 0.4)], ids=["plain", "delayed", "threshold"]
)
def test_folded_equals_direct_for_graded_spikes(delay: int, threshold: float | None) -> None:
    """The folded val_bus reproduces the direct registered values: identical rasters."""
    graph = _graph(_GRADED, delay=delay, source_threshold=threshold)
    direct, folded = _parity_rasters(graph, [1.5], 5)
    assert len(direct) == len(folded) > 0
    assert direct == folded
    assert any("1" in row[:2] for row in direct), "the targets spike"


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
def test_a_graded_network_differs_from_the_binary_one() -> None:
    """The control: the same graph with binary spikes gives another target raster."""
    graded, _ = _parity_rasters(_graph(_GRADED), [1.5], 5)
    binary, _ = _parity_rasters(_graph(_SOURCE), [1.5], 5)
    assert [row[:2] for row in graded] != [row[:2] for row in binary]


def test_graded_sources_keep_a_large_network_off_the_address_event_path() -> None:
    """The AER fan-out carries addresses only, so auto selection stays direct and says why."""
    src = NeuronSpec(name="src", neuron_type="src_t", n_neurons=70, dt=0.25, template=_GRADED)
    dst = NeuronSpec(name="dst", neuron_type="dst_t", n_neurons=2, dt=0.25, template=_TARGET)
    graph = NeuronGraph(
        populations=[src, dst],
        connections=[
            ConnectionSpec(src="stim", dst="src", weights=np.ones((70, 1), np.float32)),
            ConnectionSpec(src="src", dst="dst", weights=np.full((2, 70), 0.1, np.float32)),
        ],
        input_pop="stim",
        output_pop="dst",
        dt=0.25,
    )
    result = compile_network_to_fpga(graph, module_name="sc_graded_big")
    assert result.interconnect == "direct"
    assert any("graded spikes" in warning for warning in result.warnings)
    assert re.search(r"\.spike_value_out\(p0_n69_val\)", result.top_module)
