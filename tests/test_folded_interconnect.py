# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded interconnect golden-parity tests

"""The folded (time-multiplexed) interconnect matches direct, spike-for-spike.

For a single connection-less population driven by per-neuron external current,
the direct interconnect instantiates one module per neuron; the folded one shares
a single datapath PE + BRAM state across all neurons, one per cycle. Both are
co-simulated from the same current and their spike rasters must be identical: the
direct module sampled every clock (step t), the folded sampled at each ``tick_done``
(also step t). The PE arithmetic is already bit-exact (test_folded_datapath), so the
folded sequencer carries that to the network output.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import quantise_graph
from sc_neurocore.nir_bridge.fpga_compiler import (
    _build_neuron_module,
    _build_top_direct,
    _build_top_folded,
)
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec

_HAVE_IVERILOG = shutil.which("iverilog") is not None and shutil.which("vvp") is not None
pytestmark = pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")

_DW, _FR = 16, 8
_N = 6
_STEPS = 60
_CURRENTS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # all above v_threshold=1.0 → varied spike rates


def _single_lif_graph() -> NeuronGraph:
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=_N, params={}, dt=1.0)
    return NeuronGraph(
        populations=[pop], connections=[], input_pop="pop0", output_pop="pop0", dt=1.0
    )


def _flat_current_literal() -> str:
    q = Q88(data_width=_DW, fraction=_FR)
    mask = (1 << _DW) - 1
    packed = 0
    for n, cur in enumerate(_CURRENTS):
        packed |= (q.encode(cur) & mask) << (n * _DW)
    return f"{_N * _DW}'h{packed:x}"


def _direct_tb(flat: str, n_dst: int = _N) -> str:
    return "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_direct;",
            "reg clk; reg rst_n; reg en;",
            f"wire [{n_dst - 1}:0] spike_bus;",
            "sc_fold_test uut (.clk(clk), .rst_n(rst_n), .en(en),",
            f"    .I_ext_flat({flat}), .spike_bus(spike_bus));",
            "initial clk = 0;",
            "always #5 clk = ~clk;",
            "integer t;",
            "initial begin",
            "    rst_n = 0; en = 1;",
            "    #23; rst_n = 1;  // deassert between clock edges (no reset-release race)",
            f"    for (t = 0; t < {_STEPS}; t = t + 1) begin",
            "        @(posedge clk); #1;",
            '        $display("R %b", spike_bus);',
            "    end",
            "    $finish;",
            "end",
            "endmodule",
        ]
    )


def _folded_tb(flat: str, n_dst: int = _N) -> str:
    return "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_folded;",
            "reg clk; reg rst_n; reg en;",
            f"wire [{n_dst - 1}:0] spike_bus; wire tick_done;",
            "sc_fold_test_folded uut (.clk(clk), .rst_n(rst_n), .en(en),",
            f"    .I_ext_flat({flat}), .spike_bus(spike_bus), .tick_done(tick_done));",
            "initial clk = 0;",
            "always #5 clk = ~clk;",
            "integer ticks;",
            "initial begin",
            "    rst_n = 0; en = 1; ticks = 0;",
            "    #23; rst_n = 1;  // deassert between clock edges (no reset-release race)",
            f"    while (ticks < {_STEPS}) begin",
            "        @(posedge clk); #1;",
            "        if (tick_done) begin",
            '            $display("R %b", spike_bus);',
            "            ticks = ticks + 1;",
            "        end",
            "    end",
            "    $finish;",
            "end",
            "endmodule",
        ]
    )


def _cosim(sources: dict[str, str], tb_name: str) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        files = []
        for name, src in sources.items():
            p = dp / f"{name}.v"
            p.write_text(src)
            files.append(str(p))
        out = dp / "sim"
        comp = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out), *files],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert comp.returncode == 0, f"iverilog ({tb_name}) failed: {comp.stderr}"
        run = subprocess.run(["vvp", str(out)], capture_output=True, text=True, timeout=180)
        return re.findall(r"R ([01]+)", run.stdout)


def test_folded_matches_direct_spike_raster() -> None:
    ng = _single_lif_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    flat = _flat_current_literal()

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_source, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )

    direct_raster = _cosim({"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat)}, "direct")
    folded_raster = _cosim({"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat)}, "folded")

    assert len(direct_raster) == _STEPS, f"direct produced {len(direct_raster)} rows"
    assert len(folded_raster) == _STEPS, f"folded produced {len(folded_raster)} rows"
    assert folded_raster == direct_raster, (
        "folded spike raster diverged from direct:\n"
        f"  first mismatch at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    # Non-vacuous: the workload spikes.
    assert any("1" in row for row in direct_raster), "test workload should produce spikes"


def _weighted_ff_graph() -> tuple[NeuronGraph, list[float], int]:
    """Single LIF population fed by one external-source weighted connection.

    Returns ``(graph, external_currents, n_dst)``. Weights vary per row so the
    per-neuron weighted input spans sub-threshold to spiking.
    """
    import numpy as np

    n_dst, n_src = 5, 3
    rows = [[0.5, 0.3, 0.2], [0.6, 0.3, 0.2], [0.7, 0.3, 0.2], [0.4, 0.4, 0.3], [0.3, 0.2, 0.1]]
    weights = np.array(rows, dtype=np.float32)
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    conn = ConnectionSpec(src="stim", dst="pop0", weights=weights)
    ng = NeuronGraph(
        populations=[pop], connections=[conn], input_pop="stim", output_pop="pop0", dt=1.0
    )
    return ng, [2.0, 1.5, 1.0], n_dst


def test_folded_weighted_external_matches_direct() -> None:
    ng, currents, n_dst = _weighted_ff_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_source, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_dst)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_dst)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded weighted-input raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "weighted workload should spike"


def test_folded_state_is_bram_backed_and_shares_one_pe() -> None:
    ng = _single_lif_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    _pe, folded_top = _build_top_folded("sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR)
    # Shared datapath: exactly one PE instance, BRAM state, a sequencer — no per-neuron unroll.
    assert folded_top.count("pe_inst") == 1
    assert 'ram_style = "block"' in folded_top
    assert "state_bram" in folded_top
    assert "p0_n0_inst" not in folded_top  # the direct per-neuron instance name must be absent


def test_compile_network_folded_opt_in() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    result = compile_network_to_fpga(_single_lif_graph(), interconnect="folded")
    assert result.interconnect == "folded"
    assert "lif_pe" in result.neuron_modules  # the shared datapath PE was emitted
    assert "state_bram" in result.top_module


def test_compile_network_folded_rejects_unsupported_graph() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    # Two populations are outside the v1 folded subset.
    pop_a = NeuronSpec(name="a", neuron_type="lif", n_neurons=4, params={}, dt=1.0)
    pop_b = NeuronSpec(name="b", neuron_type="lif", n_neurons=4, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[pop_a, pop_b], connections=[], input_pop="a", output_pop="b", dt=1.0
    )
    with pytest.raises(ValueError, match="folded"):
        compile_network_to_fpga(ng, interconnect="folded")
