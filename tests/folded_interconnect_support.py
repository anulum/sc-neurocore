# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_folded_interconnect.py

from __future__ import annotations


"""The folded (time-multiplexed) interconnect matches direct, spike-for-spike.

For a single connection-less population driven by per-neuron external current,
the direct interconnect instantiates one module per neuron; the folded one shares
a single datapath PE + BRAM state across all neurons, one per cycle. Both are
co-simulated from the same current and their spike rasters must be identical: the
direct module sampled every clock (step t), the folded sampled at each ``tick_done``
(also step t). The PE arithmetic is already bit-exact (test_folded_datapath), so the
folded sequencer carries that to the network output.
"""


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
    _dequantised_pop,
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


def _recurrent_graph() -> tuple[NeuronGraph, list[float], int]:
    """Single LIF population: external drive + a recurrent (self) spiking connection.

    The recurrent term reads prior-tick spikes; folding it exercises the spike_bus
    double-buffer. Returns ``(graph, external_currents, n_dst)``.
    """
    import numpy as np

    n_dst, n_src = 5, 3
    ext_rows = [[0.5, 0.3, 0.2], [0.6, 0.3, 0.2], [0.7, 0.3, 0.2], [0.4, 0.4, 0.3], [0.3, 0.2, 0.1]]
    ext_w = np.array(ext_rows, dtype=np.float32)
    # Ring excitation: neuron i feeds neuron (i+1) mod N on a spike.
    rec_w = np.zeros((n_dst, n_dst), dtype=np.float32)
    for i in range(n_dst):
        rec_w[(i + 1) % n_dst, i] = 0.4
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    ext_conn = ConnectionSpec(src="stim", dst="pop0", weights=ext_w)
    rec_conn = ConnectionSpec(src="pop0", dst="pop0", weights=rec_w)
    ng = NeuronGraph(
        populations=[pop],
        connections=[ext_conn, rec_conn],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    return ng, [2.0, 1.5, 1.0], n_dst


def _two_pop_ff_graph() -> tuple[NeuronGraph, list[float], int]:
    """Two-population feedforward net: external-weighted input pop → spiking output pop.

    pop ``inp`` is fed by a weighted external projection; pop ``out`` is driven only
    by ``inp``'s spikes (inter-population fan-in). Both share one LIF PE under the
    fold. Returns ``(graph, external_currents, total_neurons)``.
    """
    import numpy as np

    n_in, n_out, n_src = 4, 3, 2
    ext = np.array([[1.4, 1.0], [1.6, 0.8], [1.2, 1.2], [1.8, 0.6]], dtype=np.float32)
    # Each output neuron pools several input spikes with strong weight so the inter-pop
    # fan-in drives the downstream LIF above threshold.
    ff = np.array(
        [[2.0, 0.0, 1.8, 1.2], [1.2, 2.0, 0.0, 1.8], [1.6, 1.6, 1.4, 0.0]], dtype=np.float32
    )
    inp = NeuronSpec(name="inp", neuron_type="lif", n_neurons=n_in, params={}, dt=1.0)
    out = NeuronSpec(name="out", neuron_type="lif", n_neurons=n_out, params={}, dt=1.0)
    ext_conn = ConnectionSpec(src="stim", dst="inp", weights=ext)
    ff_conn = ConnectionSpec(src="inp", dst="out", weights=ff)
    ng = NeuronGraph(
        populations=[inp, out],
        connections=[ext_conn, ff_conn],
        input_pop="stim",
        output_pop="out",
        dt=1.0,
    )
    return ng, [4.0, 3.5], n_in + n_out


def _two_pop_recurrent_graph() -> tuple[NeuronGraph, list[float], int]:
    """Two LIF populations with external drive, inter-pop fan-in, and self-recurrence.

    Exercises every folded fan-in source at once: external-weighted (pop ``a``), an
    ``a → b`` inter-population spiking projection, and a ``b → b`` recurrent ring.
    """
    import numpy as np

    n_a, n_b = 3, 4
    ext = np.array([[0.8, 0.4], [0.6, 0.5], [0.7, 0.3]], dtype=np.float32)
    a_to_b = np.array(
        [[0.6, 0.0, 0.6], [0.0, 0.6, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.6]], dtype=np.float32
    )
    rec_b = np.zeros((n_b, n_b), dtype=np.float32)
    for i in range(n_b):
        rec_b[(i + 1) % n_b, i] = 0.3
    pop_a = NeuronSpec(name="a", neuron_type="lif", n_neurons=n_a, params={}, dt=1.0)
    pop_b = NeuronSpec(name="b", neuron_type="lif", n_neurons=n_b, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[pop_a, pop_b],
        connections=[
            ConnectionSpec(src="stim", dst="a", weights=ext),
            ConnectionSpec(src="a", dst="b", weights=a_to_b),
            ConnectionSpec(src="b", dst="b", weights=rec_b),
        ],
        input_pop="stim",
        output_pop="b",
        dt=1.0,
    )
    return ng, [3.5, 3.0], n_a + n_b


def _delayed_recurrent_graph() -> tuple[NeuronGraph, list[float], int]:
    """Single LIF population with external drive and a DELAYED recurrent ring.

    The recurrent self-connection carries a two-tick synaptic delay, so folding it
    exercises the spike_bus history shift-register. Returns ``(graph, currents, n)``.
    """
    import numpy as np

    n_dst, n_src = 5, 3
    ext_rows = [[0.5, 0.3, 0.2], [0.6, 0.3, 0.2], [0.7, 0.3, 0.2], [0.4, 0.4, 0.3], [0.3, 0.2, 0.1]]
    ext_w = np.array(ext_rows, dtype=np.float32)
    rec_w = np.zeros((n_dst, n_dst), dtype=np.float32)
    for i in range(n_dst):
        rec_w[(i + 1) % n_dst, i] = 0.4
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    ext_conn = ConnectionSpec(src="stim", dst="pop0", weights=ext_w)
    rec_conn = ConnectionSpec(src="pop0", dst="pop0", weights=rec_w, delay_steps=2)
    ng = NeuronGraph(
        populations=[pop],
        connections=[ext_conn, rec_conn],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    return ng, [2.0, 1.5, 1.0], n_dst


def _two_pop_delayed_ff_graph() -> tuple[NeuronGraph, list[float], int]:
    """Two-population feedforward net with a delayed inter-population projection.

    pop ``inp`` is external-weighted; the ``inp → out`` projection carries per-column
    synaptic delays (mixed 0/1/2 ticks), exercising the folded history register on an
    inter-population edge. Returns ``(graph, currents, total_neurons)``.
    """
    import numpy as np

    n_in, n_out = 4, 3
    ext = np.array([[1.4, 1.0], [1.6, 0.8], [1.2, 1.2], [1.8, 0.6]], dtype=np.float32)
    ff = np.array(
        [[2.0, 0.0, 1.8, 1.2], [1.2, 2.0, 0.0, 1.8], [1.6, 1.6, 1.4, 0.0]], dtype=np.float32
    )
    inp = NeuronSpec(name="inp", neuron_type="lif", n_neurons=n_in, params={}, dt=1.0)
    out = NeuronSpec(name="out", neuron_type="lif", n_neurons=n_out, params={}, dt=1.0)
    ext_conn = ConnectionSpec(src="stim", dst="inp", weights=ext)
    # Per-source-column delays: input neuron 0 undelayed, 1 by one tick, 2 by two, 3 by one.
    ff_conn = ConnectionSpec(
        src="inp", dst="out", weights=ff, delay_steps=np.array([0, 1, 2, 1], dtype=np.int64)
    )
    ng = NeuronGraph(
        populations=[inp, out],
        connections=[ext_conn, ff_conn],
        input_pop="stim",
        output_pop="out",
        dt=1.0,
    )
    return ng, [4.0, 3.5], n_in + n_out


def _parity_rasters(ng: NeuronGraph, currents: list[float], n_total: int) -> tuple[list, list]:
    """Co-simulate ``ng`` through both interconnects and return ``(direct, folded)`` rasters."""
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"
    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    # One per-instance module per distinct neuron type — a multi-type graph (e.g. an
    # analogue li source feeding a lif population) instantiates each type's module.
    type_pop: dict[str, NeuronSpec] = {}
    for pop in qgraph.populations:
        type_pop.setdefault(pop.neuron_type, pop)
    direct_modules = {
        f"mod_{ntype}": _build_neuron_module(
            ntype, _dequantised_pop(pop, _FR), data_width=_DW, fraction=_FR
        )
        for ntype, pop in type_pop.items()
    }
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())
    direct_raster = _cosim(
        {**direct_modules, "top": direct_top, "tb": _direct_tb(flat, n_total)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_total)}, "folded"
    )
    return direct_raster, folded_raster


__all__ = ['re', 'shutil', 'subprocess', 'tempfile', 'Path', 'pytest', 'Q88', 'quantise_graph', '_build_neuron_module', '_build_top_direct', '_build_top_folded', '_dequantised_pop', 'ConnectionSpec', 'NeuronGraph', 'NeuronSpec', '_HAVE_IVERILOG', 'pytestmark', '_DW', '_FR', '_N', '_STEPS', '_CURRENTS', '_single_lif_graph', '_flat_current_literal', '_direct_tb', '_folded_tb', '_cosim', '_weighted_ff_graph', '_recurrent_graph', '_two_pop_ff_graph', '_two_pop_recurrent_graph', '_delayed_recurrent_graph', '_two_pop_delayed_ff_graph', '_parity_rasters']

