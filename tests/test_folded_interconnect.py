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
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

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
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

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


def test_folded_recurrent_matches_direct() -> None:
    ng, currents, n_dst = _recurrent_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_dst)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_dst)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded recurrent raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "recurrent workload should spike"


def test_folded_state_is_bram_backed_and_shares_one_pe() -> None:
    ng = _single_lif_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    # Shared datapath: exactly one PE instance, BRAM state, a sequencer — no per-neuron unroll.
    assert folded_top.count("pe_inst") == 1
    assert 'ram_style = "block"' in folded_top
    assert "state_bram" in folded_top
    assert "p0_n0_inst" not in folded_top  # the direct per-neuron instance name must be absent
    assert set(pe_modules) == {"lif_pe"}  # one PE module per distinct neuron type


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


def test_folded_two_population_feedforward_matches_direct() -> None:
    ng, currents, n_total = _two_pop_ff_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_total)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_total)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded two-population feedforward raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    # Both layers must be exercised. spike_bus is [6:0] printed MSB-first: the output
    # pop occupies bits [4:6] (the leading 3 chars), the input pop bits [0:3] (trailing 4).
    assert any("1" in row[3:] for row in direct_raster), "input population should spike"
    assert any("1" in row[:3] for row in direct_raster), "output population should spike"


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


def test_folded_two_population_recurrent_matches_direct() -> None:
    ng, currents, n_total = _two_pop_recurrent_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_total)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_total)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded two-population recurrent raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "recurrent workload should spike"


def test_folded_multi_population_shares_one_pe_per_type() -> None:
    ng, _currents, _n = _two_pop_ff_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    # Two LIF populations collapse to one shared PE instance and a per-population BRAM.
    assert set(pe_modules) == {"lif_pe"}
    assert folded_top.count("pe_inst_lif") == 1
    assert "state_bram_0" in folded_top and "state_bram_1" in folded_top
    assert "p0_n0_inst" not in folded_top  # no direct per-neuron unrolling


def test_compile_network_folded_two_populations() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    ng, _currents, n_total = _two_pop_ff_graph()
    result = compile_network_to_fpga(ng, interconnect="folded")
    assert result.interconnect == "folded"
    assert "lif_pe" in result.neuron_modules
    assert result.folded_metrics is not None
    assert result.folded_metrics.populations == 2
    assert result.folded_metrics.neurons == n_total
    assert result.folded_metrics.pe_instances == 1  # one PE shared across both LIF pops


def test_compile_network_folded_opt_in() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    result = compile_network_to_fpga(_single_lif_graph(), interconnect="folded")
    assert result.interconnect == "folded"
    assert "lif_pe" in result.neuron_modules  # the shared datapath PE was emitted
    assert "state_bram" in result.top_module


def test_compile_network_folded_analogue_source() -> None:
    import numpy as np

    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    # An analogue li source feeding a lif population folds, emitting the global voltage
    # bus; the source's three columns are counted as shared multipliers (v * weight).
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=3, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=3, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[li, lif],
        connections=[ConnectionSpec(src="a", dst="b", weights=np.full((3, 3), 0.5, np.float32))],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    result = compile_network_to_fpga(ng, interconnect="folded")
    assert result.interconnect == "folded"
    assert "v_bus" in result.top_module  # the analogue voltage double-buffer was emitted
    assert {"li_pe", "lif_pe"} <= set(result.neuron_modules)
    assert result.folded_metrics is not None
    assert result.folded_metrics.populations == 2
    assert result.folded_metrics.pe_instances == 2  # distinct types: li + lif
    assert result.folded_metrics.shared_multipliers == 3  # the analogue source's 3 columns


def test_compile_network_folded_rejects_unsupported_graph() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    # A *delayed* analogue source connection is the only remaining fan-in shape outside
    # the folded subset (it would need a voltage-bus history register) → direct fallback.
    # An undelayed analogue source folds, so the delay is what triggers the rejection.
    import numpy as np

    pop_a = NeuronSpec(name="a", neuron_type="li", n_neurons=4, params={}, dt=1.0)
    pop_b = NeuronSpec(name="b", neuron_type="lif", n_neurons=4, params={}, dt=1.0)
    delayed_analogue = ConnectionSpec(
        src="a",
        dst="b",
        weights=np.ones((4, 4), np.float32) * 0.4,
        delay_steps=2,
    )
    ng = NeuronGraph(
        populations=[pop_a, pop_b],
        connections=[delayed_analogue],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    with pytest.raises(ValueError, match="folded"):
        compile_network_to_fpga(ng, interconnect="folded")


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


def test_folded_delayed_recurrent_matches_direct() -> None:
    ng, currents, n_dst = _delayed_recurrent_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())
    # The two-tick delay materialises a depth-2 spike-bus history shift-register.
    assert "spike_bus_hist_2" in folded_top
    assert "spike_bus_hist_3" not in folded_top

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_dst)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_dst)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded delayed-recurrent raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "delayed recurrent workload should spike"


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


def test_folded_two_population_delayed_feedforward_matches_direct() -> None:
    ng, currents, n_total = _two_pop_delayed_ff_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_total)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_total)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded delayed two-population raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row[:3] for row in direct_raster), "output population should spike"


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
        f"mod_{ntype}": _build_neuron_module(ntype, pop, data_width=_DW, fraction=_FR)
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


def test_folded_external_source_threshold_matches_direct() -> None:
    import numpy as np

    # External-weighted input gated per-column by a NIR source Threshold: a column
    # contributes its (un-multiplied) weight only when its external input exceeds the
    # threshold. Currents straddle the thresholds so the gates toggle.
    n_dst, n_src = 5, 3
    weights = np.array(
        [[1.2, 0.8, 0.6], [1.0, 0.9, 0.7], [1.3, 0.5, 0.4], [0.9, 1.0, 0.6], [1.1, 0.7, 0.5]],
        dtype=np.float32,
    )
    src_thr = np.array([1.0, 1.2, 0.8], dtype=np.float32)
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    conn = ConnectionSpec(src="stim", dst="pop0", weights=weights, source_threshold=src_thr)
    ng = NeuronGraph(
        populations=[pop], connections=[conn], input_pop="stim", output_pop="pop0", dt=1.0
    )
    direct_raster, folded_raster = _parity_rasters(ng, [1.5, 1.1, 1.3], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded external source-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "source-threshold workload should spike"


def test_folded_destination_threshold_matches_direct() -> None:
    import numpy as np

    # Mixed fan-in into one population: a destination-thresholded external connection
    # (the whole weighted sum, gated per neuron, emits one spike-magnitude) plus a plain
    # baseline external connection. The baseline alone stays sub-threshold; only neurons
    # whose thresholded connection also fires reach the LIF threshold, so the per-neuron
    # destination threshold toggles which neurons spike.
    n_dst = 4
    thr_w = np.full((n_dst, 3), 0.8, dtype=np.float32)  # raw ≈ (2.0+1.5+1.0)*0.8 = 3.6
    dst_thr = np.array([1.0, 3.0, 4.0, 5.0], dtype=np.float32)  # neurons 0,1 fire; 2,3 do not
    base_w = np.full((n_dst, 2), 0.3, dtype=np.float32)  # baseline ≈ 0.6, sub-threshold alone
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[pop],
        connections=[
            ConnectionSpec(src="stim", dst="pop0", weights=thr_w, destination_threshold=dst_thr),
            ConnectionSpec(src="base", dst="pop0", weights=base_w),
        ],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    # External bus layout follows connection order: stim (3 lanes) then base (2 lanes).
    direct_raster, folded_raster = _parity_rasters(ng, [2.0, 1.5, 1.0, 1.0, 1.0], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded destination-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "destination-threshold workload should spike"


def test_folded_two_population_source_threshold_matches_direct() -> None:
    import numpy as np

    # Inter-population spiking fan-in gated by a source Threshold: the spike magnitude
    # (1.0 in Q8.8) is compared against the per-column threshold before the weight gates.
    n_in, n_out = 4, 3
    ext = np.array([[1.4, 1.0], [1.6, 0.8], [1.2, 1.2], [1.8, 0.6]], dtype=np.float32)
    ff = np.array(
        [[2.0, 0.0, 1.8, 1.2], [1.2, 2.0, 0.0, 1.8], [1.6, 1.6, 1.4, 0.0]], dtype=np.float32
    )
    # Thresholds below the 1.0 spike magnitude so a spike passes the gate (column 1 above
    # it, so that column never contributes — exercising both sides of the comparator).
    src_thr = np.array([0.5, 1.5, 0.5, 0.5], dtype=np.float32)
    inp = NeuronSpec(name="inp", neuron_type="lif", n_neurons=n_in, params={}, dt=1.0)
    out = NeuronSpec(name="out", neuron_type="lif", n_neurons=n_out, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[inp, out],
        connections=[
            ConnectionSpec(src="stim", dst="inp", weights=ext),
            ConnectionSpec(src="inp", dst="out", weights=ff, source_threshold=src_thr),
        ],
        input_pop="stim",
        output_pop="out",
        dt=1.0,
    )
    direct_raster, folded_raster = _parity_rasters(ng, [4.0, 3.5], n_in + n_out)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded inter-pop source-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row[:3] for row in direct_raster), "output population should spike"


def test_folded_external_bias_matches_direct() -> None:
    import numpy as np

    # External-weighted input plus a per-destination-neuron bias constant. The bias is
    # added in ACC_WIDTH to the connection's term sum, so the steady-state LIF current
    # (≈ weighted input + bias) crosses the threshold for some neurons and not others —
    # the per-neuron bias ROM toggles which spike (positive, near-zero, and negative
    # biases all exercised).
    n_dst, n_src = 4, 2
    weights = np.full((n_dst, n_src), 0.2, dtype=np.float32)  # weighted ≈ 0.4 per neuron
    bias = np.array([1.0, 0.7, 0.3, -0.5], dtype=np.float32)  # I ≈ [1.4, 1.1, 0.7, -0.1]
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    conn = ConnectionSpec(src="stim", dst="pop0", weights=weights, bias=bias)
    ng = NeuronGraph(
        populations=[pop], connections=[conn], input_pop="stim", output_pop="pop0", dt=1.0
    )
    direct_raster, folded_raster = _parity_rasters(ng, [1.0, 1.0], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded bias raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "biased workload should spike"


def test_folded_bias_with_destination_threshold_matches_direct() -> None:
    import numpy as np

    # A destination-thresholded connection whose per-neuron bias participates in the
    # ``raw`` accumulator (raw = weighted sum + bias, compared against the per-neuron
    # destination threshold), mixed with a plain baseline connection. Equal weights
    # leave the bias as the discriminator: with raw ≈ 1.75 + bias and a threshold of
    # 2.0, neurons whose bias lifts raw above 2.0 emit a spike-magnitude that — together
    # with the sub-threshold baseline — pushes the LIF over, while the others stay quiet.
    n_dst = 4
    thr_w = np.full((n_dst, 2), 0.5, dtype=np.float32)  # weighted ≈ (2.0+1.5)*0.5 = 1.75
    bias = np.array([0.5, 0.0, -1.0, 0.3], dtype=np.float32)  # raw ≈ [2.25,1.75,0.75,2.05]
    dst_thr = np.full(n_dst, 2.0, dtype=np.float32)  # neurons 0,3 fire; 1,2 do not
    base_w = np.full((n_dst, 2), 0.3, dtype=np.float32)  # baseline ≈ 0.6, sub-threshold alone
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[pop],
        connections=[
            ConnectionSpec(
                src="stim",
                dst="pop0",
                weights=thr_w,
                bias=bias,
                destination_threshold=dst_thr,
            ),
            ConnectionSpec(src="base", dst="pop0", weights=base_w),
        ],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    # External bus layout follows connection order: stim (2 lanes) then base (2 lanes).
    direct_raster, folded_raster = _parity_rasters(ng, [2.0, 1.5, 1.0, 1.0], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded bias+destination-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "biased threshold workload should spike"


def test_folded_analogue_source_matches_direct() -> None:
    import numpy as np

    # An analogue li source population (membrane voltage, no spikes) feeds a lif
    # population: the lif input multiplies each source voltage by a weight (>> fraction),
    # reading the prior-tick committed voltage from the global v_bus — exactly the direct
    # path's registered v_out term. The li pop (pop 0) is driven by per-neuron external
    # current; its voltage rises toward I so the lif fan-in eventually crosses threshold.
    n_a, n_b = 3, 3
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=n_a, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=n_b, params={}, dt=1.0)
    weights = np.full((n_b, n_a), 0.5, dtype=np.float32)
    ng = NeuronGraph(
        populations=[li, lif],
        connections=[ConnectionSpec(src="a", dst="b", weights=weights)],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    # External lanes drive the li pop (the connection-less first population).
    direct_raster, folded_raster = _parity_rasters(ng, [4.0, 3.5, 3.0], n_a + n_b)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded analogue-source raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "analogue-fed lif population should spike"


def test_folded_analogue_source_threshold_matches_direct() -> None:
    import numpy as np

    # An analogue li source gated by a per-column source Threshold: a column contributes
    # its (un-multiplied) sign-extended weight only while the source voltage exceeds the
    # column threshold. The thresholds straddle the per-neuron steady-state voltages so
    # the gates toggle as the li voltages rise (column 2's threshold is never reached).
    n_a, n_b = 3, 3
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=n_a, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=n_b, params={}, dt=1.0)
    weights = np.full((n_b, n_a), 0.8, dtype=np.float32)  # gated weight ≈ 0.8 per passing column
    src_thr = np.array([1.0, 2.5, 5.0], dtype=np.float32)  # col 0 early, col 1 later, col 2 never
    ng = NeuronGraph(
        populations=[li, lif],
        connections=[ConnectionSpec(src="a", dst="b", weights=weights, source_threshold=src_thr)],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    direct_raster, folded_raster = _parity_rasters(ng, [4.0, 3.0, 2.0], n_a + n_b)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded analogue source-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "gated analogue-fed lif should spike"
