# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — trace STDP in the FPGA network compiler, against its integer golden

"""Trace STDP compiles in the direct and the folded network, bit for bit with its golden.

:func:`sc_neurocore.nir_bridge.stdp.stdp_tick` is the integer rule; it follows
``Projection.update_plasticity`` within the format's rounding. Both interconnects
are simulated in Icarus: every tick the raster and every weight are printed, and
the weights must equal the golden applied to the printed raster, the folded raster
must equal the direct one, and learning must change the weights. Unsupported
plastic connections are refused. Without Icarus the cosimulation cases skip.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from sc_neurocore.nir_bridge.stdp import STDPRule, stdp_tick
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
    _build_top_folded,
    _dequantised_pop,
    quantise_graph,
)

_LIF = {
    "equations": ["dv/dt = -v / tau + I"],
    "threshold": "v > 1.0",
    "reset": "v = 0.0",
    "default_params": {"tau": 4.0},
}
_RULE = STDPRule(a_plus=0.08, a_minus=0.05, tau=3.0, w_min=0.0, w_max=1.5)
_N_PRE, _N_POST, _STEPS = 4, 2, 120
_WEIGHTS = [[0.9, 0.8, 0.0, 0.7], [0.6, 1.0, 0.9, 0.0]]


def _graph(rule: STDPRule | None = _RULE, **options: Any) -> NeuronGraph:
    """Return stim → four pre neurons (different drives) → two post neurons, plastic."""
    pre = NeuronSpec(name="pre", neuron_type="pre_t", n_neurons=_N_PRE, dt=0.25, template=_LIF)
    post = NeuronSpec(name="post", neuron_type="post_t", n_neurons=_N_POST, dt=0.25, template=_LIF)
    drive = np.array([[2.0], [1.6], [1.2], [1.0]], dtype=np.float32)
    weights = np.array(_WEIGHTS, dtype=np.float32)
    conns = [
        ConnectionSpec(src="stim", dst="pre", weights=drive),
        ConnectionSpec(src="pre", dst="post", weights=weights, plasticity=rule, **options),
    ]
    return NeuronGraph(
        populations=[pre, post], connections=conns, input_pop="stim", output_pop="post", dt=0.25
    )


def _synapses() -> list[tuple[int, int]]:
    """Return the (post, pre) pairs whose initial weight is non-zero."""
    return [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 2)]


def _testbench(top: str, weight: str | None, folded: bool) -> str:
    """Return a testbench printing the raster and (unless ``None``) every weight per tick."""
    pairs = _synapses() if weight is not None else []
    names = "".join(" %0d" for _ in pairs)
    args = "".join(f", $signed(uut.{(weight or '').format(j=j, i=i)})" for j, i in pairs)
    ports = ".spike_bus(spike_bus), .tick_done(tick_done)" if folded else ".spike_bus(spike_bus)"
    sample = "if (tick_done) begin" if folded else "begin"
    stim = Q88(data_width=_DW, fraction=_FR).encode(1.0)
    return "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb;",
            "reg clk; reg rst_n; reg en; wire [5:0] spike_bus; wire tick_done;",
            f"{top} uut (.clk(clk), .rst_n(rst_n), .en(en), .I_ext_flat({_DW}'d{stim}), {ports});",
            "initial clk = 0; always #5 clk = ~clk;",
            "integer ticks;",
            "initial begin",
            "  rst_n = 0; en = 1; ticks = 0; #23; rst_n = 1;",
            f"  while (ticks < {_STEPS}) begin @(posedge clk); #1;",
            f'    {sample} $display("T %b{names}", spike_bus{args}); ticks = ticks + 1; end',
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )


def _run(sources: dict[str, str]) -> list[tuple[str, list[int]]]:
    """Simulate and return (raster, weights) per tick."""
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
        out = subprocess.run(["vvp", f"{d}/sim"], capture_output=True, text=True, timeout=180)
    rows = []
    for line in out.stdout.splitlines():
        if line.startswith("T "):
            fields = line.split()
            rows.append((fields[1], [int(x) for x in fields[2:]]))
    assert len(rows) == _STEPS
    return rows


def _simulate(
    graph: NeuronGraph,
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """Return the per-tick (raster, weights) of the direct and the folded top."""
    qgraph = quantise_graph(graph, Q88(data_width=_DW, fraction=_FR))
    modules = {
        f"mod_{pop.neuron_type}": _build_neuron_module(
            pop.neuron_type, _dequantised_pop(pop, _FR), data_width=_DW, fraction=_FR
        )
        for pop in qgraph.populations
    }
    direct_top = _build_top_direct("sc_stdp", qgraph, data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded("sc_stdp_f", qgraph, data_width=_DW, fraction=_FR)
    direct = _run(
        {**modules, "top": direct_top, "tb": _testbench("sc_stdp", "stdp1_w{j}_{i}_next", False)}
    )
    folded = _run(
        {
            "pe": "\n\n".join(pe_modules.values()),
            "top": folded_top,
            "tb": _testbench("sc_stdp_f", "stdp1_w{j}_{i}", True),
        }
    )
    return direct, folded


def _golden(rows: list[tuple[str, list[int]]]) -> list[list[int]]:
    """Apply stdp_tick to the printed raster from the initial weights; return weights per tick."""
    rule = _RULE.quantised(_DW, _FR)
    q = Q88(data_width=_DW, fraction=_FR)
    initial = np.array(_WEIGHTS, dtype=np.float32)
    weights = [[q.encode(float(w)) if w else 0 for w in row] for row in initial]
    weights = [[w - (1 << _DW) if w >= 1 << (_DW - 1) else w for w in row] for row in weights]
    mask = [[bool(w) for w in row] for row in initial]
    pre, post = [0] * _N_PRE, [0] * _N_POST
    out = []
    for bits, _ in rows:
        spikes = [int(b) for b in reversed(bits)]
        weights, pre, post = stdp_tick(
            weights, mask, pre, post, spikes[:_N_PRE], spikes[_N_PRE:], rule
        )
        out.append([weights[j][i] for j, i in _synapses()])
    return out


def test_the_rule_is_validated_and_quantised() -> None:
    """A negative amplitude, a non-positive tau, crossed bounds or a lost amplitude are refused."""
    for bad in (
        {"a_plus": -0.1},
        {"tau": 0.0},
        {"w_min": 1.0, "w_max": 0.5},
        {"a_minus": float("nan")},
    ):
        with pytest.raises(ValueError, match="STDP"):
            STDPRule(**bad)
    with pytest.raises(ValueError, match="rounds to zero"):
        STDPRule(a_plus=0.001).quantised(16, 8)
    words = _RULE.quantised(16, 8)
    assert (words.decay, words.a_plus, words.a_minus, words.w_max) == (183, 20, 13, 384)


def test_the_golden_follows_the_network_rule_within_rounding() -> None:
    """At Q16.16 the integer golden stays within 1e-3 of Projection.update_plasticity for 50 ticks."""
    rng = np.random.default_rng(7)
    rule = STDPRule(a_plus=0.01, a_minus=0.012, tau=20.0, w_min=0.0, w_max=100.0)
    words = rule.quantised(32, 16)
    initial = np.array([[0.5, 0.4, 0.0], [0.2, 0.6, 0.5]])
    source, target = Population("SCLapicqueLIFNeuron", 3), Population("SCLapicqueLIFNeuron", 2)
    indptr = np.array([0, 2, 4, 5])
    indices = np.array([0, 1, 0, 1, 1])
    data = np.array([0.5, 0.2, 0.4, 0.6, 0.5])
    projection = Projection(
        source, target, 1.0, topology=(indptr, indices, data), plasticity="stdp"
    )
    weights = [[round(w * 65536) for w in row] for row in initial]
    mask = [[bool(w) for w in row] for row in initial]
    pre, post = [0] * 3, [0] * 2
    for _ in range(50):
        pre_spikes = (rng.random(3) < 0.3).astype(np.int8)
        post_spikes = (rng.random(2) < 0.3).astype(np.int8)
        projection.update_plasticity(pre_spikes, post_spikes, a_plus=0.01, a_minus=0.012, tau=20.0)
        weights, pre, post = stdp_tick(
            weights, mask, pre, post, pre_spikes.tolist(), post_spikes.tolist(), words
        )
    dense = np.zeros((2, 3))
    for i in range(3):
        for k in range(indptr[i], indptr[i + 1]):
            dense[indices[k], i] = projection.data[k]
    np.testing.assert_allclose(np.array(weights) / 65536.0, dense, atol=1e-3)


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
def test_direct_and_folded_learn_exactly_as_the_golden() -> None:
    """Every tick, both tops' weights equal stdp_tick on the raster, and the rasters agree."""
    direct, folded = _simulate(_graph())
    assert [bits for bits, _ in folded] == [bits for bits, _ in direct]
    golden = _golden(direct)
    assert [weights for _, weights in direct] == golden
    assert [weights for _, weights in folded] == golden
    assert golden[-1] != golden[0], "the weights learned"
    assert any(bits[:2] != "00" for bits, _ in direct), "the post neurons spike"


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog/vvp not installed")
def test_learning_changes_the_post_raster() -> None:
    """The control: without plasticity the same graph spikes differently."""
    plastic, _ = _simulate(_graph())
    qgraph = quantise_graph(_graph(None), Q88(data_width=_DW, fraction=_FR))
    assert all(conn.plasticity is None for conn in qgraph.connections)
    static_top = _build_top_direct("sc_stdp", qgraph, data_width=_DW, fraction=_FR)
    assert "stdp1_" not in static_top
    static = _run(
        {
            **{
                f"mod_{pop.neuron_type}": _build_neuron_module(
                    pop.neuron_type, _dequantised_pop(pop, _FR), data_width=_DW, fraction=_FR
                )
                for pop in qgraph.populations
            },
            "top": static_top,
            "tb": _testbench("sc_stdp", None, False),
        }
    )
    assert [bits[:2] for bits, _ in plastic] != [bits[:2] for bits, _ in static]


@pytest.mark.parametrize(
    "options",
    [{"delay_steps": 1}, {"source_threshold": np.full(_N_PRE, 0.5, dtype=np.float32)}],
)
def test_unsupported_plastic_connections_are_refused(options: dict[str, Any]) -> None:
    """A delayed or thresholded plastic connection is refused by both interconnects."""
    qgraph = quantise_graph(_graph(**options), Q88(data_width=_DW, fraction=_FR))
    with pytest.raises(ValueError, match="STDP"):
        _build_top_direct("sc_stdp", qgraph, data_width=_DW, fraction=_FR)
    with pytest.raises(ValueError, match="STDP"):
        _build_top_folded("sc_stdp_f", qgraph, data_width=_DW, fraction=_FR)


def test_an_external_plastic_source_is_refused() -> None:
    """STDP needs spike-timed pre neurons; an external input column has none."""
    graph = _graph()
    graph.connections[0] = ConnectionSpec(
        src="stim", dst="pre", weights=np.ones((_N_PRE, 1), np.float32), plasticity=_RULE
    )
    qgraph = quantise_graph(graph, Q88(data_width=_DW, fraction=_FR))
    with pytest.raises(ValueError, match="binary spiking source"):
        _build_top_direct("sc_stdp", qgraph, data_width=_DW, fraction=_FR)


def test_a_large_plastic_network_stays_off_the_address_event_path() -> None:
    """The AER fan-out keeps no weight state, so auto selection stays direct and says why."""
    pre = NeuronSpec(name="pre", neuron_type="pre_t", n_neurons=70, dt=0.25, template=_LIF)
    post = NeuronSpec(name="post", neuron_type="post_t", n_neurons=2, dt=0.25, template=_LIF)
    graph = NeuronGraph(
        populations=[pre, post],
        connections=[
            ConnectionSpec(src="stim", dst="pre", weights=np.ones((70, 1), np.float32)),
            ConnectionSpec(
                src="pre", dst="post", weights=np.full((2, 70), 0.1, np.float32), plasticity=_RULE
            ),
        ],
        input_pop="stim",
        output_pop="post",
        dt=0.25,
    )
    result = compile_network_to_fpga(graph, module_name="sc_stdp_big")
    assert result.interconnect == "direct"
    assert any("STDP" in warning for warning in result.warnings)
    assert "stdp1_w1_69" in result.top_module
