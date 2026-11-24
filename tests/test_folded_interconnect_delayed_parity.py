# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (delayed_parity) from former test_folded_interconnect.py

"""Delayed-spike parity of the folded and direct interconnects."""

from __future__ import annotations

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import quantise_graph
from sc_neurocore.nir_bridge.fpga_direct_interconnect import (
    build_direct_interconnect as _build_top_direct,
)
from sc_neurocore.nir_bridge.fpga_folded_interconnect import (
    build_folded_interconnect as _build_top_folded,
)
from sc_neurocore.nir_bridge.fpga_neuron_rtl import build_neuron_module as _build_neuron_module
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec
from sc_neurocore.nir_bridge.neuron_graph import NeuronGraph
from sc_neurocore.nir_bridge.neuron_graph import NeuronSpec
from tests.folded_interconnect_support import (
    _DW,
    _FR,
    _STEPS,
    _cosim,
    _delayed_recurrent_graph,
    _direct_tb,
    _folded_tb,
    _parity_rasters,
    pytestmark as pytestmark,
)


def test_folded_delayed_recurrent_matches_direct() -> None:
    """Delayed recurrent spikes give matching rasters."""
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


def test_folded_delayed_analogue_source_matches_direct() -> None:
    """A delayed analogue source gives matching rasters."""
    import numpy as np

    # A *delayed* analogue li source feeding a lif population: a delay of d ticks reads
    # v_bus_hist_d (the voltage bus committed d ticks ago), the exact double-buffer
    # analogue of spike_bus_hist for delayed spikes, mirroring direct's v_d{d} chain.
    n_a, n_b = 3, 3
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=n_a, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=n_b, params={}, dt=1.0)
    weights = np.full((n_b, n_a), 0.5, dtype=np.float32)
    ng = NeuronGraph(
        populations=[li, lif],
        connections=[ConnectionSpec(src="a", dst="b", weights=weights, delay_steps=2)],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    # A depth-2 analogue delay materialises a depth-2 voltage-bus history shift-register.
    q = Q88(data_width=_DW, fraction=_FR)
    _pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", quantise_graph(ng, q), data_width=_DW, fraction=_FR
    )
    assert "v_bus_hist_2" in folded_top
    assert "v_bus_hist_3" not in folded_top

    direct_raster, folded_raster = _parity_rasters(ng, [4.0, 3.5, 3.0], n_a + n_b)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded delayed-analogue raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "delayed analogue-fed lif should spike"
