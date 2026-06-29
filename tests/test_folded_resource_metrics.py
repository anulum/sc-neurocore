# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded interconnect resource-metric tests

"""Architectural resource accounting for the folded (time-multiplexed) interconnect.

Covers :class:`FoldedResourceMetrics`, the ``_folded_resource_metrics`` builder across
the supported fold shapes (connection-less, external-weighted, recurrent, multi-state),
and that ``compile_network_to_fpga`` attaches the metrics only for the folded path.
"""

from __future__ import annotations

import numpy as np

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import FoldedResourceMetrics, compile_network_to_fpga, quantise_graph
from sc_neurocore.nir_bridge.fpga_compiler import _folded_resource_metrics
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec

_DW, _FR = 16, 8


def _quantised(ng: NeuronGraph) -> object:
    return quantise_graph(ng, Q88(data_width=_DW, fraction=_FR))


def _lif_pop(n: int, name: str = "pop0", ntype: str = "lif") -> NeuronSpec:
    return NeuronSpec(name=name, neuron_type=ntype, n_neurons=n, params={}, dt=1.0)


def test_connectionless_metrics() -> None:
    ng = NeuronGraph(
        populations=[_lif_pop(6)], connections=[], input_pop="pop0", output_pop="pop0", dt=1.0
    )
    m = _folded_resource_metrics(_quantised(ng), data_width=_DW)
    assert m == FoldedResourceMetrics(
        neurons=6,
        state_vars_per_neuron=1,
        pe_instances=1,
        shared_multipliers=0,
        state_ram_bits=6 * 1 * _DW,
        cycles_per_tick=7,
        direct_neuron_instances=6,
    )


def test_weighted_external_metrics() -> None:
    ng = NeuronGraph(
        populations=[_lif_pop(5)],
        connections=[ConnectionSpec(src="stim", dst="pop0", weights=np.ones((5, 3), np.float32))],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    m = _folded_resource_metrics(_quantised(ng), data_width=_DW)
    assert m.shared_multipliers == 3  # one per external source column, shared across neurons
    assert m.state_ram_bits == 5 * _DW
    assert m.cycles_per_tick == 6
    assert m.pe_instances == 1 and m.direct_neuron_instances == 5


def test_recurrent_excludes_spiking_multipliers() -> None:
    # External weighted conn contributes multipliers; the spiking self-connection does not.
    ng = NeuronGraph(
        populations=[_lif_pop(5)],
        connections=[
            ConnectionSpec(src="stim", dst="pop0", weights=np.ones((5, 3), np.float32)),
            ConnectionSpec(src="pop0", dst="pop0", weights=np.zeros((5, 5), np.float32)),
        ],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    m = _folded_resource_metrics(_quantised(ng), data_width=_DW)
    assert m.shared_multipliers == 3  # recurrent (spike-gated) adds no multiplier


def test_multistate_neuron_state_ram() -> None:
    # cuba_lif has two state variables (i_syn, v) → wider BRAM word.
    ng = NeuronGraph(
        populations=[_lif_pop(4, ntype="cuba_lif")],
        connections=[],
        input_pop="pop0",
        output_pop="pop0",
        dt=1.0,
    )
    m = _folded_resource_metrics(_quantised(ng), data_width=_DW)
    assert m.state_vars_per_neuron == 2
    assert m.state_ram_bits == 4 * 2 * _DW


def test_as_dict_round_trips_all_fields() -> None:
    m = FoldedResourceMetrics(
        neurons=8,
        state_vars_per_neuron=2,
        pe_instances=1,
        shared_multipliers=4,
        state_ram_bits=256,
        cycles_per_tick=9,
        direct_neuron_instances=8,
    )
    assert m.as_dict() == {
        "neurons": 8,
        "state_vars_per_neuron": 2,
        "pe_instances": 1,
        "shared_multipliers": 4,
        "state_ram_bits": 256,
        "cycles_per_tick": 9,
        "direct_neuron_instances": 8,
    }


def test_compile_network_attaches_metrics_only_when_folded() -> None:
    ng = NeuronGraph(
        populations=[_lif_pop(6)], connections=[], input_pop="pop0", output_pop="pop0", dt=1.0
    )
    folded = compile_network_to_fpga(ng, interconnect="folded")
    assert folded.interconnect == "folded"
    assert folded.folded_metrics is not None
    assert folded.folded_metrics.neurons == 6
    assert folded.folded_metrics.direct_neuron_instances == 6

    direct = compile_network_to_fpga(ng, interconnect="direct")
    assert direct.interconnect == "direct"
    assert direct.folded_metrics is None

    auto = compile_network_to_fpga(ng)
    assert auto.folded_metrics is None  # never auto-selected
