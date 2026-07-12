# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA compilation-result contract tests

"""Verify stable public result metadata and serialisation paths."""

from __future__ import annotations

import pickle

from sc_neurocore.nir_bridge import (
    FoldedResourceMetrics,
    NetworkCompilationResult,
    compile_network_to_fpga,
)
from sc_neurocore.nir_bridge.neuron_graph import NeuronGraph, NeuronSpec


def test_folded_result_metrics_are_json_ready_and_pickle_stable() -> None:
    graph = NeuronGraph(
        populations=[NeuronSpec("layer", "lif", 3)],
        connections=[],
        input_pop="layer",
        output_pop="layer",
    )

    result = compile_network_to_fpga(graph, interconnect="folded")

    assert isinstance(result, NetworkCompilationResult)
    assert isinstance(result.folded_metrics, FoldedResourceMetrics)
    assert result.folded_metrics.as_dict() == {
        "neurons": 3,
        "state_vars_per_neuron": 1,
        "pe_instances": 1,
        "shared_multipliers": 0,
        "state_ram_bits": 48,
        "cycles_per_tick": 4,
        "direct_neuron_instances": 3,
        "populations": 1,
        "param_rom_bits": 0,
    }
    restored = pickle.loads(pickle.dumps(result.folded_metrics))
    assert restored == result.folded_metrics


def test_result_types_keep_the_historical_public_module_path() -> None:
    assert FoldedResourceMetrics.__module__ == "sc_neurocore.nir_bridge.fpga_compiler"
    assert NetworkCompilationResult.__module__ == "sc_neurocore.nir_bridge.fpga_compiler"
