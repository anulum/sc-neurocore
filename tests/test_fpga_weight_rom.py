# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA weight-ROM emission tests

"""Exercise combined weight-ROM artefacts through network compilation."""

from __future__ import annotations

import numpy as np

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import compile_network_to_fpga
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec


def test_connectionless_graph_emits_zero_weight_rom() -> None:
    graph = NeuronGraph([NeuronSpec("layer", "lif", 1)], [], "layer", "layer")

    result = compile_network_to_fpga(graph, interconnect="direct")

    assert "Auto-generated weight ROM (empty" in result.weight_rom
    assert "assign data = 16'sd0" in result.weight_rom


def test_weight_rom_flattens_connections_in_graph_order() -> None:
    populations = [NeuronSpec("left", "lif", 2), NeuronSpec("right", "lif", 1)]
    connections = [
        ConnectionSpec("stim", "left", np.array([[0.5], [-0.25]], dtype=np.float32)),
        ConnectionSpec("left", "right", np.array([[0.125, 0.0]], dtype=np.float32)),
    ]
    graph = NeuronGraph(populations, connections, "stim", "right")
    q = Q88(data_width=16, fraction=8)

    result = compile_network_to_fpga(graph, interconnect="direct")

    expected = [q.encode(0.5), q.encode(-0.25), q.encode(0.125), q.encode(0.0)]
    for address, value in enumerate(expected):
        encoded = value & 0xFFFF
        assert f"2'd{address}: data = 16'sh{encoded:04x}" in result.weight_rom
    assert "// stim → left: offset=0, count=2" in result.weight_rom
    assert "// left → right: offset=2, count=2" in result.weight_rom
