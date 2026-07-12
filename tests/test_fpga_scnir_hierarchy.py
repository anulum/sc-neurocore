# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR hierarchy RTL contract tests

"""Validate concrete hierarchy-boundary Verilog and semantic weight wiring."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.ir.scnir_schema import (
    SCNIRDocument,
    SCNIRHierarchyInstance,
    SCNIRHierarchyPort,
)
from sc_neurocore.nir_bridge.fpga_direct_interconnect import build_direct_interconnect
from sc_neurocore.nir_bridge.fpga_scnir_hierarchy import (
    build_scnir_hierarchy_instance_block,
    build_scnir_hierarchy_modules,
    resolve_hierarchy_weight_literals,
)
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronSpec
from sc_neurocore.nir_bridge.quantise_params import QuantisedGraph


def _document(*instances: SCNIRHierarchyInstance) -> SCNIRDocument:
    """Build a typed hierarchy document for RTL emission."""
    return SCNIRDocument(producer="test", streams=(), hierarchy=instances)


def _quantised_graph(
    populations: list[NeuronSpec],
    connections: list[ConnectionSpec],
) -> QuantisedGraph:
    """Build typed fixed-point graph input for hierarchy interconnect tests."""
    return QuantisedGraph(
        populations,
        connections,
        Q88(),
        connections[0].src if connections else populations[0].name,
        populations[-1].name,
        1.0,
        total_neurons=sum(population.n_neurons for population in populations),
        total_synapses=sum(connection.weights.size for connection in connections),
    )


def test_hierarchy_module_emits_scalar_packed_and_zero_outputs() -> None:
    instance = SCNIRHierarchyInstance(
        instance_id="nested",
        module_name="nested_boundary",
        ports=(
            SCNIRHierarchyPort("input_word", "input", "input.stream", "weight", 16),
            SCNIRHierarchyPort("flag", "output", "flag.stream", "spike", 1),
            SCNIRHierarchyPort("zero_word", "output", "zero.stream", "weight", 16),
            SCNIRHierarchyPort("scalar", "output", "scalar.weight", "weight", 16),
            SCNIRHierarchyPort("packed", "output", "packed.weight", "weight", 32),
        ),
    )

    modules = build_scnir_hierarchy_modules(
        _document(instance),
        weight_literals={"scalar.weight": (-1,), "packed.weight": (2, -3)},
    )

    source = modules["nested_boundary"]
    assert "assign input_word" not in source
    assert "assign flag = 1'b0;" in source
    assert "assign zero_word = 16'sd0;" in source
    assert "assign scalar = 16'shffff;" in source
    assert "assign packed[0 +: 16] = 16'sh0002;" in source
    assert "assign packed[16 +: 16] = 16'shfffd;" in source


def test_hierarchy_instance_block_wires_input_and_output_widths() -> None:
    instance = SCNIRHierarchyInstance(
        instance_id="nested",
        module_name="nested_boundary",
        ports=(
            SCNIRHierarchyPort("enable", "input", "enable.stream", "spike", 1),
            SCNIRHierarchyPort("weight", "output", "weight.stream", "weight", 16),
            SCNIRHierarchyPort("wide", "output", "wide.stream", "weight", 24),
        ),
    )

    lines = build_scnir_hierarchy_instance_block((instance,), data_width=16)
    source = "\n".join(lines)

    assert "wire nested_boundary__enable;" in source
    assert "assign nested_boundary__enable = 1'b0;" in source
    assert "wire signed [DATA_WIDTH - 1:0] nested_boundary__weight;" in source
    assert "wire signed [23:0] nested_boundary__wide;" in source
    assert ".enable(nested_boundary__enable)" in source


def test_empty_hierarchy_instance_block_is_empty() -> None:
    assert build_scnir_hierarchy_instance_block((), data_width=16) == []

    invalid = SCNIRHierarchyInstance(
        "invalid",
        "invalid_module",
        (SCNIRHierarchyPort("bad", "input", "bad.stream", "weight", 0),),
    )
    with pytest.raises(ValueError, match="non-positive bit width"):
        build_scnir_hierarchy_instance_block((invalid,), data_width=16)


def test_hierarchy_contract_rejects_invalid_boundaries() -> None:
    empty = SCNIRHierarchyInstance("empty", "empty_module", ())
    with pytest.raises(ValueError, match="has no ports"):
        build_scnir_hierarchy_modules(_document(empty), weight_literals={})

    invalid_width = SCNIRHierarchyInstance(
        "bad-width",
        "bad_width",
        (SCNIRHierarchyPort("bad", "output", "bad.stream", "weight", 0),),
    )
    with pytest.raises(ValueError, match="non-positive bit width"):
        build_scnir_hierarchy_modules(_document(invalid_width), weight_literals={})

    duplicate_a = SCNIRHierarchyInstance(
        "a",
        "same_module",
        (SCNIRHierarchyPort("a", "output", "a.stream", "weight", 1),),
    )
    duplicate_b = SCNIRHierarchyInstance(
        "b",
        "same_module",
        (SCNIRHierarchyPort("b", "output", "b.stream", "weight", 1),),
    )
    with pytest.raises(ValueError, match="duplicate SC-NIR hierarchy module name"):
        build_scnir_hierarchy_modules(
            _document(duplicate_a, duplicate_b),
            weight_literals={},
        )


def test_packed_hierarchy_literal_width_must_be_divisible() -> None:
    instance = SCNIRHierarchyInstance(
        "nested",
        "nested_boundary",
        (SCNIRHierarchyPort("packed", "output", "packed.weight", "weight", 24),),
    )

    with pytest.raises(ValueError, match="not divisible by literal count"):
        build_scnir_hierarchy_modules(
            _document(instance),
            weight_literals={"packed.weight": (1, 2, 3, 4, 5)},
        )


def test_hierarchy_weight_literal_lookup_rejects_unknown_and_misaligned_streams() -> None:
    connection = ConnectionSpec(
        "left",
        "right",
        np.array([[1, 2, 3]], dtype=np.int64),
    )
    qgraph = QuantisedGraph(
        populations=[],
        connections=[connection],
        q=Q88(),
        input_pop="left",
        output_pop="right",
        dt=1.0,
    )
    unknown = SCNIRHierarchyInstance(
        "unknown",
        "unknown_module",
        (SCNIRHierarchyPort("weight", "output", "unknown.weight", "weight", 16),),
    )
    with pytest.raises(ValueError, match="references unknown weight stream"):
        resolve_hierarchy_weight_literals(_document(unknown), qgraph)

    stream_id = "conn.left_to_right.weight"
    misaligned = SCNIRHierarchyInstance(
        "misaligned",
        "misaligned_module",
        (SCNIRHierarchyPort("weight", "output", stream_id, "weight", 16),),
    )
    with pytest.raises(ValueError, match="not divisible by flattened weight count"):
        resolve_hierarchy_weight_literals(_document(misaligned), qgraph)


def test_hierarchy_weight_lookup_ignores_non_weight_outputs_and_inputs() -> None:
    instance = SCNIRHierarchyInstance(
        "metadata",
        "metadata_module",
        (
            SCNIRHierarchyPort("input", "input", "missing.input", "weight", 16),
            SCNIRHierarchyPort("spike", "output", "missing.spike", "spike", 1),
        ),
    )
    qgraph = QuantisedGraph([], [], Q88(), "", "", 1.0)

    assert resolve_hierarchy_weight_literals(_document(instance), qgraph) == {}


def test_weight_resolution_sanitises_non_identifier_connection_names() -> None:
    connection = ConnectionSpec(
        "12 bad/name",
        "---",
        np.array([[7]], dtype=np.int64),
    )
    stream_id = "conn.s_12_bad_name_to_stream.weight"
    instance = SCNIRHierarchyInstance(
        "sanitised",
        "sanitised_module",
        (SCNIRHierarchyPort("weight", "output", stream_id, "weight", 16),),
    )
    qgraph = QuantisedGraph([], [connection], Q88(), "", "", 1.0)

    assert resolve_hierarchy_weight_literals(_document(instance), qgraph) == {stream_id: (7,)}


def test_direct_builder_rejects_duplicate_and_undersized_semantic_weight_ports() -> None:
    population = NeuronSpec("target", "lif", 1)
    base_graph = _quantised_graph([population], [])
    first = SCNIRHierarchyInstance(
        "first",
        "first_module",
        (SCNIRHierarchyPort("weight", "output", "shared.weight", "weight", 16),),
    )
    second = SCNIRHierarchyInstance(
        "second",
        "second_module",
        (SCNIRHierarchyPort("weight", "output", "shared.weight", "weight", 16),),
    )

    assert "first_module" in build_direct_interconnect(
        "unrelated",
        base_graph,
        scnir_hierarchy=(first,),
        scnir_semantic_hierarchy_stream_ids=frozenset(),
    )
    with pytest.raises(ValueError, match="duplicate hierarchy output"):
        build_direct_interconnect(
            "duplicate",
            base_graph,
            scnir_hierarchy=(first, second),
            scnir_semantic_hierarchy_stream_ids=frozenset({"shared.weight"}),
        )

    connection = ConnectionSpec(
        "stim",
        "target",
        np.array([[1, 2]], dtype=np.int64),
    )
    stream_id = "conn.stim_to_target.weight"
    undersized = SCNIRHierarchyInstance(
        "undersized",
        "undersized_module",
        (SCNIRHierarchyPort("weights", "output", stream_id, "weight", 24),),
    )
    connected_graph = _quantised_graph([population], [connection])
    with pytest.raises(ValueError, match="cannot provide weight index"):
        build_direct_interconnect(
            "undersized",
            connected_graph,
            scnir_hierarchy=(undersized,),
            scnir_semantic_hierarchy_stream_ids=frozenset({stream_id}),
        )
