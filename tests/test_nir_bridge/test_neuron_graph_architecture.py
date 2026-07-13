# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware neuron graph architecture contracts

"""Lock compatibility, ownership, acyclicity, and responsibility boundaries."""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from sc_neurocore import nir_bridge
from sc_neurocore.nir_bridge import neuron_graph as historical

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "src/sc_neurocore/nir_bridge"
TEST_DIR = ROOT / "tests/test_nir_bridge"
BENCHMARK_PATH = ROOT / "benchmarks/bench_nir_graph.py"
PROBE_PATH = ROOT / "benchmarks/_nir_graph_probe.py"
BENCHMARK_RESULT_PATH = ROOT / "benchmarks/results/bench_nir_graph.json"

SYMBOL_OWNERS = {
    "neuron_graph_builder": {"from_scnetwork"},
    "neuron_graph_connections": {
        "_fold_connection_scales",
        "_resolve_weight_destination",
        "_resolve_weight_source",
    },
    "neuron_graph_contracts": {
        "ConnectionSpec",
        "HierarchyInstanceSpec",
        "NeuronGraph",
        "NeuronSpec",
    },
    "neuron_graph_dense": {
        "_conv1d_to_dense_matrix",
        "_conv2d_to_dense_matrix",
        "_pool2d_to_dense_matrix",
        "_weight_matrix_and_bias",
    },
    "neuron_graph_hierarchy": {
        "_hdl_identifier_fragment",
        "_inline_single_port_subgraphs",
        "_topological_order",
    },
    "neuron_graph_metadata": {
        "_broadcast_scale",
        "_broadcast_threshold",
        "_compose_delay_steps",
        "_compose_scale",
        "_delay_steps",
        "_delay_steps_array",
        "_fit_delay_steps_to_width",
        "_flatten_widths",
        "_scale_vector",
        "_shape_width",
        "_threshold_vector",
    },
    "neuron_graph_nodes": {"_extract_neuron_params", "_node_logical_width"},
}
RESPONSIBILITY_MODULES = {
    "neuron_graph_builder",
    "neuron_graph_connections",
    "neuron_graph_contracts",
    "neuron_graph_dense",
    "neuron_graph_hierarchy",
    "neuron_graph_metadata",
    "neuron_graph_nodes",
}
HISTORICAL_SYMBOLS = {
    "ConnectionSpec",
    "HierarchyInstanceSpec",
    "NeuronGraph",
    "NeuronSpec",
    "_broadcast_scale",
    "_broadcast_threshold",
    "_compose_delay_steps",
    "_compose_scale",
    "_conv1d_to_dense_matrix",
    "_conv2d_to_dense_matrix",
    "_delay_steps",
    "_delay_steps_array",
    "_extract_neuron_params",
    "_fit_delay_steps_to_width",
    "_flatten_widths",
    "_fold_connection_scales",
    "_hdl_identifier_fragment",
    "_inline_single_port_subgraphs",
    "_node_logical_width",
    "_pool2d_to_dense_matrix",
    "_resolve_weight_destination",
    "_resolve_weight_source",
    "_scale_vector",
    "_shape_width",
    "_threshold_vector",
    "_topological_order",
    "_weight_matrix_and_bias",
    "from_scnetwork",
}


def _tree(path: Path) -> ast.Module:
    """Parse one tracked Python source file."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_definitions(path: Path) -> set[str]:
    """Return top-level class and function definitions."""
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _module_dependencies(module_name: str) -> set[str]:
    """Return imports from other neuron-graph responsibility modules."""
    dependencies: set[str] = set()
    prefix = "sc_neurocore.nir_bridge."
    for node in ast.walk(_tree(SOURCE_DIR / f"{module_name}.py")):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module.startswith(prefix):
            dependency = node.module.removeprefix(prefix).split(".", maxsplit=1)[0]
            if dependency in RESPONSIBILITY_MODULES:
                dependencies.add(dependency)
    return dependencies


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    """Reject cycles and report the active import path."""
    complete: set[str] = set()
    active: list[str] = []

    def visit(module_name: str) -> None:
        if module_name in active:
            cycle = " -> ".join([*active[active.index(module_name) :], module_name])
            raise AssertionError(f"neuron-graph import cycle: {cycle}")
        if module_name in complete:
            return
        active.append(module_name)
        for dependency in sorted(graph[module_name]):
            visit(dependency)
        active.pop()
        complete.add(module_name)

    for module_name in sorted(graph):
        visit(module_name)


def test_historical_facade_defines_no_implementation() -> None:
    """Keep the historical module as a re-export-only compatibility surface."""
    assert _top_level_definitions(SOURCE_DIR / "neuron_graph.py") == set()


def test_historical_facade_reexports_every_established_definition() -> None:
    """Retain every public and private symbol defined by the monolithic module."""
    assert {name for name in HISTORICAL_SYMBOLS if hasattr(historical, name)} == HISTORICAL_SYMBOLS


def test_historical_definitions_have_exactly_one_owner() -> None:
    """Give every established definition one focused implementation owner."""
    actual: dict[str, list[str]] = {}
    for module_name, expected_symbols in SYMBOL_OWNERS.items():
        definitions = _top_level_definitions(SOURCE_DIR / f"{module_name}.py")
        for symbol in expected_symbols & definitions:
            actual.setdefault(symbol, []).append(module_name)

    assert actual == {
        symbol: [module_name]
        for module_name, symbols in SYMBOL_OWNERS.items()
        for symbol in symbols
    }


def test_responsibility_import_graph_is_acyclic() -> None:
    """Keep implementation dependencies directed toward lower-level contracts."""
    graph = {
        module_name: _module_dependencies(module_name) for module_name in RESPONSIBILITY_MODULES
    }
    _assert_acyclic(graph)


def test_historical_identity_and_pickle_paths_are_stable() -> None:
    """Preserve direct imports and serialised qualified names."""
    names = (
        "ConnectionSpec",
        "HierarchyInstanceSpec",
        "NeuronGraph",
        "NeuronSpec",
        "from_scnetwork",
    )
    for name in names:
        definition = getattr(historical, name)
        assert definition.__module__ == "sc_neurocore.nir_bridge.neuron_graph"
        assert pickle.loads(pickle.dumps(definition)) is definition


def test_package_exports_reuse_historical_objects() -> None:
    """Keep package-root imports identical to the historical module objects."""
    assert nir_bridge.ConnectionSpec is historical.ConnectionSpec
    assert nir_bridge.NeuronGraph is historical.NeuronGraph
    assert nir_bridge.NeuronSpec is historical.NeuronSpec
    assert nir_bridge.from_scnetwork is historical.from_scnetwork
    assert "HierarchyInstanceSpec" not in nir_bridge.__all__


def test_graph_summary_reports_population_and_connection_contracts() -> None:
    """Render counts, types, bias, and delay through the public graph record."""
    source = historical.NeuronSpec("source", "li", 2)
    target = historical.NeuronSpec("target", "lif", 1)
    connection = historical.ConnectionSpec(
        "source",
        "target",
        np.array([[0.25, -0.5]], dtype=np.float32),
        bias=np.array([0.125], dtype=np.float32),
        delay_steps=2,
    )
    graph = historical.NeuronGraph(
        [source, target],
        [connection],
        input_pop="source",
        output_pop="target",
        dt=0.5,
    )

    assert graph.total_neurons == 3
    assert graph.total_synapses == 2
    assert graph.neuron_types == {"li", "lif"}
    assert graph.summary() == (
        "NeuronGraph: 2 populations, 1 connections\n"
        "  Total neurons:  3\n"
        "  Total synapses: 2\n"
        "  Neuron types:   li, lif\n"
        "  Input:  source\n"
        "  Output: target\n"
        "  dt: 0.5\n\n"
        "  Populations:\n"
        "    source: li × 2\n"
        "    target: lif × 1\n\n"
        "  Connections:\n"
        "    source → target: 2→1 +bias delay=2"
    )


def test_source_and_test_files_remain_responsibility_sized() -> None:
    """Prevent replacement source or test GodFiles in this refactor lane."""
    source_sizes = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in SOURCE_DIR.glob("neuron_graph*.py")
    }
    test_sizes = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in TEST_DIR.glob("*.py")
    }

    assert max(source_sizes.values()) <= 330, source_sizes
    assert max(test_sizes.values()) <= 300, test_sizes
    assert len(BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()) <= 380
    assert len(PROBE_PATH.read_text(encoding="utf-8").splitlines()) <= 240


def test_benchmark_evidence_is_source_bound_and_byte_identical() -> None:
    """Pin the measured parent/candidate fidelity result to its live producers."""
    payload = json.loads(BENCHMARK_RESULT_PATH.read_text(encoding="utf-8"))
    source_paths = sorted(SOURCE_DIR.glob("neuron_graph*.py"))
    expected_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }

    assert payload["benchmark_sha256"] == hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()
    assert payload["probe_sha256"] == hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest()
    assert payload["source_file_count"] == len(source_paths) == 8
    assert payload["source_hashes"] == expected_hashes
    assert payload["configuration"]["iterations"] == 30
    assert payload["configuration"]["warmups"] == 2
    assert payload["results"]["parent"]["sample_count"] == 30
    assert payload["results"]["candidate"]["sample_count"] == 30
    expected_comparison = {
        "generated_output_byte_identical": True,
        "generated_sha256": "32498fa1106229a4fe064862e20b86e0f0b1f0d42f8598d1988e06e68c13ef13",
        "generated_bytes": 22860,
        "population_count": 2,
        "connection_count": 2,
        "total_neurons": 8,
        "total_synapses": 36,
    }
    assert {key: payload["comparison"][key] for key in expected_comparison} == expected_comparison
