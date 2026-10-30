# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR graph identity across execution and reference file I/O

"""Exercise primitive, branch, nested and recurrent NIR file round trips."""

from dataclasses import fields
from pathlib import Path
from typing import Any
import json

import nir
import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_nir, to_nir
from sc_neurocore.nir_bridge.execution_metadata import EXECUTION_METADATA_KEY
from sc_neurocore.nir_bridge.graph_identity import GRAPH_ORDER_KEY


def _chain(node: Any, shape: tuple[int, ...]) -> Any:
    graph = nir.NIRGraph(
        nodes={
            "z_input": nir.Input(np.array(shape, dtype=int)),
            "a_processing": node,
            "m_output": nir.Output(np.asarray(next(iter(node.output_type.values())))),
        },
        edges=[("z_input", "a_processing"), ("a_processing", "m_output")],
        metadata={"experiment": "non-alphabetical scheduling", "revision": 7},
    )
    node.metadata["label"] = "independent primitive metadata"
    return graph


def _corpus() -> list[tuple[str, Any, tuple[int, ...]]]:
    ones = np.ones(4, dtype=np.float64)
    weights = np.arange(12, dtype=np.float64).reshape(3, 4) / 7 - 0.5
    corpus = [
        ("lif", nir.LIF(ones * 2, ones * 1.25, ones * 0.1, ones, ones * -0.2), (4,)),
        ("lif-matrix", nir.LIF(*(np.full((2, 2), v) for v in [2, 1.25, 0.1, 1, -0.2])), (2, 2)),
        ("if", nir.IF(ones * 1.5, ones * 0.7, ones * -0.1), (4,)),
        ("li", nir.LI(ones * 3, ones * 0.8, ones * 0.2), (4,)),
        ("i", nir.I(ones), (4,)),
        ("linear", nir.Linear(weights), (4,)),
        (
            "affine",
            nir.Affine(weights.astype(np.float32), np.arange(3, dtype=np.float32) / 10),
            (4,),
        ),
        ("scale", nir.Scale(np.arange(4, dtype=np.float32) + 0.5), (4,)),
        ("threshold", nir.Threshold(ones * 0.6), (4,)),
        ("delay", nir.Delay(np.array([0, 0.25, 0.5, 0.75])), (4,)),
        (
            "flatten",
            nir.Flatten(input_type={"input": np.array([2, 3, 4])}, start_dim=1, end_dim=-1),
            (2, 3, 4),
        ),
        (
            "cubalif",
            nir.CubaLIF(
                tau_syn=ones * 3,
                tau_mem=ones * 2,
                r=ones,
                v_leak=ones * 0.1,
                v_threshold=ones * 0.8,
                w_in=ones * 1.5,
            ),
            (4,),
        ),
        ("cubali", nir.CubaLI(ones * 3, ones * 2, ones, ones * 0.1, ones * 1.5), (4,)),
        (
            "conv1d",
            nir.Conv1d(
                7,
                np.arange(12, dtype=float).reshape(2, 2, 3) / 9,
                1,
                1,
                1,
                1,
                np.array([0.2, -0.1]),
            ),
            (2, 7),
        ),
        (
            "conv2d",
            nir.Conv2d(
                (5, 6),
                np.arange(16, dtype=float).reshape(2, 2, 2, 2) / 15,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
                np.array([0.2, -0.1]),
            ),
            (2, 5, 6),
        ),
    ]
    for kind in [nir.SumPool2d, nir.AvgPool2d]:
        pool = kind(kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0]))
        pool.input_type = {"input": np.array([1, 4, 6])}
        pool.output_type = {"output": np.array([1, 2, 3])}
        corpus.append((kind.__name__, pool, (1, 4, 6)))
    corpus.extend(
        [
            (
                "conv1d-singleton",
                nir.Conv1d(3, np.ones((1, 1, 3)), 1, 0, 1, 1, np.zeros(1)),
                (1, 3),
            ),
            (
                "conv2d-singleton",
                nir.Conv2d((2, 2), np.ones((1, 1, 2, 2)), 1, 0, 1, 1, np.zeros(1)),
                (1, 2, 2),
            ),
        ]
    )
    return corpus


def _assert_definition(expected: Any, actual: Any) -> None:
    assert type(actual) is type(expected)
    for field in fields(expected):
        a, b = getattr(expected, field.name), getattr(actual, field.name)
        if field.name == "metadata":
            ignored = {EXECUTION_METADATA_KEY, GRAPH_ORDER_KEY}
            assert {k: v for k, v in a.items() if k not in ignored} == {
                k: v for k, v in b.items() if k not in ignored
            }
        elif field.name == "nodes":
            assert list(a) == list(b)
            for name in a:
                _assert_definition(a[name], b[name])
        elif isinstance(a, tuple) and isinstance(b, np.ndarray):
            # Reference HDF5 represents structural tuples as integer arrays.
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype
        elif isinstance(a, dict):
            assert set(a) == set(b)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a == b


@pytest.mark.parametrize(
    "name,node,shape", _corpus(), ids=lambda x: x if isinstance(x, str) else None
)
@pytest.mark.parametrize("reset_mode", ["reset", "subtract"])
def test_primitive_definitions_and_runtime_survive_file(
    name: str, node: Any, shape: tuple[int, ...], reset_mode: str, tmp_path: Path
) -> None:
    """Preserve parameter arrays, tensor ranks and actual execution across HDF5."""
    source = _chain(node, shape)
    first = from_nir(source, dt=0.25, reset_mode=reset_mode)
    rng = np.random.default_rng(462)
    drive = rng.uniform(-0.3, 2.5, size=(24, *shape))
    reference = [first.step({"z_input": x})["m_output"].copy() for x in drive]
    expected_shape = tuple(int(v) for v in source.nodes["m_output"].output_type["output"])
    assert all(value.shape == expected_shape for value in reference)
    path = tmp_path / f"{name}.nir"
    exported = to_nir(first, path)
    _assert_definition(source, exported)
    restored = from_nir(path)
    _assert_definition(source, to_nir(restored))
    assert list(restored.nodes) == list(source.nodes)
    for x, expected in zip(drive, reference, strict=True):
        np.testing.assert_array_equal(restored.step({"z_input": x})["m_output"], expected)


def test_recurrent_branch_and_nested_graph_preserve_scheduling(tmp_path: Path) -> None:
    """Keep recurrent latency, summation order and nested topology through a real file."""
    inner = nir.NIRGraph(
        nodes={
            "z_input": nir.Input(np.array([4])),
            "b_cell": nir.CubaLIF(*(np.full(4, v) for v in [3, 2, 2, 0, 1, 3])),
            "a_recurrent": nir.Linear(np.eye(4) * 0.35),
            "c_skip": nir.Scale(np.full(4, 0.25)),
            "m_output": nir.Output(np.array([4])),
        },
        edges=[
            ("z_input", "b_cell"),
            ("b_cell", "a_recurrent"),
            ("a_recurrent", "b_cell"),
            ("z_input", "c_skip"),
            ("b_cell", "m_output"),
            ("c_skip", "m_output"),
        ],
    )
    source = _chain(inner, (4,))
    first = from_nir(source, dt=0.25, reset_mode="subtract")
    xs = np.random.default_rng(91).uniform(-0.2, 2.0, (64, 4))
    expected = [first.step({"z_input": x})["m_output"].copy() for x in xs]
    path = tmp_path / "recurrent.nir"
    _assert_definition(source, to_nir(first, path))
    second = from_nir(path)
    _assert_definition(source, to_nir(second))
    for x, y in zip(xs, expected, strict=True):
        np.testing.assert_array_equal(second.step({"z_input": x})["m_output"], y)


def test_changed_parameters_are_exported_instead_of_stale_definitions() -> None:
    """Export edited live parameters with their current precision and values."""
    source = _chain(nir.Linear(np.arange(12, dtype=np.float32).reshape(3, 4)), (4,))
    network = from_nir(source)
    network.nodes["a_processing"].weight = np.full((3, 4), 0.1234567890123)
    actual = to_nir(network).nodes["a_processing"].weight
    np.testing.assert_array_equal(actual, network.nodes["a_processing"].weight)
    assert actual.dtype == np.float64
    assert not np.array_equal(actual, source.nodes["a_processing"].weight)


@pytest.mark.parametrize(
    "contract",
    [
        5,
        "not JSON",
        "[]",
        "{}",
        '{"schema_version":"unknown","nodes":[],"edges":[]}',
        '{"schema_version":"sc-neurocore.nir-order.v1","nodes":[5],"edges":[]}',
        '{"schema_version":"sc-neurocore.nir-order.v1","nodes":[],"edges":[]}',
    ],
)
def test_malformed_or_stale_order_refused_at_public_import(contract: Any) -> None:
    """Reject an order declaration that is malformed or no longer names this graph."""
    graph = _chain(nir.Linear(np.eye(4)), (4,))
    graph.metadata[GRAPH_ORDER_KEY] = contract
    with pytest.raises(ValueError, match="graph order metadata"):
        from_nir(graph)


@pytest.mark.parametrize(
    "edges", [None, [["z_input"]], [[5, "m_output"]], [], [["m_output", "z_input"]]]
)
def test_bad_recorded_edges_refused(edges: Any) -> None:
    """Reject corrupted edge order before executing a differently scheduled graph."""
    graph = _chain(nir.Linear(np.eye(4)), (4,))
    graph.metadata[GRAPH_ORDER_KEY] = json.dumps(
        {
            "schema_version": "sc-neurocore.nir-order.v1",
            "nodes": list(graph.nodes),
            "edges": edges,
        }
    )
    with pytest.raises(ValueError, match="graph order metadata"):
        from_nir(graph)


def test_inconsistent_upstream_non_square_convolution_shape_is_refused() -> None:
    """Refuse a real upstream convolution declaration with contradictory tensor geometry."""
    node = nir.Conv2d(
        (5, 6),
        np.arange(24, dtype=float).reshape(2, 2, 2, 3) / 21,
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        np.zeros(2),
    )
    source = _chain(node, (2, 5, 6))
    with pytest.raises(ValueError, match="Conv2d output shape conflicts"):
        from_nir(source)


@pytest.mark.parametrize("shape", [(4, 3), (5, 5)])
def test_new_executable_graph_and_changed_matrix_geometry_are_exported(
    shape: tuple[int, int],
) -> None:
    """Export newly built graphs and honour an edited matrix's new port widths."""
    from sc_neurocore.nir_bridge import SCNetwork
    from sc_neurocore.nir_bridge.node_map import map_node

    graph = _chain(nir.Linear(np.arange(12, dtype=float).reshape(3, 4)), (4,))
    manual = SCNetwork(
        nodes={name: map_node(name, node) for name, node in graph.nodes.items()},
        edges=graph.edges.copy(),
        input_nodes=["z_input"],
        output_nodes=["m_output"],
    )
    exported = to_nir(manual)
    np.testing.assert_array_equal(
        exported.nodes["a_processing"].weight, graph.nodes["a_processing"].weight
    )
    edited = from_nir(graph)
    edited.nodes["a_processing"].weight = (
        np.arange(12, dtype=float).reshape(shape) if shape == (4, 3) else np.full(shape, 0.25)
    )
    edited.nodes["z_input"].shape = (shape[1],)
    edited.nodes["m_output"].shape = (shape[0],)
    exported = to_nir(edited)
    assert tuple(exported.nodes["z_input"].input_type["input"]) == (shape[1],)
    actual = from_nir(exported).step({"z_input": np.ones(shape[1])})["m_output"]
    np.testing.assert_array_equal(actual, edited.nodes["a_processing"].weight @ np.ones(shape[1]))
