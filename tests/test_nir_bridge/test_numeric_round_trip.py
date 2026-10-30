# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Numeric NIR round trip through a real file

"""An exported network, read back from its ``.nir`` file, computes the same thing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nir
import numpy as np
import pytest

from sc_neurocore.nir_bridge import from_nir, to_nir

_STEPS = 40


def _feedforward() -> Any:
    rng = np.random.default_rng(7)
    return nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type={"input": np.array([3])}),
            "affine": nir.Affine(
                weight=rng.uniform(0.2, 1.0, size=(2, 3)), bias=rng.uniform(0.0, 0.2, size=2)
            ),
            "lif": nir.LIF(
                tau=np.array([4.0, 9.0]),
                r=np.array([1.0, 1.5]),
                v_leak=np.array([0.0, 0.1]),
                v_threshold=np.array([0.8, 1.2]),
                v_reset=np.array([0.0, -0.2]),
            ),
            "out": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("in", "affine"), ("affine", "lif"), ("lif", "out")],
    )


def _current_based_with_delay_and_recurrence() -> Any:
    return nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type={"input": np.array([2])}),
            "linear": nir.Linear(weight=np.array([[0.9, -0.3], [0.2, 0.7]])),
            "delay": nir.Delay(delay=np.array([2.0, 3.0])),
            "cuba": nir.CubaLIF(
                tau_syn=np.array([2.0, 3.0]),
                tau_mem=np.array([5.0, 6.0]),
                r=np.array([1.0, 1.0]),
                v_leak=np.array([0.0, 0.0]),
                v_threshold=np.array([0.5, 0.6]),
                v_reset=np.array([0.0, 0.0]),
                w_in=np.array([1.0, 1.0]),
            ),
            "scale": nir.Scale(scale=np.array([0.5, 0.5])),
            "out": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("in", "linear"),
            ("linear", "delay"),
            ("delay", "cuba"),
            ("cuba", "scale"),
            ("scale", "cuba"),
            ("cuba", "out"),
        ],
        type_check=False,
    )


def _drive(width: int) -> list[np.ndarray[Any, Any]]:
    rng = np.random.default_rng(11)
    return [rng.uniform(0.0, 3.0, size=width) for _ in range(_STEPS)]


def _trace(network: Any, drive: list[np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
    (source,) = network.input_nodes
    (sink,) = network.output_nodes
    return [network.step({source: value})[sink].copy() for value in drive]


@pytest.mark.parametrize(
    ("build", "width", "dt"),
    [(_feedforward, 3, 1.0), (_current_based_with_delay_and_recurrence, 2, 0.5)],
    ids=["feedforward-lif", "cuba-delay-recurrent"],
)
def test_a_network_read_back_from_its_nir_file_computes_identically(
    tmp_path: Path, build: Any, width: int, dt: float
) -> None:
    """Preserve outputs, shapes and parameters after a real file round trip."""
    original = from_nir(build(), dt=dt)
    path = tmp_path / "round_trip.nir"
    to_nir(original, path)
    restored = from_nir(path, dt=dt)

    assert restored.nir_version == nir.version
    # Both networks realise recurrent edges with their internal delay nodes.
    assert restored.topo_order == original.topo_order
    assert set(restored.nodes) == set(original.nodes)
    for name, node in original.nodes.items():
        for field_name, value in vars(node).items():
            if isinstance(value, np.ndarray) and not field_name.startswith("_"):
                np.testing.assert_array_equal(getattr(restored.nodes[name], field_name), value)

    drive = _drive(width)
    expected = _trace(original, drive)
    actual = _trace(restored, drive)
    assert [trace.shape for trace in actual] == [trace.shape for trace in expected]
    for step, (left, right) in enumerate(zip(expected, actual, strict=True)):
        np.testing.assert_array_equal(right, left, err_msg=f"step {step}")
    # A silent network would agree trivially; both cases spike repeatedly.
    assert sum(int(np.count_nonzero(trace)) for trace in expected) >= 5


@pytest.mark.parametrize(
    ("build", "width", "dt"),
    [(_feedforward, 3, 1.0), (_current_based_with_delay_and_recurrence, 2, 0.5)],
    ids=["feedforward-lif", "cuba-delay-recurrent"],
)
def test_import_owns_parameters_and_metadata_after_source_mutation(
    tmp_path: Path, build: Any, width: int, dt: float
) -> None:
    """Keep imported and re-exported dynamics independent of caller-owned arrays."""
    graph = build()
    graph.metadata = {"note": {"tag": "original"}}
    reference = from_nir(graph, dt=dt)
    network = from_nir(graph, dt=dt)

    for node in graph.nodes.values():
        for value in vars(node).values():
            if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
                value[...] += 7
    graph.metadata["note"]["tag"] = "changed"

    drive = _drive(width)
    expected = _trace(reference, drive)
    for actual, wanted in zip(_trace(network, drive), expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    assert sum(int(np.count_nonzero(trace)) for trace in expected) >= 5

    path = tmp_path / "owned_parameters.nir"
    to_nir(network, path)
    restored = from_nir(path, dt=dt)
    assert restored.nir_metadata["note"]["tag"] == "original"
    for actual, wanted in zip(_trace(restored, drive), expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_import_owns_mutable_convolution_geometry_and_export_definition(tmp_path: Path) -> None:
    """Isolate mutable convolution shapes, weights and retained export definitions."""
    shape = np.array([3, 3])
    conv = nir.Conv2d(
        input_shape=shape,
        weight=np.ones((1, 1, 2, 2)),
        bias=np.zeros(1),
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    graph = nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type=conv.input_type),
            "conv": conv,
            "out": nir.Output(output_type=conv.output_type),
        },
        edges=[("in", "conv"), ("conv", "out")],
    )
    network = from_nir(graph)
    conv.input_shape[...] = 99
    conv.weight[...] = 19
    conv.bias[...] = -3
    mapped = network.nodes["conv"]
    np.testing.assert_array_equal(mapped.input_shape, [3, 3])
    np.testing.assert_array_equal(network.step({"in": np.ones(9)})["out"], np.full((1, 2, 2), 4.0))

    mapped.weight[...] = 2
    np.testing.assert_array_equal(conv.weight, np.full((1, 1, 2, 2), 19.0))
    np.testing.assert_array_equal(
        network.nir_node_definitions["conv"].weight, np.ones((1, 1, 2, 2))
    )
    path = tmp_path / "owned_convolution.nir"
    to_nir(network, path)
    restored = from_nir(path)
    np.testing.assert_array_equal(restored.step({"in": np.ones(9)})["out"], np.full((1, 2, 2), 8.0))


def _convolution_fixture(
    dimensions: int,
) -> tuple[Any, Any, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build reference convolutions with caller-owned NumPy geometry."""
    weights: np.ndarray[Any, np.dtype[np.float64]]
    inputs: np.ndarray[Any, np.dtype[np.float64]]
    if dimensions == 1:
        weights = np.arange(1.0, 13.0).reshape(2, 2, 3)
        inputs = np.arange(14.0).reshape(2, 7)
        conv = nir.Conv1d(
            input_shape=7,
            weight=weights,
            bias=np.array([0.25, -0.5]),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        expected = (
            np.stack(
                [
                    sum(np.correlate(inputs[c], weights[o, c], mode="valid") for c in range(2))
                    for o in range(2)
                ]
            )
            + conv.bias[:, None]
        )
    else:
        weights = np.arange(1.0, 17.0).reshape(2, 2, 2, 2)
        inputs = np.arange(18.0).reshape(2, 3, 3)
        conv = nir.Conv2d(
            input_shape=(3, 3),
            weight=weights,
            bias=np.array([0.25, -0.5]),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        windows = np.lib.stride_tricks.sliding_window_view(inputs, (2, 2), axis=(1, 2))
        expected = np.einsum("ockl,cijkl->oij", weights, windows) + conv.bias[:, None, None]
    graph = nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type=conv.input_type),
            "conv": conv,
            "out": nir.Output(output_type=conv.output_type),
        },
        edges=[("in", "conv"), ("conv", "out")],
    )
    # Admit the real reference graph with equivalent mutable NumPy geometry.
    # NIR's constructor already established its input and output edge types.
    for field in ("stride", "padding", "dilation", "groups", "input_shape"):
        setattr(conv, field, np.asarray(getattr(conv, field)).copy())
    return graph, conv, inputs, expected


@pytest.mark.parametrize("dimensions", [1, 2], ids=["conv1d", "conv2d"])
def test_mutable_source_convolution_geometry_cannot_change_execution_or_file(
    tmp_path: Path,
    dimensions: int,
) -> None:
    """Own every geometry field across import, source mutation and real file execution."""
    graph, conv, inputs, expected = _convolution_fixture(dimensions)
    network = from_nir(graph)
    np.testing.assert_array_equal(network.step({"in": inputs})["out"], expected)
    for field in ("stride", "padding", "dilation", "groups", "input_shape"):
        getattr(conv, field)[...] = 2 if field == "groups" else 99
    np.testing.assert_array_equal(network.step({"in": inputs})["out"], expected)
    path = tmp_path / f"owned_geometry_{dimensions}d.nir"
    to_nir(network, path)
    restored = from_nir(path)
    np.testing.assert_array_equal(restored.step({"in": inputs})["out"], expected)
    for imported in (network, restored):
        mapped = imported.nodes["conv"]
        for field in ("stride", "padding", "dilation", "groups", "input_shape"):
            value = getattr(mapped, field)
            assert type(value) in (int, tuple)
            if isinstance(value, tuple):
                assert all(type(item) is int for item in value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("groups", True),
        ("groups", 1.5),
        ("groups", 0),
        ("groups", np.array([1, 2])),
        ("stride", 0),
        ("dilation", 0),
        ("padding", -1),
        ("input_shape", 0),
    ],
)
def test_convolution_geometry_refuses_noninteger_or_invalid_extents_at_import(
    field: str,
    value: Any,
) -> None:
    """Refuse lossy scalar coercion and invalid geometry before real graph execution."""
    graph, conv, _, _ = _convolution_fixture(1)
    setattr(conv, field, value)
    with pytest.raises(ValueError, match=f"Convolution {field}"):
        from_nir(graph)
