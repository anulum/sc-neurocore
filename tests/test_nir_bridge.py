# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir
from sc_neurocore.nir_bridge.node_map import (
    SCLIFNode,
    SCIFNode,
    SCLINode,
    SCAffineNode,
    SCLinearNode,
    SCScaleNode,
    SCThresholdNode,
    SCFlattenNode,
    SCIntegratorNode,
    SCInputNode,
    SCOutputNode,
    map_node,
)
from sc_neurocore.nir_bridge.parser import SCNetwork, SCSubgraphNode


@pytest.fixture
def local_tmp_path():
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _make_lif_affine_graph(n_in=3, n_out=2):
    """Build a minimal NIR graph: Input → Affine → LIF → Output."""
    nodes = {
        "input": nir.Input(input_type={"input": np.array([n_in])}),
        "affine": nir.Affine(
            weight=np.random.RandomState(42).randn(n_out, n_in).astype(np.float32),
            bias=np.zeros(n_out, dtype=np.float32),
        ),
        "lif": nir.LIF(
            tau=np.full(n_out, 20.0),
            r=np.ones(n_out),
            v_leak=np.zeros(n_out),
            v_threshold=np.ones(n_out),
        ),
        "output": nir.Output(output_type={"output": np.array([n_out])}),
    }
    edges = [("input", "affine"), ("affine", "lif"), ("lif", "output")]
    return nir.NIRGraph(nodes=nodes, edges=edges)


# --- Node mapping tests ---


class TestNodeMapping:
    def test_map_input(self):
        node = nir.Input(input_type={"input": np.array([4])})
        sc = map_node("inp", node)
        assert isinstance(sc, SCInputNode)
        assert sc.shape == (4,)

    def test_map_output(self):
        node = nir.Output(output_type={"output": np.array([3])})
        sc = map_node("out", node)
        assert isinstance(sc, SCOutputNode)
        assert sc.shape == (3,)

    def test_map_lif(self):
        node = nir.LIF(
            tau=np.array([10.0, 20.0]),
            r=np.array([1.0, 1.0]),
            v_leak=np.array([0.0, 0.0]),
            v_threshold=np.array([1.0, 1.0]),
        )
        sc = map_node("lif", node)
        assert isinstance(sc, SCLIFNode)
        assert len(sc.neurons) == 2
        assert sc.neurons[0].tau_mem == 10.0
        assert sc.neurons[1].tau_mem == 20.0

    def test_map_if(self):
        node = nir.IF(
            r=np.array([1.0]),
            v_threshold=np.array([0.5]),
        )
        sc = map_node("if", node)
        assert isinstance(sc, SCIFNode)
        assert sc.n_neurons == 1
        sc.forward(np.array([1.0]))
        sc.reset()
        np.testing.assert_allclose(sc.v, [0.0])

    def test_map_li(self):
        node = nir.LI(
            tau=np.array([15.0]),
            r=np.array([1.0]),
            v_leak=np.array([-0.5]),
        )
        sc = map_node("li", node, dt=0.5)
        assert isinstance(sc, SCLINode)
        assert sc.dt == 0.5
        sc.forward(np.array([1.0]))
        sc.reset()
        np.testing.assert_allclose(sc.v, [-0.5])

    def test_map_affine(self):
        node = nir.Affine(
            weight=np.eye(3, dtype=np.float32),
            bias=np.ones(3, dtype=np.float32),
        )
        sc = map_node("aff", node)
        assert isinstance(sc, SCAffineNode)
        out = sc.forward(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [2.0, 3.0, 4.0])

    def test_map_linear(self):
        node = nir.Linear(weight=np.eye(2, dtype=np.float32))
        sc = map_node("lin", node)
        assert isinstance(sc, SCLinearNode)
        out = sc.forward(np.array([5.0, 7.0]))
        np.testing.assert_allclose(out, [5.0, 7.0])

    def test_map_scale(self):
        node = nir.Scale(scale=np.array([2.0, 0.5]))
        sc = map_node("scl", node)
        assert isinstance(sc, SCScaleNode)
        out = sc.forward(np.array([3.0, 4.0]))
        np.testing.assert_allclose(out, [6.0, 2.0])

    def test_map_threshold(self):
        node = nir.Threshold(threshold=np.array([0.5]))
        sc = map_node("thr", node)
        assert isinstance(sc, SCThresholdNode)
        assert sc.forward(np.array([0.6]))[0] == 1.0
        assert sc.forward(np.array([0.3]))[0] == 0.0

    def test_map_flatten(self):
        node = nir.Flatten(start_dim=0, end_dim=-1)
        sc = map_node("flat", node)
        assert isinstance(sc, SCFlattenNode)
        out = sc.forward(np.arange(6).reshape(2, 3))
        np.testing.assert_array_equal(out, np.arange(6))

    def test_map_integrator(self):
        node = nir.I(r=np.array([1.0]))
        sc = map_node("integ", node)
        assert isinstance(sc, SCIntegratorNode)
        sc.forward(np.array([1.0]))
        sc.forward(np.array([1.0]))
        np.testing.assert_allclose(sc.v, [2.0])

    def test_unsupported_raises(self):
        node = nir.Delay(delay=np.array([1.0]))
        with pytest.raises(NotImplementedError, match="Delay"):
            map_node("dly", node)


# --- Graph parsing tests ---


class TestGraphParsing:
    def test_parse_lif_affine(self):
        graph = _make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        assert isinstance(net, SCNetwork)
        assert len(net.nodes) == 4
        assert len(net.edges) == 3
        assert "input" in net.input_nodes
        assert "output" in net.output_nodes

    def test_topological_order(self):
        graph = _make_lif_affine_graph()
        net = from_nir(graph)
        order = net.topo_order
        assert order.index("input") < order.index("affine")
        assert order.index("affine") < order.index("lif")
        assert order.index("lif") < order.index("output")

    def test_summary(self):
        graph = _make_lif_affine_graph()
        net = from_nir(graph)
        s = net.summary()
        assert "SCNetwork" in s
        assert "SCLIFNode" in s
        assert "SCAffineNode" in s

    def test_cycle_raises_value_error(self):
        net = SCNetwork(nodes={"a": object(), "b": object()}, edges=[("a", "b"), ("b", "a")])
        with pytest.raises(ValueError, match="cycle"):
            _ = net.topo_order

    def test_nested_subgraph_requires_single_io(self):
        inner = SCNetwork(
            nodes={
                "input_a": SCInputNode(name="input_a", shape=(1,)),
                "input_b": SCInputNode(name="input_b", shape=(1,)),
                "output": SCOutputNode(name="output", shape=(1,)),
            },
            edges=[("input_a", "output")],
            input_nodes=["input_a", "input_b"],
            output_nodes=["output"],
        )
        with pytest.raises(ValueError, match="exactly one input and one output"):
            SCSubgraphNode(name="subgraph", network=inner)


# --- Execution tests ---


class TestExecution:
    def test_single_step(self):
        graph = _make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        out = net.step({"input": np.array([1.0, 0.5, 0.2])})
        assert "output" in out
        assert out["output"].shape == (2,)

    def test_run_multiple_steps(self):
        graph = _make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        results = net.run({"input": np.array([2.0, 1.0, 0.5])}, steps=50)
        assert "output" in results
        assert len(results["output"]) == 50
        spikes = sum(r.sum() for r in results["output"])
        assert spikes > 0

    def test_reset(self):
        graph = _make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        net.run({"input": np.array([2.0, 1.0, 0.5])}, steps=10)
        net.reset()
        lif_node = net.nodes["lif"]
        for neuron in lif_node.neurons:
            assert neuron.v == neuron.v_rest

    def test_if_neuron_fires(self):
        nodes = {
            "input": nir.Input(input_type={"input": np.array([1])}),
            "if": nir.IF(r=np.array([1.0]), v_threshold=np.array([3.0])),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("input", "if"), ("if", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        # 3 steps of current=1.0 → v reaches 3.0 → spike
        results = net.run({"input": np.array([1.0])}, steps=5)
        spikes = [r[0] for r in results["output"]]
        assert 1.0 in spikes

    def test_li_leaks(self):
        nodes = {
            "input": nir.Input(input_type={"input": np.array([1])}),
            "li": nir.LI(tau=np.array([10.0]), r=np.array([1.0]), v_leak=np.array([0.0])),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("input", "li"), ("li", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        # Drive with constant input, check it converges (doesn't explode)
        results = net.run({"input": np.array([1.0])}, steps=200)
        final = results["output"][-1][0]
        assert np.isfinite(final)
        # Steady state: v_leak + R*I = 0 + 1*1 = 1.0 (for large tau, slow approach)
        assert 0 < final < 1.5

    def test_linear_chain(self):
        """Input → Linear → Scale → Threshold → Output"""
        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "linear": nir.Linear(weight=np.array([[1.0, 0.0], [0.0, 1.0]])),
            "scale": nir.Scale(scale=np.array([2.0, 2.0])),
            "threshold": nir.Threshold(threshold=np.array([1.5, 1.5])),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [
            ("input", "linear"),
            ("linear", "scale"),
            ("scale", "threshold"),
            ("threshold", "output"),
        ]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        # [1.0, 0.5] → linear: [1.0, 0.5] → scale: [2.0, 1.0] → threshold: [1, 0]
        out = net.step({"input": np.array([1.0, 0.5])})
        np.testing.assert_allclose(out["output"], [1.0, 0.0])

    def test_fan_in_sums_predecessors(self):
        nodes = {
            "left": nir.Input(input_type={"input": np.array([1])}),
            "right": nir.Input(input_type={"input": np.array([1])}),
            "scale": nir.Scale(scale=np.array([1.0])),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("left", "scale"), ("right", "scale"), ("scale", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        out = net.step({"left": np.array([2.0]), "right": np.array([3.0])})
        np.testing.assert_allclose(out["output"], [5.0])

    def test_node_without_predecessor_uses_zero_input(self):
        net = SCNetwork(
            nodes={
                "orphan": SCScaleNode(name="orphan", scale=np.array([3.0])),
                "output": SCOutputNode(name="output", shape=(1,)),
            },
            edges=[("orphan", "output")],
            output_nodes=["output"],
        )

        out = net.step({})
        np.testing.assert_allclose(out["output"], [0.0])

    def test_nested_subgraph_executes_and_resets(self):
        inner = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([1])}),
                "integrator": nir.I(r=np.array([1.0])),
                "output": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "integrator"), ("integrator", "output")],
        )
        outer = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([1])}),
                "subgraph": inner,
                "output": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "subgraph"), ("subgraph", "output")],
        )
        net = from_nir(outer)

        first = net.step({"input": np.array([1.0])})
        second = net.step({"input": np.array([1.0])})
        np.testing.assert_allclose(first["output"], [1.0])
        np.testing.assert_allclose(second["output"], [2.0])

        net.reset()
        reset_first = net.step({"input": np.array([1.0])})
        np.testing.assert_allclose(reset_first["output"], [1.0])

    def test_flatten_respects_dim_range(self):
        node = map_node("flat", nir.Flatten(start_dim=1, end_dim=1))
        out = node.forward(np.arange(24).reshape(2, 3, 4))
        assert out.shape == (2, 3, 4)

        node = map_node("flat_all", nir.Flatten(start_dim=1, end_dim=-1))
        out = node.forward(np.arange(24).reshape(2, 3, 4))
        assert out.shape == (2, 12)

    def test_flatten_invalid_dims_raise(self):
        node = map_node("flat", nir.Flatten(start_dim=2, end_dim=1))
        with pytest.raises(ValueError, match="Invalid flatten dims"):
            node.forward(np.arange(6).reshape(2, 3))

    def test_flatten_scalar_input(self):
        node = map_node("flat", nir.Flatten(start_dim=0, end_dim=-1))
        out = node.forward(np.array(3.0))
        np.testing.assert_allclose(out, [3.0])

    def test_flatten_scalar_invalid_dims_raise(self):
        node = map_node("flat", nir.Flatten(start_dim=1, end_dim=1))
        with pytest.raises(ValueError, match="Invalid flatten dims"):
            node.forward(np.array(3.0))


class TestFileIO:
    def test_from_nir_file(self, local_tmp_path):
        graph = _make_lif_affine_graph()
        path = local_tmp_path / "test_model.nir"
        nir.write(str(path), graph)

        net = from_nir(str(path))
        assert len(net.nodes) == 4

    def test_from_nir_path_object(self, local_tmp_path):
        graph = _make_lif_affine_graph()
        path = local_tmp_path / "test_model.nir"
        nir.write(str(path), graph)

        net = from_nir(path)
        assert "lif" in net.nodes

    def test_invalid_source_raises(self):
        with pytest.raises(TypeError, match="Expected NIRGraph"):
            from_nir(42)


class TestExport:
    def test_to_nir_not_implemented(self):
        from sc_neurocore.nir_bridge import to_nir

        with pytest.raises(NotImplementedError, match="Phase 3"):
            to_nir(None)
