# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir
from sc_neurocore.nir_bridge.node_map import (
    SCScaleNode,
    SCOutputNode,
    map_node,
)
from sc_neurocore.nir_bridge.parser import (
    SCNetwork,
)

from tests.test_nir_bridge.support import make_lif_affine_graph


class TestExecution:
    def test_single_step(self) -> None:
        graph = make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        out = net.step({"input": np.array([1.0, 0.5, 0.2])})
        assert "output" in out
        assert out["output"].shape == (2,)

    def test_run_multiple_steps(self) -> None:
        graph = make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        results = net.run({"input": np.array([2.0, 1.0, 0.5])}, steps=50)
        assert "output" in results
        assert len(results["output"]) == 50
        spikes = sum(r.sum() for r in results["output"])
        assert spikes > 0

    def test_reset(self) -> None:
        graph = make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        net.run({"input": np.array([2.0, 1.0, 0.5])}, steps=10)
        net.reset()
        lif_node = net.nodes["lif"]
        np.testing.assert_allclose(lif_node.v, lif_node.v_leak)

    def test_if_neuron_fires(self) -> None:
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

    def test_li_leaks(self) -> None:
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

    def test_linear_chain(self) -> None:
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

    def test_fan_in_sums_predecessors(self) -> None:
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

    def test_node_without_predecessor_uses_zero_input(self) -> None:
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

    def test_nested_subgraph_executes_and_resets(self) -> None:
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

    def test_flatten_respects_dim_range(self) -> None:
        node = map_node("flat", nir.Flatten(start_dim=1, end_dim=1))
        out = node.forward(np.arange(24).reshape(2, 3, 4))
        assert out.shape == (2, 3, 4)

        node = map_node("flat_all", nir.Flatten(start_dim=1, end_dim=-1))
        out = node.forward(np.arange(24).reshape(2, 3, 4))
        assert out.shape == (2, 12)

    def test_flatten_invalid_dims_raise(self) -> None:
        node = map_node("flat", nir.Flatten(start_dim=2, end_dim=1))
        with pytest.raises(ValueError, match="Invalid flatten dims"):
            node.forward(np.arange(6).reshape(2, 3))

    def test_flatten_scalar_input(self) -> None:
        node = map_node("flat", nir.Flatten(start_dim=0, end_dim=-1))
        out = node.forward(np.array(3.0))
        np.testing.assert_allclose(out, [3.0])

    def test_flatten_scalar_invalid_dims_raise(self) -> None:
        node = map_node("flat", nir.Flatten(start_dim=1, end_dim=1))
        with pytest.raises(ValueError, match="Invalid flatten dims"):
            node.forward(np.array(3.0))
