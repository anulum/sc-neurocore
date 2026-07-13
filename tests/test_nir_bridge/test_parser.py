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
    SCInputNode,
    SCOutputNode,
)
from sc_neurocore.nir_bridge.parser import (
    SCMultiPortSubgraphNode,
    SCNetwork,
    SCSubgraphNode,
    _UnitDelayNode,
)

from tests.test_nir_bridge.support import make_lif_affine_graph


class TestGraphParsing:
    def test_parse_lif_affine(self) -> None:
        graph = make_lif_affine_graph(n_in=3, n_out=2)
        net = from_nir(graph)
        assert isinstance(net, SCNetwork)
        assert len(net.nodes) == 4
        assert len(net.edges) == 3
        assert "input" in net.input_nodes
        assert "output" in net.output_nodes

    def test_topological_order(self) -> None:
        graph = make_lif_affine_graph()
        net = from_nir(graph)
        order = net.topo_order
        assert order.index("input") < order.index("affine")
        assert order.index("affine") < order.index("lif")
        assert order.index("lif") < order.index("output")

    def test_summary(self) -> None:
        graph = make_lif_affine_graph()
        net = from_nir(graph)
        s = net.summary()
        assert "SCNetwork" in s
        assert "SCLIFNode" in s
        assert "SCAffineNode" in s

    def test_cycle_inserts_delay_node(self) -> None:
        """Recurrent edges should be broken by implicit delay nodes."""
        node_a = SCInputNode(name="a", shape=(1,))
        node_b = SCOutputNode(name="b", shape=(1,))
        net = SCNetwork(
            nodes={"a": node_a, "b": node_b},
            edges=[("a", "b"), ("b", "a")],
            input_nodes=["a"],
            output_nodes=["b"],
        )
        order = net.topo_order
        assert len(order) == 3  # a, b, + 1 delay node
        delay_names = [n for n in order if n.startswith("_delay_")]
        assert len(delay_names) == 1
        assert isinstance(net.nodes[delay_names[0]], _UnitDelayNode)

    def test_recurrent_network_runs(self) -> None:
        """A network with feedback via scale node executes across timesteps."""
        # input → scale → output, with scale → scale self-recurrence
        input_node = SCInputNode(name="input", shape=(1,))
        scale_node = SCScaleNode(name="scale", scale=np.array([0.5]))
        output_node = SCOutputNode(name="output", shape=(1,))
        net = SCNetwork(
            nodes={"input": input_node, "scale": scale_node, "output": output_node},
            edges=[("input", "scale"), ("scale", "output"), ("scale", "scale")],
            input_nodes=["input"],
            output_nodes=["output"],
        )
        results = net.run({"input": np.array([1.0])}, steps=5)
        assert "output" in results
        assert len(results["output"]) == 5
        # Step 0: scale gets input=1.0, delay feedback=0.0 → output = 0.5*1.0 = 0.5
        np.testing.assert_allclose(results["output"][0], [0.5], atol=1e-10)
        # Step 1: scale gets input=1.0 + delay=0.5 → output = 0.5*1.5 = 0.75
        np.testing.assert_allclose(results["output"][1], [0.75], atol=1e-10)

    def test_recurrent_reset_clears_delay(self) -> None:
        """Reset should clear delay node buffers."""
        scale_node = SCScaleNode(name="s", scale=np.array([1.0]))
        net = SCNetwork(
            nodes={"s": scale_node},
            edges=[("s", "s")],
            input_nodes=[],
            output_nodes=[],
        )
        _ = net.topo_order  # trigger delay insertion
        delay_names = [n for n in net.nodes if n.startswith("_delay_")]
        assert len(delay_names) == 1
        # Force a buffer value
        net.nodes[delay_names[0]].update_buffer(np.array([5.0]))
        net.reset()
        assert net.nodes[delay_names[0]]._buffer is None

    def test_summary_shows_recurrent(self) -> None:
        """Summary should mention recurrent connections."""
        scale_node = SCScaleNode(name="s", scale=np.array([1.0]))
        net = SCNetwork(
            nodes={"s": scale_node},
            edges=[("s", "s")],
            input_nodes=[],
            output_nodes=[],
        )
        _ = net.topo_order
        s = net.summary()
        assert "recurrent" in s

    def test_nested_subgraph_requires_single_io(self) -> None:
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

    def test_multi_port_subgraph_creation(self) -> None:
        """Multi-port subgraph should accept multiple I/O."""
        inner = SCNetwork(
            nodes={
                "in_a": SCInputNode(name="in_a", shape=(1,)),
                "in_b": SCInputNode(name="in_b", shape=(1,)),
                "out": SCOutputNode(name="out", shape=(1,)),
            },
            edges=[("in_a", "out"), ("in_b", "out")],
            input_nodes=["in_a", "in_b"],
            output_nodes=["out"],
        )
        sub = SCMultiPortSubgraphNode(name="multi", network=inner)
        assert sub.input_ports == ["in_a", "in_b"]
        assert sub.output_ports == ["out"]

    def test_multi_port_forward_single(self) -> None:
        """Single-input convenience should work."""
        inner = SCNetwork(
            nodes={
                "in_a": SCInputNode(name="in_a", shape=(1,)),
                "in_b": SCInputNode(name="in_b", shape=(1,)),
                "out": SCOutputNode(name="out", shape=(1,)),
            },
            edges=[("in_a", "out"), ("in_b", "out")],
            input_nodes=["in_a", "in_b"],
            output_nodes=["out"],
        )
        sub = SCMultiPortSubgraphNode(name="multi", network=inner)
        result = sub.forward(np.array([1.0]))
        assert result.shape == (1,)

    def test_multi_port_forward_multi(self) -> None:
        """Multi-input forward should return named outputs."""
        inner = SCNetwork(
            nodes={
                "in_a": SCInputNode(name="in_a", shape=(1,)),
                "in_b": SCInputNode(name="in_b", shape=(1,)),
                "out": SCOutputNode(name="out", shape=(1,)),
            },
            edges=[("in_a", "out"), ("in_b", "out")],
            input_nodes=["in_a", "in_b"],
            output_nodes=["out"],
        )
        sub = SCMultiPortSubgraphNode(name="multi", network=inner)
        result = sub.forward_multi({"in_a": np.array([1.0]), "in_b": np.array([2.0])})
        assert "out" in result
        np.testing.assert_allclose(result["out"], [3.0])

    def test_multi_port_requires_io(self) -> None:
        """Empty I/O should raise."""
        inner = SCNetwork(nodes={}, edges=[], input_nodes=[], output_nodes=[])
        with pytest.raises(ValueError, match="at least one input"):
            SCMultiPortSubgraphNode(name="empty", network=inner)


# --- Execution tests ---
