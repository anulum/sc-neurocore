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
    SCDelayNode,
    SCCubaLIFNode,
    SCCubaLINode,
    SCSumPool2dNode,
    SCAvgPool2dNode,
    SCConv1dNode,
    SCConv2dNode,
    SCOutputNode,
    map_node,
)
from sc_neurocore.nir_bridge.parser import (
    SCMultiPortSubgraphNode,
    SCNetwork,
    SCSubgraphNode,
    _UnitDelayNode,
)


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
        assert sc.n_neurons == 2
        assert sc.tau[0] == 10.0
        assert sc.tau[1] == 20.0

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

    def test_map_delay(self):
        node = nir.Delay(delay=np.array([2.0]))
        sc = map_node("dly", node, dt=1.0)
        assert isinstance(sc, SCDelayNode)
        # Feed 1.0, expect 0.0 for 2 steps (delay=2), then 1.0
        assert sc.forward(np.array([1.0]))[0] == 0.0
        assert sc.forward(np.array([2.0]))[0] == 0.0
        assert sc.forward(np.array([3.0]))[0] == 1.0
        sc.reset()
        assert sc.forward(np.array([5.0]))[0] == 0.0

    def test_map_cuba_lif(self):
        node = nir.CubaLIF(
            tau_syn=np.array([5.0]),
            tau_mem=np.array([20.0]),
            r=np.ones(1),
            v_leak=np.zeros(1),
            v_threshold=np.ones(1),
            w_in=np.ones(1),
        )
        sc = map_node("cuba_lif", node, dt=1.0)
        assert isinstance(sc, SCCubaLIFNode)
        spikes = sum(float(sc.forward(np.array([2.0]))[0]) for _ in range(100))
        assert spikes > 0
        sc.reset()
        np.testing.assert_allclose(sc.i_syn, [0.0])

    def test_map_cuba_li(self):
        node = nir.CubaLI(
            tau_syn=np.array([5.0]),
            tau_mem=np.array([20.0]),
            r=np.ones(1),
            v_leak=np.zeros(1),
            w_in=np.ones(1),
        )
        sc = map_node("cuba_li", node, dt=1.0)
        assert isinstance(sc, SCCubaLINode)
        for _ in range(50):
            out = sc.forward(np.array([1.0]))
        assert np.isfinite(out[0])
        assert out[0] > 0
        sc.reset()
        np.testing.assert_allclose(sc.v, [0.0])

    def test_map_sum_pool2d(self):
        node = nir.SumPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        )
        sc = map_node("spool", node)
        assert isinstance(sc, SCSumPool2dNode)
        x = np.ones((1, 4, 4))
        out = sc.forward(x)
        np.testing.assert_allclose(out, np.full((1, 2, 2), 4.0).squeeze())

    def test_map_avg_pool2d(self):
        node = nir.AvgPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        )
        sc = map_node("apool", node)
        assert isinstance(sc, SCAvgPool2dNode)
        x = np.ones((1, 4, 4))
        out = sc.forward(x)
        np.testing.assert_allclose(out, np.ones((1, 2, 2)).squeeze())

    def test_map_conv1d(self):
        weight = np.ones((1, 1, 3), dtype=np.float32)
        node = nir.Conv1d(
            input_shape=5,
            weight=weight,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=np.zeros(1, dtype=np.float32),
        )
        sc = map_node("conv1d", node)
        assert isinstance(sc, SCConv1dNode)
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out = sc.forward(x)
        np.testing.assert_allclose(out, [6.0, 9.0, 12.0])

    def test_map_conv2d(self):
        weight = np.ones((1, 1, 2, 2), dtype=np.float32)
        node = nir.Conv2d(
            input_shape=(3, 3),
            weight=weight,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=np.zeros(1, dtype=np.float32),
        )
        sc = map_node("conv2d", node)
        assert isinstance(sc, SCConv2dNode)
        x = np.ones((1, 3, 3))
        out = sc.forward(x)
        np.testing.assert_allclose(out, np.full((2, 2), 4.0))

    def test_all_18_primitives_mapped(self):
        """Verify all 18 NIR primitives have entries in NODE_MAP."""
        from sc_neurocore.nir_bridge.node_map import NODE_MAP

        expected = {
            nir.Input,
            nir.Output,
            nir.LIF,
            nir.IF,
            nir.LI,
            nir.I,
            nir.Affine,
            nir.Linear,
            nir.Scale,
            nir.Threshold,
            nir.Flatten,
            nir.Delay,
            nir.CubaLIF,
            nir.CubaLI,
            nir.SumPool2d,
            nir.AvgPool2d,
            nir.Conv1d,
            nir.Conv2d,
        }
        assert set(NODE_MAP.keys()) == expected, (
            f"Missing: {expected - set(NODE_MAP.keys())}, Extra: {set(NODE_MAP.keys()) - expected}"
        )


# --- Scalar param broadcast tests ---


class TestScalarBroadcast:
    """Verify neuron nodes with scalar params auto-broadcast to input size."""

    def test_lif_scalar_broadcast(self):
        node = nir.LIF(
            tau=np.array([10.0]),
            r=np.array([1.0]),
            v_leak=np.array([0.0]),
            v_threshold=np.array([1.0]),
        )
        sc = map_node("lif", node, dt=1.0)
        assert sc.n_neurons == 1
        out = sc.forward(np.array([0.5, 0.3, 0.1, 0.0]))
        assert sc.n_neurons == 4
        assert len(out) == 4
        assert len(sc.tau) == 4
        assert np.all(sc.tau == 10.0)

    def test_if_scalar_broadcast(self):
        node = nir.IF(r=np.array([1.0]), v_threshold=np.array([0.5]))
        sc = map_node("if", node)
        assert sc.n_neurons == 1
        out = sc.forward(np.array([1.0, 0.2, 0.8]))
        assert sc.n_neurons == 3
        assert len(out) == 3

    def test_li_scalar_broadcast(self):
        node = nir.LI(
            tau=np.array([10.0]),
            r=np.array([1.0]),
            v_leak=np.array([0.0]),
        )
        sc = map_node("li", node, dt=1.0)
        assert sc.n_neurons == 1
        out = sc.forward(np.array([1.0, 2.0]))
        assert sc.n_neurons == 2
        assert len(out) == 2

    def test_cubalif_scalar_broadcast(self):
        node = nir.CubaLIF(
            tau_syn=np.array([5.0]),
            tau_mem=np.array([10.0]),
            r=np.array([1.0]),
            v_leak=np.array([0.0]),
            v_threshold=np.array([1.0]),
            w_in=np.array([1.0]),
        )
        sc = map_node("cubalif", node, dt=1.0)
        assert sc.n_neurons == 1
        out = sc.forward(np.array([5.0, 3.0, 1.0]))
        assert sc.n_neurons == 3
        assert len(out) == 3
        assert len(sc.tau_syn) == 3
        assert len(sc.w_in) == 3

    def test_cubali_scalar_broadcast(self):
        node = nir.CubaLI(
            tau_syn=np.array([5.0]),
            tau_mem=np.array([10.0]),
            r=np.array([1.0]),
            v_leak=np.array([0.0]),
            w_in=np.array([1.0]),
        )
        sc = map_node("cubali", node, dt=1.0)
        assert sc.n_neurons == 1
        out = sc.forward(np.array([1.0, 2.0, 3.0, 4.0]))
        assert sc.n_neurons == 4
        assert len(out) == 4

    def test_cubalif_scalar_in_graph(self):
        """Scalar CubaLIF after Affine(4,6) — the real-world snnTorch pattern.

        snnTorch exports CubaLIF with scalar params and empty input_type.
        nirtorch builds the graph without strict type inference in this case.
        We replicate that by importing the snnTorch-exported graph directly.
        """
        pytest = __import__("pytest")
        try:
            import snntorch as snn
            from snntorch.export_nir import export_to_nir
            import torch
        except ImportError:
            pytest.skip("snntorch not installed")

        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            snn.Synaptic(alpha=0.9, beta=0.8),
        )
        graph = export_to_nir(model, torch.randn(1, 4))

        from sc_neurocore.nir_bridge import from_nir

        net = from_nir(graph, dt=1e-4)
        results = net.run({"input": np.array([1.0, 2.0, 3.0, 4.0])}, steps=10)
        assert len(results["output"]) == 10
        assert len(results["output"][0]) == 6


# --- Threshold and reset mode tests ---


class TestThresholdAndReset:
    def test_strict_threshold(self):
        """NIR spec: z=1 when v > v_threshold (strict, not >=)."""
        node = nir.LIF(
            tau=np.array([100.0]),
            r=np.array([1.0]),
            v_leak=np.array([0.0]),
            v_threshold=np.array([1.0]),
        )
        sc = map_node("lif", node, dt=1.0)
        # Feed exactly threshold worth of current: v should reach 1.0 but not spike
        # dv = (0 - 0 + 1*100) * 1/100 = 1.0, so v = 1.0
        sc.forward(np.array([100.0]))
        assert sc.v[0] == 1.0  # strict >: v == threshold means no spike
        # Larger input to push v above threshold
        # dv = (0 - 1.0 + 1*200) * 0.01 = 199*0.01 = 1.99, v = 1.0 + 1.99 = 2.99
        out = sc.forward(np.array([200.0]))
        assert out[0] == 1.0  # now fires (v > threshold)

    def test_reset_mode_default(self):
        """Default reset: v = v_reset after spike."""
        from sc_neurocore.nir_bridge import from_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([1])}),
            "lif": nir.LIF(
                tau=np.array([1.0]),
                r=np.array([1.0]),
                v_leak=np.array([0.0]),
                v_threshold=np.array([0.5]),
                v_reset=np.array([0.1]),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("input", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0, reset_mode="reset")
        net.run({"input": np.array([10.0])}, steps=3)
        lif = net.nodes["lif"]
        # After spiking, v should be near v_reset (0.1), not near v-threshold
        assert lif.v[0] < 1.0

    def test_reset_mode_subtract(self):
        """Subtract reset: v = v - v_threshold after spike."""
        from sc_neurocore.nir_bridge import from_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([1])}),
            "lif": nir.LIF(
                tau=np.array([1.0]),
                r=np.array([1.0]),
                v_leak=np.array([0.0]),
                v_threshold=np.array([0.5]),
                v_reset=np.array([0.0]),
            ),
            "output": nir.Output(output_type={"output": np.array([1])}),
        }
        edges = [("input", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0, reset_mode="subtract")
        # Large input -> spike -> v should be v_at_spike - threshold, not v_reset
        out = net.step({"input": np.array([2.0])})
        assert out["output"][0] == 1.0  # spiked
        lif = net.nodes["lif"]
        # v was 2.0 (from r*I*dt/tau = 1*2*1/1 = 2), spiked, subtract: 2.0-0.5=1.5
        assert lif.v[0] == pytest.approx(1.5)

    def test_cubalif_subtract_reset(self):
        """CubaLIF subtract reset mode."""
        from sc_neurocore.nir_bridge import from_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "cubalif": nir.CubaLIF(
                tau_syn=np.array([1.0, 1.0]),
                tau_mem=np.array([1.0, 1.0]),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
                w_in=np.ones(2),
                v_reset=np.zeros(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "cubalif"), ("cubalif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0, reset_mode="subtract")
        out = net.run({"input": np.array([5.0, 5.0])}, steps=3)
        # Should produce spikes with subtract reset
        total = sum(r.sum() for r in out["output"])
        assert total > 0


class TestSpikingJellyInterop:
    def test_spikingjelly_lif_roundtrip(self):
        """SpikingJelly LIFNode -> NIR -> SC-NeuroCore produces identical spikes."""
        pytest_mod = __import__("pytest")
        try:
            import torch
            from spikingjelly.activation_based import neuron, layer, functional
            from spikingjelly.activation_based.nir_exchange import export_to_nir
        except ImportError:
            pytest_mod.skip("spikingjelly not installed from git")

        from sc_neurocore.nir_bridge import from_nir

        torch.manual_seed(42)

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = layer.Linear(4, 6)
                self.lif1 = neuron.LIFNode(tau=2.0)
                self.fc2 = layer.Linear(6, 2)
                self.lif2 = neuron.LIFNode(tau=2.0)

            def forward(self, x):
                x = self.lif1(self.fc1(x))
                x = self.lif2(self.fc2(x))
                return x

        model = Net()
        functional.set_step_mode(model, "s")
        graph = export_to_nir(model, torch.randn(1, 4), dt=1e-4)
        net = from_nir(graph, dt=1e-4)

        inp_t = torch.tensor([[5.0, 3.0, 1.0, 2.0]])
        inp_np = np.array([5.0, 3.0, 1.0, 2.0])
        functional.reset_net(model)

        mismatches = 0
        for _ in range(50):
            sj_out = model(inp_t).detach().numpy().flatten()
            sc_out = net.step({"x": inp_np})["output"]
            if not np.array_equal(sj_out, sc_out):
                mismatches += 1

        assert mismatches == 0, f"{mismatches}/50 mismatches"


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

    def test_cycle_inserts_delay_node(self):
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

    def test_recurrent_network_runs(self):
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

    def test_recurrent_reset_clears_delay(self):
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

    def test_summary_shows_recurrent(self):
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

    def test_multi_port_subgraph_creation(self):
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

    def test_multi_port_forward_single(self):
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

    def test_multi_port_forward_multi(self):
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

    def test_multi_port_requires_io(self):
        """Empty I/O should raise."""
        inner = SCNetwork(nodes={}, edges=[], input_nodes=[], output_nodes=[])
        with pytest.raises(ValueError, match="at least one input"):
            SCMultiPortSubgraphNode(name="empty", network=inner)


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
        np.testing.assert_allclose(lif_node.v, lif_node.v_leak)

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
    def test_roundtrip_lif_affine(self):
        from sc_neurocore.nir_bridge import from_nir, to_nir

        graph = _make_lif_affine_graph(n_in=3, n_out=2)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported, nir.NIRGraph)
        assert len(exported.nodes) == 4
        assert len(exported.edges) == 3
        assert isinstance(exported.nodes["lif"], nir.LIF)
        assert isinstance(exported.nodes["affine"], nir.Affine)

    def test_roundtrip_file_io(self, local_tmp_path):
        from sc_neurocore.nir_bridge import from_nir, to_nir

        graph = _make_lif_affine_graph()
        network = from_nir(graph)
        path = local_tmp_path / "exported.nir"
        to_nir(network, path=str(path))
        assert path.exists()
        reloaded = from_nir(str(path))
        assert len(reloaded.nodes) == 4

    def test_export_type_error(self):
        from sc_neurocore.nir_bridge import to_nir

        with pytest.raises(TypeError, match="Expected SCNetwork"):
            to_nir("not a network")

    def test_export_linear_chain(self):
        from sc_neurocore.nir_bridge import from_nir, to_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "linear": nir.Linear(weight=np.eye(2, dtype=np.float32)),
            "scale": nir.Scale(scale=np.array([2.0, 2.0])),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "linear"), ("linear", "scale"), ("scale", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["linear"], nir.Linear)
        assert isinstance(exported.nodes["scale"], nir.Scale)

    def test_export_all_basic_types(self):
        """Roundtrip every exportable node type through from_nir -> to_nir."""
        from sc_neurocore.nir_bridge import from_nir, to_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "lif": nir.LIF(
                tau=np.array([10.0, 10.0]),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "if": nir.IF(r=np.array([1.0, 1.0]), v_threshold=np.array([0.5, 0.5])),
            "li": nir.LI(tau=np.array([10.0, 10.0]), r=np.ones(2), v_leak=np.zeros(2)),
            "integ": nir.I(r=np.array([1.0, 1.0])),
            "aff": nir.Affine(
                weight=np.eye(2, dtype=np.float32), bias=np.zeros(2, dtype=np.float32)
            ),
            "lin": nir.Linear(weight=np.eye(2, dtype=np.float32)),
            "scale": nir.Scale(scale=np.array([2.0, 2.0])),
            "thr": nir.Threshold(threshold=np.array([0.5, 0.5])),
            "flat": nir.Flatten(start_dim=0, end_dim=-1),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [
            ("input", "lif"),
            ("lif", "if"),
            ("if", "li"),
            ("li", "integ"),
            ("integ", "aff"),
            ("aff", "lin"),
            ("lin", "scale"),
            ("scale", "thr"),
            ("thr", "flat"),
            ("flat", "output"),
        ]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["lif"], nir.LIF)
        assert isinstance(exported.nodes["if"], nir.IF)
        assert isinstance(exported.nodes["li"], nir.LI)
        assert isinstance(exported.nodes["integ"], nir.I)
        assert isinstance(exported.nodes["aff"], nir.Affine)
        assert isinstance(exported.nodes["lin"], nir.Linear)
        assert isinstance(exported.nodes["scale"], nir.Scale)
        assert isinstance(exported.nodes["thr"], nir.Threshold)
        assert isinstance(exported.nodes["flat"], nir.Flatten)

    def test_export_cubalif_roundtrip(self):
        """CubaLIF roundtrip preserves all 7 parameters exactly."""
        from sc_neurocore.nir_bridge import from_nir, to_nir

        orig = nir.CubaLIF(
            tau_syn=np.array([5.0, 5.0]),
            tau_mem=np.array([10.0, 10.0]),
            r=np.array([0.8, 0.8]),
            v_leak=np.array([-0.1, -0.1]),
            v_threshold=np.ones(2),
            w_in=np.array([1.2, 1.2]),
            v_reset=np.array([-0.25, -0.25]),
        )
        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "cubalif": orig,
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "cubalif"), ("cubalif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph, dt=1.0)
        exported = to_nir(network)
        out = exported.nodes["cubalif"]
        assert isinstance(out, nir.CubaLIF)
        for param in ["tau_syn", "tau_mem", "r", "v_leak", "v_threshold", "w_in", "v_reset"]:
            np.testing.assert_array_equal(getattr(out, param), getattr(orig, param))

    def test_export_cubali_roundtrip(self):
        """CubaLI roundtrip preserves parameters."""
        from sc_neurocore.nir_bridge import from_nir, to_nir

        orig = nir.CubaLI(
            tau_syn=np.array([5.0, 5.0]),
            tau_mem=np.array([10.0, 10.0]),
            r=np.ones(2),
            v_leak=np.zeros(2),
            w_in=np.array([1.5, 1.5]),
        )
        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "cubali": orig,
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "cubali"), ("cubali", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        out = exported.nodes["cubali"]
        assert isinstance(out, nir.CubaLI)
        for param in ["tau_syn", "tau_mem", "r", "v_leak", "w_in"]:
            np.testing.assert_array_equal(getattr(out, param), getattr(orig, param))

    def test_export_delay_roundtrip(self):
        """Delay node roundtrip."""
        from sc_neurocore.nir_bridge import from_nir, to_nir

        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "lif": nir.LIF(
                tau=np.array([10.0, 10.0]),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "delay": nir.Delay(delay=np.array([1.0, 1.0])),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "lif"), ("lif", "delay"), ("delay", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["delay"], nir.Delay)

    def test_export_conv_pool_roundtrip(self):
        """Conv1d, Conv2d, SumPool2d, AvgPool2d roundtrip."""
        from sc_neurocore.nir_bridge import from_nir, to_nir

        w1d = np.random.randn(2, 1, 3).astype(np.float32)
        conv1d_node = nir.Conv1d(
            input_shape=8,
            weight=w1d,
            bias=np.zeros(2, dtype=np.float32),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        nodes = {
            "input": nir.Input(input_type=conv1d_node.input_type),
            "conv1d": conv1d_node,
            "output": nir.Output(output_type=conv1d_node.output_type),
        }
        edges = [("input", "conv1d"), ("conv1d", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        network = from_nir(graph)
        exported = to_nir(network)
        assert isinstance(exported.nodes["conv1d"], nir.Conv1d)
        np.testing.assert_array_equal(exported.nodes["conv1d"].weight, w1d)

        w2d = np.random.randn(2, 1, 3, 3).astype(np.float32)
        conv2d_node = nir.Conv2d(
            input_shape=np.array([4, 4]),
            weight=w2d,
            bias=np.zeros(2, dtype=np.float32),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        nodes2 = {
            "input": nir.Input(input_type=conv2d_node.input_type),
            "conv2d": conv2d_node,
            "output": nir.Output(output_type=conv2d_node.output_type),
        }
        edges2 = [("input", "conv2d"), ("conv2d", "output")]
        graph2 = nir.NIRGraph(nodes=nodes2, edges=edges2)
        network2 = from_nir(graph2)
        exported2 = to_nir(network2)
        assert isinstance(exported2.nodes["conv2d"], nir.Conv2d)
        np.testing.assert_array_equal(exported2.nodes["conv2d"].weight, w2d)

        nodes3 = {
            "input": nir.Input(input_type={"input": np.array([1, 4, 4])}),
            "spool": nir.SumPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([2, 2]),
                padding=np.array([0, 0]),
            ),
            "output": nir.Output(output_type={"output": np.array([1, 2, 2])}),
        }
        edges3 = [("input", "spool"), ("spool", "output")]
        graph3 = nir.NIRGraph(nodes=nodes3, edges=edges3)
        network3 = from_nir(graph3)
        exported3 = to_nir(network3)
        assert isinstance(exported3.nodes["spool"], nir.SumPool2d)

        nodes4 = {
            "input": nir.Input(input_type={"input": np.array([1, 4, 4])}),
            "apool": nir.AvgPool2d(
                kernel_size=np.array([2, 2]),
                stride=np.array([2, 2]),
                padding=np.array([0, 0]),
            ),
            "output": nir.Output(output_type={"output": np.array([1, 2, 2])}),
        }
        edges4 = [("input", "apool"), ("apool", "output")]
        graph4 = nir.NIRGraph(nodes=nodes4, edges=edges4)
        network4 = from_nir(graph4)
        exported4 = to_nir(network4)
        assert isinstance(exported4.nodes["apool"], nir.AvgPool2d)
