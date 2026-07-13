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


class TestNodeMapping:
    def test_map_input(self) -> None:
        node = nir.Input(input_type={"input": np.array([4])})
        sc = map_node("inp", node)
        assert isinstance(sc, SCInputNode)
        assert sc.shape == (4,)

    def test_map_output(self) -> None:
        node = nir.Output(output_type={"output": np.array([3])})
        sc = map_node("out", node)
        assert isinstance(sc, SCOutputNode)
        assert sc.shape == (3,)

    def test_map_lif(self) -> None:
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

    def test_map_if(self) -> None:
        node = nir.IF(
            r=np.array([1.0]),
            v_threshold=np.array([0.5]),
        )
        sc = map_node("if", node)
        assert isinstance(sc, SCIFNode)
        assert sc.n_neurons == 1
        sc.forward(np.array([1.0]))
        sc.reset()
        assert sc.v is not None
        np.testing.assert_allclose(sc.v, [0.0])

    def test_map_li(self) -> None:
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
        assert sc.v is not None
        np.testing.assert_allclose(sc.v, [-0.5])

    def test_map_affine(self) -> None:
        node = nir.Affine(
            weight=np.eye(3, dtype=np.float32),
            bias=np.ones(3, dtype=np.float32),
        )
        sc = map_node("aff", node)
        assert isinstance(sc, SCAffineNode)
        out = sc.forward(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [2.0, 3.0, 4.0])

    def test_map_linear(self) -> None:
        node = nir.Linear(weight=np.eye(2, dtype=np.float32))
        sc = map_node("lin", node)
        assert isinstance(sc, SCLinearNode)
        out = sc.forward(np.array([5.0, 7.0]))
        np.testing.assert_allclose(out, [5.0, 7.0])

    def test_map_scale(self) -> None:
        node = nir.Scale(scale=np.array([2.0, 0.5]))
        sc = map_node("scl", node)
        assert isinstance(sc, SCScaleNode)
        out = sc.forward(np.array([3.0, 4.0]))
        np.testing.assert_allclose(out, [6.0, 2.0])

    def test_map_threshold(self) -> None:
        node = nir.Threshold(threshold=np.array([0.5]))
        sc = map_node("thr", node)
        assert isinstance(sc, SCThresholdNode)
        assert sc.forward(np.array([0.6]))[0] == 1.0
        assert sc.forward(np.array([0.3]))[0] == 0.0

    def test_map_flatten(self) -> None:
        node = nir.Flatten(start_dim=0, end_dim=-1)
        sc = map_node("flat", node)
        assert isinstance(sc, SCFlattenNode)
        out = sc.forward(np.arange(6).reshape(2, 3))
        np.testing.assert_array_equal(out, np.arange(6))

    def test_map_integrator(self) -> None:
        node = nir.I(r=np.array([1.0]))
        sc = map_node("integ", node)
        assert isinstance(sc, SCIntegratorNode)
        sc.forward(np.array([1.0]))
        sc.forward(np.array([1.0]))
        assert sc.v is not None
        np.testing.assert_allclose(sc.v, [2.0])

    def test_map_delay(self) -> None:
        node = nir.Delay(delay=np.array([2.0]))
        sc = map_node("dly", node, dt=1.0)
        assert isinstance(sc, SCDelayNode)
        # Feed 1.0, expect 0.0 for 2 steps (delay=2), then 1.0
        assert sc.forward(np.array([1.0]))[0] == 0.0
        assert sc.forward(np.array([2.0]))[0] == 0.0
        assert sc.forward(np.array([3.0]))[0] == 1.0
        sc.reset()
        assert sc.forward(np.array([5.0]))[0] == 0.0

    def test_map_cuba_lif(self) -> None:
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
        assert sc.i_syn is not None
        np.testing.assert_allclose(sc.i_syn, [0.0])

    def test_map_cuba_li(self) -> None:
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
        assert sc.v is not None
        np.testing.assert_allclose(sc.v, [0.0])

    def test_map_sum_pool2d(self) -> None:
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

    def test_map_avg_pool2d(self) -> None:
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

    def test_map_conv1d(self) -> None:
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

    def test_map_conv2d(self) -> None:
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

    def test_all_18_primitives_mapped(self) -> None:
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
