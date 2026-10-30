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
    map_node,
)


class TestScalarBroadcast:
    """Verify neuron nodes with scalar params auto-broadcast to input size."""

    def test_lif_scalar_broadcast(self) -> None:
        """A one-element LIF node maps to a single neuron that broadcasts over a wider input."""
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

    def test_if_scalar_broadcast(self) -> None:
        """A one-element IF node maps to a single neuron and fires on inputs above threshold."""
        node = nir.IF(r=np.array([1.0]), v_threshold=np.array([0.5]))
        sc = map_node("if", node)
        assert sc.n_neurons == 1
        out = sc.forward(np.array([1.0, 0.2, 0.8]))
        assert sc.n_neurons == 3
        assert len(out) == 3

    def test_li_scalar_broadcast(self) -> None:
        """A one-element LI node maps to a single leaky integrator over a wider input."""
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

    def test_cubalif_scalar_broadcast(self) -> None:
        """A one-element CubaLIF node maps to a single current-based LIF neuron."""
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

    def test_cubali_scalar_broadcast(self) -> None:
        """A one-element CubaLI node maps to a single current-based leaky integrator."""
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

    def test_cubalif_scalar_in_graph(self) -> None:
        """Scalar CubaLIF after Affine(4,6) with an empty input type: the snnTorch export shape.

        snnTorch exports CubaLIF with scalar parameters and an empty ``input_type``.
        The graph is built here directly, without NIR's type inference, so the
        bridge's handling of that shape is checked whether or not snnTorch is
        installed or its exporter works with the installed NIR.
        """
        rng = np.random.default_rng(42)
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([4])}),
                "0": nir.Affine(weight=rng.standard_normal((6, 4)), bias=rng.standard_normal(6)),
                "1": nir.CubaLIF(
                    tau_syn=np.array(1e-3),
                    tau_mem=np.array(5e-4),
                    r=np.array(1.0),
                    v_leak=np.array(0.0),
                    v_threshold=np.array(1.0),
                ),
                "output": nir.Output(output_type={"output": np.array([6])}),
            },
            edges=[("input", "0"), ("0", "1"), ("1", "output")],
            type_check=False,
        )
        net = from_nir(graph, dt=1e-4)
        results = net.run({"input": np.array([1.0, 2.0, 3.0, 4.0])}, steps=10)
        assert len(results["output"]) == 10
        assert len(results["output"][0]) == 6

    def test_snntorch_export_reaches_the_bridge(self) -> None:
        """Scalar snnTorch parameters cross the supported public FX import path."""
        snn = pytest.importorskip("snntorch")
        torch = pytest.importorskip("torch")
        pytest.importorskip("nirtorch")
        from sc_neurocore.nir_bridge import from_snntorch

        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 6, dtype=torch.float64),
            snn.Synaptic(alpha=0.9, beta=0.8, init_hidden=True, reset_delay=False),
        ).double()
        net = from_snntorch(model, torch.ones(4, dtype=torch.float64), dt=1e-4)
        results = net.run({net.input_nodes[0]: np.array([1.0, 2.0, 3.0, 4.0])}, steps=10)
        assert len(results[net.output_nodes[0]]) == 10
        assert len(results[net.output_nodes[0]][0]) == 6


# --- Threshold and reset mode tests ---
