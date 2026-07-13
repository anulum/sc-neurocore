# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

from typing import Any

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import from_nir
from sc_neurocore.nir_bridge.node_map import (
    SCCubaLIFNode,
)
from sc_neurocore.nir_bridge.parser import (
    SCSubgraphNode,
)


class TestSnnTorchRSynapticInterop:
    def _snntorch_rsynaptic_graph(
        self,
        n: int = 4,
        beta: float = 0.8,
        alpha: float = 0.9,
    ) -> tuple[Any, float]:
        """Construct an RSynaptic-style NIR graph matching snnTorch export.

        snnTorch encodes: tau_mem = dt/(1-beta), tau_syn = dt/(1-alpha),
        r = tau_mem/dt = 1/(1-beta), w_in = tau_syn/dt = 1/(1-alpha).
        """
        dt = 1e-4
        tau_mem = dt / (1 - beta)  # 5e-4
        tau_syn = dt / (1 - alpha)  # 1e-3
        r = tau_mem / dt  # 5.0
        w_in = tau_syn / dt  # 10.0
        rng = np.random.RandomState(88)
        w_rec = np.abs(rng.randn(n, n).astype(np.float32)) * 0.05

        # RSynaptic exports as a subgraph with recurrent CubaLIF
        sub_nodes = {
            "input": nir.Input(input_type={"input": np.array([n])}),
            "cubalif": nir.CubaLIF(
                tau_syn=np.full(n, tau_syn),
                tau_mem=np.full(n, tau_mem),
                r=np.full(n, r),
                v_leak=np.zeros(n),
                v_threshold=np.ones(n),
                w_in=np.full(n, w_in),
                v_reset=np.zeros(n),
            ),
            "w_rec": nir.Linear(weight=w_rec),
            "output": nir.Output(output_type={"output": np.array([n])}),
        }
        sub_edges = [
            ("input", "cubalif"),
            ("cubalif", "w_rec"),
            ("w_rec", "cubalif"),
            ("cubalif", "output"),
        ]
        rsynaptic_subgraph = nir.NIRGraph(nodes=sub_nodes, edges=sub_edges)

        # Wrap in outer graph: Affine → RSynaptic subgraph
        # Positive weights to ensure excitatory drive
        nodes = {
            "input": nir.Input(input_type={"input": np.array([n])}),
            "linear": nir.Affine(
                weight=np.abs(rng.randn(n, n).astype(np.float32)) * 0.5,
                bias=np.zeros(n, dtype=np.float32),
            ),
            "rsynaptic": rsynaptic_subgraph,
            "output": nir.Output(output_type={"output": np.array([n])}),
        }
        edges = [("input", "linear"), ("linear", "rsynaptic"), ("rsynaptic", "output")]
        return nir.NIRGraph(nodes=nodes, edges=edges), dt

    def test_rsynaptic_parses(self) -> None:
        """RSynaptic subgraph parses into SCSubgraphNode."""
        graph, dt = self._snntorch_rsynaptic_graph()
        net = from_nir(graph, dt=dt)
        sub = net.nodes["rsynaptic"]
        assert isinstance(sub, SCSubgraphNode)
        inner = sub.network
        assert "cubalif" in inner.nodes
        assert isinstance(inner.nodes["cubalif"], SCCubaLIFNode)
        # Trigger topological sort (populates _recurrent_map lazily)
        _ = inner.topo_order
        # w_rec creates a cycle: cubalif→w_rec→cubalif, broken by delay node
        assert len(inner._recurrent_map) > 0

    def test_rsynaptic_runs(self) -> None:
        """RSynaptic graph produces output over multiple timesteps."""
        graph, dt = self._snntorch_rsynaptic_graph()
        net = from_nir(graph, dt=dt)
        results = net.run({"input": np.array([10.0, 5.0, 3.0, 1.0])}, steps=100)
        assert len(results["output"]) == 100
        # With strong input, at least some timesteps should produce spikes
        total_spikes = sum(r.sum() for r in results["output"])
        assert total_spikes > 0

    def test_rsynaptic_cubalif_params(self) -> None:
        """Verify snnTorch CubaLIF parameter encoding is preserved."""
        dt = 1e-4
        beta, alpha = 0.8, 0.9
        tau_mem = dt / (1 - beta)
        tau_syn = dt / (1 - alpha)
        r_expected = tau_mem / dt
        w_in_expected = tau_syn / dt

        graph, _ = self._snntorch_rsynaptic_graph(beta=beta, alpha=alpha)
        net = from_nir(graph, dt=dt)
        cubalif = net.nodes["rsynaptic"].network.nodes["cubalif"]
        np.testing.assert_allclose(cubalif.tau_mem[0], tau_mem, rtol=1e-10)
        np.testing.assert_allclose(cubalif.tau_syn[0], tau_syn, rtol=1e-10)
        np.testing.assert_allclose(cubalif.r[0], r_expected, rtol=1e-10)
        np.testing.assert_allclose(cubalif.w_in[0], w_in_expected, rtol=1e-10)

    def test_rsynaptic_reset_clears_subgraph(self) -> None:
        """Reset on outer network propagates to RSynaptic subgraph."""
        graph, dt = self._snntorch_rsynaptic_graph()
        net = from_nir(graph, dt=dt)
        net.run({"input": np.array([10.0, 5.0, 3.0, 1.0])}, steps=20)
        cubalif = net.nodes["rsynaptic"].network.nodes["cubalif"]
        # i_syn accumulates during simulation (v may be 0 from resets)
        assert not np.allclose(cubalif.i_syn, 0.0)
        net.reset()
        np.testing.assert_allclose(cubalif.v, cubalif.v_leak)
        np.testing.assert_allclose(cubalif.i_syn, 0.0)


# --- Cross-framework r-encoding comparison ---
