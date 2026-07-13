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
        node = nir.IF(r=np.array([1.0]), v_threshold=np.array([0.5]))
        sc = map_node("if", node)
        assert sc.n_neurons == 1
        out = sc.forward(np.array([1.0, 0.2, 0.8]))
        assert sc.n_neurons == 3
        assert len(out) == 3

    def test_li_scalar_broadcast(self) -> None:
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
        """Scalar CubaLIF after Affine(4,6) — the real-world snnTorch pattern.

        snnTorch exports CubaLIF with scalar params and empty input_type.
        nirtorch builds the graph without strict type inference in this case.
        We replicate that by importing the snnTorch-exported graph directly.
        """
        pytest = __import__("pytest")
        try:
            import snntorch as snn  # type: ignore[import-not-found]  # optional dependency lacks stubs
            from snntorch.export_nir import (  # type: ignore[import-not-found]  # optional dependency lacks stubs
                export_to_nir,
            )
            import torch
        except ImportError:
            pytest.skip("snntorch not installed")

        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            snn.Synaptic(alpha=0.9, beta=0.8),
        )
        graph = export_to_nir(model, torch.randn(1, 4))

        net = from_nir(graph, dt=1e-4)
        results = net.run({"input": np.array([1.0, 2.0, 3.0, 4.0])}, steps=10)
        assert len(results["output"]) == 10
        assert len(results["output"][0]) == 6


# --- Threshold and reset mode tests ---
