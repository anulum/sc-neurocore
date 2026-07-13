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
    SCLIFNode,
    SCLINode,
    SCCubaLIFNode,
)


class TestRockpoolInterop:
    def _rockpool_r(self, tau: float, dt: float) -> float:
        """Compute r the way rockpool encodes it: r = tau * exp(-dt/tau) / dt."""
        return float(tau * np.exp(-dt / tau) / dt)

    def test_rockpool_lif_graph(self) -> None:
        """Rockpool LIFNeuronTorch → nir.LIF with encoded r."""
        dt = 1e-3
        tau = 10.0
        r = self._rockpool_r(tau, dt)  # ~9990.005
        n = 4
        rng = np.random.RandomState(77)
        nodes = {
            "input": nir.Input(input_type={"input": np.array([n])}),
            # Rockpool transposes weights on export
            "linear": nir.Linear(weight=rng.randn(n, n).astype(np.float32)),
            "lif": nir.LIF(
                tau=np.full(n, tau),
                r=np.full(n, r),
                v_leak=np.zeros(n),
                v_threshold=np.ones(n),
            ),
            "output": nir.Output(output_type={"output": np.array([n])}),
        }
        edges = [("input", "linear"), ("linear", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        # Must use matching dt for correct dynamics
        net = from_nir(graph, dt=dt)
        assert isinstance(net.nodes["lif"], SCLIFNode)
        assert net.nodes["lif"].dt == dt
        # Verify r was loaded correctly
        np.testing.assert_allclose(net.nodes["lif"].r[0], r, rtol=1e-10)
        results = net.run({"input": np.array([5.0, 3.0, 1.0, 2.0])}, steps=200)
        total_spikes = sum(r_.sum() for r_ in results["output"])
        assert total_spikes > 0

    def test_rockpool_cubalif_graph(self) -> None:
        """Rockpool LIFTorch → nir.CubaLIF with encoded r."""
        dt = 1e-3
        tau_mem = 10.0
        tau_syn = 5.0
        r_mem = self._rockpool_r(tau_mem, dt)
        n = 3
        nodes = {
            "input": nir.Input(input_type={"input": np.array([n])}),
            "cubalif": nir.CubaLIF(
                tau_syn=np.full(n, tau_syn),
                tau_mem=np.full(n, tau_mem),
                r=np.full(n, r_mem),
                v_leak=np.zeros(n),
                v_threshold=np.ones(n),
                w_in=np.ones(n),  # rockpool: w_in defaults to 1.0
            ),
            "output": nir.Output(output_type={"output": np.array([n])}),
        }
        edges = [("input", "cubalif"), ("cubalif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=dt)
        assert isinstance(net.nodes["cubalif"], SCCubaLIFNode)
        results = net.run({"input": np.array([10.0, 10.0, 10.0])}, steps=500)
        total_spikes = sum(r_.sum() for r_ in results["output"])
        assert total_spikes > 0

    def test_rockpool_li_graph(self) -> None:
        """Rockpool ExpSynTorch → nir.LI."""
        dt = 1e-3
        tau = 0.02
        r = self._rockpool_r(tau, dt)
        n = 2
        nodes = {
            "input": nir.Input(input_type={"input": np.array([n])}),
            "li": nir.LI(
                tau=np.full(n, tau),
                r=np.full(n, r),
                v_leak=np.zeros(n),
            ),
            "output": nir.Output(output_type={"output": np.array([n])}),
        }
        edges = [("input", "li"), ("li", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=dt)
        assert isinstance(net.nodes["li"], SCLINode)
        out = net.step({"input": np.array([1.0, 2.0])})
        # dv = (0 - 0 + r*I) * dt/tau
        expected = r * np.array([1.0, 2.0]) * dt / tau
        np.testing.assert_allclose(out["output"], expected, rtol=1e-10)

    def test_rockpool_euler_vs_exponential_divergence(self) -> None:
        """Document divergence between Euler (ours) and exponential (rockpool).

        Rockpool uses exact exponential: v *= exp(-dt/tau)
        We use Euler: v += (v_leak - v + r*I) * dt/tau
        For pure decay (I=0, v_leak=0): Euler gives v *= (1 - dt/tau)
        With dt=1e-3, tau=10: difference is ~5e-8 per step (negligible).
        """
        dt = 1e-3
        tau = 10.0
        euler_factor = 1 - dt / tau
        exact_factor = np.exp(-dt / tau)
        # Per-step relative error
        per_step_err = abs(euler_factor - exact_factor) / exact_factor
        assert per_step_err < 1e-6  # <1ppm per step
        # After 10000 steps (10 seconds): still tight
        v_euler = euler_factor**10000
        v_exact = exact_factor**10000
        assert abs(v_euler - v_exact) / v_exact < 0.01  # <1%


# --- snnTorch RSynaptic interop tests ---
# snnTorch RSynaptic exports as nir.NIRGraph subgraph:
#   Input → CubaLIF → Linear(w_rec) → CubaLIF → Output
# dt=1e-4 hardcoded. r=tau_mem/dt, w_in=tau_syn/dt.
