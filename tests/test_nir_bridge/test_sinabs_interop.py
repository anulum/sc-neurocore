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
    SCLIFNode,
    SCIFNode,
    SCLINode,
)


class TestSinabsInterop:
    def _sinabs_lif_graph(self, n: int = 4, tau_mem: float = 10.0) -> Any:
        """Reproduce what sinabs.nir.to_nir() exports for Linear→LIF."""
        rng = np.random.RandomState(99)
        nodes = {
            "input": nir.Input(input_type={"input": np.array([n])}),
            # Sinabs always exports Affine (bias=zeros even if no bias)
            "affine": nir.Affine(
                weight=rng.randn(n, n).astype(np.float32),
                bias=np.zeros(n, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(n, tau_mem),
                r=np.ones(n),  # sinabs: r=1 always
                v_leak=np.zeros(n),  # sinabs: v_leak=0 always
                v_threshold=np.ones(n),
            ),
            "output": nir.Output(output_type={"output": np.array([n])}),
        }
        edges = [("input", "affine"), ("affine", "lif"), ("lif", "output")]
        return nir.NIRGraph(nodes=nodes, edges=edges)

    def test_sinabs_lif_loads(self) -> None:
        """Sinabs-style LIF graph parses and runs."""
        graph = self._sinabs_lif_graph()
        net = from_nir(graph, dt=1.0)
        assert isinstance(net.nodes["lif"], SCLIFNode)
        assert net.nodes["lif"].tau[0] == 10.0
        assert net.nodes["lif"].r[0] == 1.0
        assert net.nodes["lif"].v_leak[0] == 0.0
        results = net.run({"input": np.array([5.0, 3.0, 1.0, 2.0])}, steps=50)
        total_spikes = sum(r.sum() for r in results["output"])
        assert total_spikes > 0

    def test_sinabs_euler_vs_exponential(self) -> None:
        """Quantify Euler (ours) vs exponential (sinabs) decay mismatch.

        Sinabs: alpha = exp(-dt/tau), v *= alpha
        SC-NeuroCore (Euler): v += (v_leak - v) * dt/tau
        For v_leak=0: v *= (1 - dt/tau) [Euler] vs v *= exp(-dt/tau) [exact]
        With dt=1, tau=10: Euler=0.9, exact=0.9048. ~0.5% per step.
        """
        tau = 10.0
        dt = 1.0
        euler_decay = 1 - dt / tau  # 0.9
        exact_decay = np.exp(-dt / tau)  # 0.9048
        # After 50 steps of pure leak (no input):
        # Euler: v0 * 0.9^50 = 0.00515
        # Exact: v0 * 0.9048^50 = 0.00657
        # These diverge — this is expected, not a bug.
        v_euler = 1.0 * euler_decay**50
        v_exact = 1.0 * exact_decay**50
        # Euler underestimates voltage (decays faster)
        assert v_euler < v_exact
        # Within 30% after 50 steps — acceptable for discrete-time bridge
        assert abs(v_euler - v_exact) / v_exact < 0.3

    def test_sinabs_iaf_graph(self) -> None:
        """Sinabs IAF → nir.IF roundtrip through bridge."""
        nodes = {
            "input": nir.Input(input_type={"input": np.array([2])}),
            "if": nir.IF(
                r=np.ones(2),  # sinabs: r=1
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        }
        edges = [("input", "if"), ("if", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0)
        assert isinstance(net.nodes["if"], SCIFNode)
        results = net.run({"input": np.array([0.6, 0.6])}, steps=5)
        # IF accumulates: step1 v=0.6, step2 v=1.2>1→spike
        spike_counts = [r.sum() for r in results["output"]]
        assert sum(spike_counts) > 0

    def test_sinabs_expleak_graph(self) -> None:
        """Sinabs ExpLeak → nir.LI roundtrip through bridge."""
        tau = 5.0
        nodes = {
            "input": nir.Input(input_type={"input": np.array([3])}),
            "li": nir.LI(
                tau=np.full(3, tau),
                r=np.ones(3),
                v_leak=np.zeros(3),
            ),
            "output": nir.Output(output_type={"output": np.array([3])}),
        }
        edges = [("input", "li"), ("li", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph, dt=1.0)
        assert isinstance(net.nodes["li"], SCLINode)
        out = net.step({"input": np.array([1.0, 2.0, 3.0])})
        # dt/tau = 0.2, dv = (0 - 0 + 1*I) * 0.2 = 0.2*I
        np.testing.assert_allclose(out["output"], [0.2, 0.4, 0.6])


# --- Rockpool interop tests ---
# Rockpool exports: nir.LIF, nir.CubaLIF, nir.LI, nir.Linear/Affine.
# Key: r = tau * exp(-dt/tau) / dt (encodes dt into r).
# Weights are transposed on export (NIR is (out,in), rockpool is (in,out)).
# Tests use dt=1e-3 and tau_mem=10.0, matching rockpool's test suite.
