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


class TestCrossFrameworkREncoding:
    """Verify that the same physical neuron produces different NIR r values
    depending on which framework exported it, and that our bridge handles
    all encodings when the user provides the correct dt."""

    def test_r_encoding_differences(self) -> None:
        """Same tau/dt produce different r across frameworks."""
        tau = 10.0
        dt = 1e-3
        r_sinabs = 1.0  # sinabs: r=1 always
        r_snntorch = tau / dt  # 10000.0
        r_rockpool = tau * np.exp(-dt / tau) / dt  # ~9999.0005
        # All three are different
        assert r_sinabs != r_snntorch
        assert r_snntorch != r_rockpool
        # snnTorch and rockpool are close (differ by ~0.005% at small dt/tau)
        assert abs(r_snntorch - r_rockpool) / r_snntorch < 0.001

    def test_all_r_encodings_produce_spikes(self) -> None:
        """All three r-encodings produce spikes with matching dt."""
        tau = 10.0
        dt = 1e-3
        encodings = {
            # Sinabs: r=1, no dt baked in — use dt=1.0 (physical tau)
            "sinabs": (1.0, 1.0),
            "snntorch": (tau / dt, dt),
            "rockpool": (tau * np.exp(-dt / tau) / dt, dt),
        }
        for name, (r, use_dt) in encodings.items():
            nodes = {
                "input": nir.Input(input_type={"input": np.array([1])}),
                "lif": nir.LIF(
                    tau=np.array([tau]),
                    r=np.array([r]),
                    v_leak=np.zeros(1),
                    v_threshold=np.ones(1),
                ),
                "output": nir.Output(output_type={"output": np.array([1])}),
            }
            edges = [("input", "lif"), ("lif", "output")]
            graph = nir.NIRGraph(nodes=nodes, edges=edges)
            net = from_nir(graph, dt=use_dt)
            results = net.run({"input": np.array([5.0])}, steps=500)
            spikes = sum(r_.sum() for r_ in results["output"])
            assert spikes > 0, f"{name} r-encoding produced no spikes"
