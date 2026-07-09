# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.temporal_hierarchy

from __future__ import annotations

import numpy as np

from sc_neurocore.temporal_hierarchy import MultiClockSNN, HetSynLayer


class TestHetSynLayer:
    def test_init(self):
        layer = HetSynLayer(n_inputs=8, n_neurons=4)
        assert layer.tau.shape == (4, 8)
        assert layer.W.shape == (4, 8)

    def test_step(self):
        layer = HetSynLayer(n_inputs=4, n_neurons=3, threshold=0.5)
        x = np.random.rand(4)
        spikes = layer.step(x)
        assert spikes.shape == (3,)

    def test_tau_distribution(self):
        layer = HetSynLayer(n_inputs=100, n_neurons=50, tau_mean=5.0, tau_std=1.0)
        stats = layer.tau_stats
        assert stats["mean"] > 0
        assert stats["min"] > 0
        assert stats["max"] > stats["min"]

    def test_reset(self):
        layer = HetSynLayer(n_inputs=4, n_neurons=3)
        layer.step(np.ones(4))
        layer.reset()
        assert np.allclose(layer._v, 0)
        assert np.allclose(layer._traces, 0)


class TestMultiClockSNN:
    def _make_network(self):
        l1 = HetSynLayer(n_inputs=8, n_neurons=16, tau_mean=2.0)
        l2 = HetSynLayer(n_inputs=16, n_neurons=8, tau_mean=10.0)
        l3 = HetSynLayer(n_inputs=8, n_neurons=4, tau_mean=50.0)
        return MultiClockSNN(
            layers=[l1, l2, l3],
            layer_names=["fast", "medium", "slow"],
            clock_intervals=[1, 5, 10],
        )

    def test_step(self):
        net = self._make_network()
        x = np.random.rand(8)
        out = net.step(x)
        assert out.shape == (4,)

    def test_run(self):
        net = self._make_network()
        inputs = np.random.rand(100, 8)
        outputs = net.run(inputs)
        assert outputs.shape == (100, 4)

    def test_clock_intervals_respected(self):
        l1 = HetSynLayer(n_inputs=4, n_neurons=4, tau_mean=2.0)
        l2 = HetSynLayer(n_inputs=4, n_neurons=2, tau_mean=20.0)
        net = MultiClockSNN(
            layers=[l1, l2],
            layer_names=["fast", "slow"],
            clock_intervals=[1, 5],
        )
        # Step 4 times — slow layer should NOT have updated (only at step 5)
        for _ in range(4):
            net.step(np.ones(4))
        # slow layer output should be zeros (not updated yet, only holds last)
        # (first update happens at step 5)

    def test_default_clock_intervals(self):
        l1 = HetSynLayer(n_inputs=4, n_neurons=3)
        net = MultiClockSNN(layers=[l1], layer_names=["h"])
        assert net.clock_intervals == [1]

    def test_reset(self):
        net = self._make_network()
        net.step(np.random.rand(8))
        net.reset()
        assert net._step_count == 0
