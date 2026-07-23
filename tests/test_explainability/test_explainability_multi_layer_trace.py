# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiLayerTrace from former test_explainability.py

"""Focused suite: TestMultiLayerTrace from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403

class TestMultiLayerTrace:
    def test_add_and_layers(self):
        mlt = MultiLayerTrace()
        tree = SpikeDecisionTree()
        n0 = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, layer_id="L1")
        n1 = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, layer_id="L2")
        mlt.add(n0)
        mlt.add(n1)
        assert "L1" in mlt.layer_ids
        assert "L2" in mlt.layer_ids

    def test_spikes_at_layer(self):
        mlt = MultiLayerTrace()
        tree = SpikeDecisionTree()
        n0 = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, layer_id="L1")
        n1 = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, layer_id="L1")
        mlt.add(n0)
        mlt.add(n1)
        assert mlt.spikes_at_layer("L1") == 1
        assert mlt.spike_rate_at_layer("L1") == 0.5

    def test_propagation_path(self):
        mlt = MultiLayerTrace()
        tree = SpikeDecisionTree()
        for lid in ["L1", "L2", "L3"]:
            n = tree.add_decision(f"n_{lid}", np.ones(8, dtype=np.uint8), 4, layer_id=lid)
            mlt.add(n)
        path = mlt.propagation_path()
        assert len(path) == 3
        assert all("spike_rate" in p for p in path)
