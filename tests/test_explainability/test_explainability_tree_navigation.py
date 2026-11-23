# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTreeNavigation from former test_explainability.py

"""Focused suite: TestTreeNavigation from former test_explainability.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.explainability.explainability import (
    SpikeDecisionTree,
)


class TestTreeNavigation:
    """Looking up nodes in a decision tree."""

    def test_get_node(self) -> None:
        """A node is found by its neuron and a missing one is None."""
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4)
        node = tree.get_node("n0")
        assert node is not None
        assert node.neuron_id == "n0"
        assert tree.get_node("missing") is None

    def test_nodes_at_layer(self) -> None:
        """Nodes are listed per layer."""
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, layer_id="L1")
        tree.add_decision("n1", np.ones(8, dtype=np.uint8), 4, layer_id="L2")
        assert len(tree.nodes_at_layer("L1")) == 1
        assert len(tree.nodes_at_layer("L2")) == 1

    def test_nodes_at_timestep(self) -> None:
        """Nodes are listed per timestep."""
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, timestep=5)
        tree.add_decision("n1", np.ones(8, dtype=np.uint8), 4, timestep=5)
        assert len(tree.nodes_at_timestep(5)) == 2

    def test_spike_path(self) -> None:
        """The spike path holds only spiking neurons."""
        tree = SpikeDecisionTree()
        root = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        child = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, parent=root)
        tree.add_decision("n2", np.ones(8, dtype=np.uint8), 4, parent=child)
        path = tree.spike_path()
        spiking = [n.neuron_id for n in path]
        assert "n0" in spiking
        assert "n1" not in spiking
        assert "n2" in spiking
