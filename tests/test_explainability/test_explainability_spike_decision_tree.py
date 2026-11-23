# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeDecisionTree from former test_explainability.py

"""Focused suite: TestSpikeDecisionTree from former test_explainability.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.explainability.explainability import (
    SpikeDecision,
    SpikeDecisionTree,
)


class TestSpikeDecisionTree:
    """The tree of spike decisions."""

    def test_add_root_decision(self) -> None:
        """The first decision becomes the root with its popcount."""
        tree = SpikeDecisionTree()
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=3)
        assert tree.root is node
        assert node.decision == SpikeDecision.SPIKE
        assert node.popcount == 4

    def test_no_spike_below_threshold(self) -> None:
        """A popcount below threshold is no spike."""
        tree = SpikeDecisionTree()
        bs = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=5)
        assert node.decision == SpikeDecision.NO_SPIKE

    def test_child_nodes(self) -> None:
        """A child decision hangs under its parent."""
        tree = SpikeDecisionTree()
        bs1 = np.ones(8, dtype=np.uint8)
        root = tree.add_decision("n0", bs1, threshold=4)
        bs2 = np.zeros(8, dtype=np.uint8)
        child = tree.add_decision("n1", bs2, threshold=4, parent=root)
        assert len(root.children) == 1
        assert child.decision == SpikeDecision.NO_SPIKE

    def test_depth(self) -> None:
        """Three nested decisions give a depth of three."""
        tree = SpikeDecisionTree()
        root = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        child = tree.add_decision("n1", np.ones(8, dtype=np.uint8), 4, parent=root)
        tree.add_decision("n2", np.ones(8, dtype=np.uint8), 4, parent=child)
        assert tree.depth == 3

    def test_num_spikes(self) -> None:
        """The tree counts its spikes."""
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4)
        assert tree.num_spikes == 1

    def test_bitstream_hash_deterministic(self) -> None:
        """The same bitstream gives the same hash."""
        tree = SpikeDecisionTree()
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        n1 = tree.add_decision("n0", bs, 2)
        tree2 = SpikeDecisionTree()
        n2 = tree2.add_decision("n0", bs, 2)
        assert n1.bitstream_hash == n2.bitstream_hash

    def test_to_dict_structure(self) -> None:
        """A node exports its neuron and decision."""
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(4, dtype=np.uint8), 2)
        d = tree.to_dict()
        assert "neuron_id" in d
        assert "decision" in d
        assert d["decision"] == "spike"
