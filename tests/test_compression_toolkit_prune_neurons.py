# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPruneNeurons from former test_compression_toolkit.py

"""Focused suite: TestPruneNeurons from former test_compression_toolkit.py."""

from __future__ import annotations

from tests.compression_toolkit_support import *  # noqa: F403

class TestPruneNeurons:
    def test_structural_pruning_by_weight_norm(self):
        w1 = np.array([[1.0, 0.5], [0.0001, 0.0001], [0.8, 0.3]])
        w2 = np.array([[0.5, 0.3, 0.1]])
        pruned, report = prune_neurons([w1, w2], activity_threshold=0.001)
        assert report.pruned_neurons == 1
        assert pruned[0].shape[0] == 2
        assert pruned[1].shape[1] == 2

    def test_no_pruning_when_all_active(self):
        w = [np.ones((3, 3))]
        pruned, report = prune_neurons(w, activity_threshold=0.001)
        assert report.pruned_neurons == 0
        np.testing.assert_array_equal(pruned[0], w[0])

    def test_with_firing_rates(self):
        w = [np.ones((4, 3))]
        rates = [np.array([0.1, 0.0, 0.05, 0.0002])]
        pruned, report = prune_neurons(w, firing_rates=rates, activity_threshold=0.001)
        # Neurons with rate <= 0.001 are pruned: index 1 (0.0) and index 3 (0.0002)
        assert report.pruned_neurons == 2
        assert pruned[0].shape[0] == 2

    def test_report_fields(self):
        w = [np.eye(3)]
        _, report = prune_neurons(w)
        assert report.original_neurons == 3
        assert report.original_params > 0
