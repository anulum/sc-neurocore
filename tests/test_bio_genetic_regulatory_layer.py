# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGeneticRegulatoryLayer from former test_bio.py

"""Focused suite: TestGeneticRegulatoryLayer from former test_bio.py."""

from __future__ import annotations

from tests.bio_support import *  # noqa: F403

class TestGeneticRegulatoryLayer:
    def test_initial_protein_zero(self):
        grn = GeneticRegulatoryLayer(n_neurons=10)
        np.testing.assert_array_equal(grn.protein_levels, 0.0)

    def test_spike_increases_protein(self):
        grn = GeneticRegulatoryLayer(n_neurons=5, production_rate=0.1)
        spikes = np.array([1, 0, 1, 0, 1], dtype=np.float64)
        grn.step(spikes)
        assert grn.protein_levels[0] > 0
        assert grn.protein_levels[1] == 0

    def test_decay_without_spikes(self):
        grn = GeneticRegulatoryLayer(n_neurons=3, production_rate=0.1, decay_rate=0.05)
        grn.protein_levels = np.array([1.0, 1.0, 1.0])
        grn.step(np.zeros(3))
        assert np.all(grn.protein_levels < 1.0)

    def test_protein_bounded(self):
        grn = GeneticRegulatoryLayer(n_neurons=2, production_rate=1.0)
        for _ in range(1000):
            grn.step(np.ones(2))
        assert np.all(grn.protein_levels <= 10.0)

    def test_modulators(self):
        grn = GeneticRegulatoryLayer(n_neurons=3)
        grn.protein_levels = np.array([0.5, 1.0, 0.0])
        m = grn.get_threshold_modulators()
        np.testing.assert_array_equal(m, grn.protein_levels)
