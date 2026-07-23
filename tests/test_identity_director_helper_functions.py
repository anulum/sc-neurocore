# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHelperFunctions from former test_identity_director.py

"""Focused suite: TestHelperFunctions from former test_identity_director.py."""

from __future__ import annotations

from tests.identity_director_support import *  # noqa: F403

class TestHelperFunctions:
    def test_add_weight_noise(self):
        data = np.array([0.0, 0.5, 0.3, 0.0, 0.8])
        original = data.copy()
        _add_weight_noise(data, scale=0.1)
        # Nonzero weights should change
        assert not np.allclose(data[1:3], original[1:3])
        # Weights should be non-negative
        assert np.all(data >= 0)
        # Zero weights should stay zero (no noise applied to zeros)
        assert data[0] == 0.0

    def test_homeostatic_scale(self):
        data = np.array([0.0, 0.1, 0.5, 0.9, 0.0])
        _homeostatic_scale(data, factor=0.5)
        # Weights should be pulled toward mean
        assert np.all(data >= 0)

    def test_homeostatic_scale_all_zero(self):
        data = np.zeros(5)
        _homeostatic_scale(data, factor=0.9)
        assert np.all(data == 0)

    def test_prune_weak(self):
        data = np.array([0.005, 0.5, 0.002, 0.8, 0.001])
        _prune_weak(data, threshold=0.01)
        assert data[0] == 0.0
        assert data[2] == 0.0
        assert data[4] == 0.0
        assert data[1] == 0.5
        assert data[3] == 0.8

    def test_grow_synapses(self):
        data = np.array([0.0, 0.5, 0.0, 0.0, 0.3])
        _grow_synapses(data, fraction=0.5, seed=42)
        # Some zeros should become positive
        grown = data[np.array([0, 2, 3])]
        assert np.any(grown > 0)

    def test_grow_synapses_no_zeros(self):
        data = np.array([0.1, 0.5, 0.3])
        _grow_synapses(data, fraction=0.5, seed=42)
