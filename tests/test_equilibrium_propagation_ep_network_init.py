# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPNetworkInit from former test_equilibrium_propagation.py

"""Focused suite: TestEPNetworkInit from former test_equilibrium_propagation.py."""

from __future__ import annotations

from tests.equilibrium_propagation_support import *  # noqa: F403

class TestEPNetworkInit:
    """Test network initialisation."""

    def test_creates_correct_layers(self) -> None:
        net = EPNetwork([10, 5, 3])
        assert len(net.weights) == 2
        assert net.weights[0].shape == (10, 5)
        assert net.weights[1].shape == (5, 3)

    def test_biases_zero_init(self) -> None:
        net = EPNetwork([4, 3, 2])
        np.testing.assert_array_equal(net.biases[0], np.zeros(3))
        np.testing.assert_array_equal(net.biases[1], np.zeros(2))

    def test_xavier_scale(self) -> None:
        # Xavier init should produce weights with moderate magnitude
        net = EPNetwork([100, 50, 10])
        for w in net.weights:
            assert abs(w.mean()) < 0.1
            assert w.std() < 0.5

    def test_deterministic_with_seed(self) -> None:
        net1 = EPNetwork([5, 3], rng_seed=42)
        net2 = EPNetwork([5, 3], rng_seed=42)
        np.testing.assert_array_equal(net1.weights[0], net2.weights[0])
