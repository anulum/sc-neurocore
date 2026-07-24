# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSwarmCoupling from former test_bio_chaos_spatial_learning.py

"""Focused suite: TestSwarmCoupling from former test_bio_chaos_spatial_learning.py."""

from __future__ import annotations

from tests.bio_chaos_spatial_learning_support import *  # noqa: F403


class TestSwarmCoupling:
    @pytest.fixture()
    def agents(self):
        a = SCLearningLayer(n_inputs=4, n_neurons=3, base_seed=42)
        b = SCLearningLayer(n_inputs=4, n_neurons=3, base_seed=99)
        return a, b

    def test_synchronize_shifts_weights(self, agents):
        a, b = agents
        wa_before = a.get_weights().copy()
        SwarmCoupling(coupling_strength=0.5).synchronize(a, b)
        assert not np.array_equal(wa_before, a.get_weights())

    def test_mismatched_raises(self):
        a = SCLearningLayer(n_inputs=4, n_neurons=3, base_seed=1)
        b = SCLearningLayer(n_inputs=4, n_neurons=5, base_seed=2)
        with pytest.raises(ValueError, match="same size"):
            SwarmCoupling().synchronize(a, b)

    def test_zero_coupling_no_change(self, agents):
        a, b = agents
        wa_before = a.get_weights().copy()
        SwarmCoupling(coupling_strength=0.0).synchronize(a, b)
        np.testing.assert_array_equal(wa_before, a.get_weights())
