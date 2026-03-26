# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for robotics CPG module

from sc_neurocore.robotics.cpg import StochasticCPG


class TestStochasticCPG:
    def test_construction(self):
        cpg = StochasticCPG()
        assert cpg.drive_current == 2.0
        assert cpg.inhibition_weight == 2.0

    def test_step_returns_binary_tuple(self):
        cpg = StochasticCPG()
        s1, s2 = cpg.step()
        assert s1 in (0, 1)
        assert s2 in (0, 1)

    def test_produces_spikes_from_both_neurons(self):
        cpg = StochasticCPG()
        s1_spikes, s2_spikes = 0, 0
        for _ in range(200):
            s1, s2 = cpg.step()
            s1_spikes += s1
            s2_spikes += s2
        assert s1_spikes > 0
        assert s2_spikes > 0

    def test_alternating_pattern(self):
        cpg = StochasticCPG()
        pairs = [cpg.step() for _ in range(500)]
        # Count timesteps where both neurons fire simultaneously
        both_fire = sum(1 for s1, s2 in pairs if s1 == 1 and s2 == 1)
        either_fire = sum(1 for s1, s2 in pairs if s1 == 1 or s2 == 1)
        # Mutual inhibition should reduce simultaneous firing
        # (both_fire / either_fire should be less than without inhibition)
        if either_fire > 0:
            co_fire_ratio = both_fire / either_fire
            assert co_fire_ratio < 0.9

    def test_custom_parameters(self):
        cpg = StochasticCPG(drive_current=3.0, inhibition_weight=4.0)
        assert cpg.drive_current == 3.0
        s1, s2 = cpg.step()
        assert s1 in (0, 1) and s2 in (0, 1)

    def test_trace_decay(self):
        cpg = StochasticCPG()
        cpg.step()
        trace_after_step = cpg.s1_trace + cpg.s2_trace
        # After many steps without spikes (impossible to force, but trace should stay bounded)
        for _ in range(100):
            cpg.step()
        # Traces should not explode
        assert abs(cpg.s1_trace) < 100
        assert abs(cpg.s2_trace) < 100
