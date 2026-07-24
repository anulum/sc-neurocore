# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLazarusPhaseGap from former test_fim_symmetry_dynamics.py

"""Focused suite: TestLazarusPhaseGap from former test_fim_symmetry_dynamics.py."""

from __future__ import annotations

from tests.fim_symmetry_dynamics_support import *  # noqa: F403


class TestLazarusPhaseGap:
    def test_activity_after_reset(self):
        """After resetting all populations, network should still produce
        spikes when driven — the structural weights survive reset."""
        net, proj, mon = _make_self_connected_network(n=20, fim_lambda=2.0)
        net.run(duration=0.1, dt=0.001)
        initial_count = mon.count

        # Reset populations (lose phase coherence, keep weights)
        for pop in net.populations:
            pop.reset_all()

        # Re-run with same drive
        mon2 = SpikeMonitor(net.populations[0], label="post_reset")
        net.spike_monitors.append(mon2)
        net.run(duration=0.1, dt=0.001)

        # Should still produce spikes (weights intact)
        assert mon2.count > 0, "no spikes after reset — weights lost"
