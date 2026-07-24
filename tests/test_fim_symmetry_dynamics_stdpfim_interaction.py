# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTDPFIMInteraction from former test_fim_symmetry_dynamics.py

"""Focused suite: TestSTDPFIMInteraction from former test_fim_symmetry_dynamics.py."""

from __future__ import annotations

from tests.fim_symmetry_dynamics_support import *  # noqa: F403


class TestSTDPFIMInteraction:
    def test_fim_and_stdp_coexist(self):
        """Both FIM and STDP should run without error."""
        net, proj, mon = _make_self_connected_network(n=20, fim_lambda=3.0)
        net.run(duration=0.1, dt=0.001)
        assert mon.count >= 0  # just verify no crash

    def test_fim_does_not_kill_spikes(self):
        """FIM should not suppress all activity."""
        net, proj, mon = _make_self_connected_network(n=30, fim_lambda=5.0)
        net.run(duration=0.5, dt=0.001)
        assert mon.count > 0, "FIM suppressed all spikes"
