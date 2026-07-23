# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStateMonitor from former test_network_monitors_stimulus.py

"""Focused suite: TestStateMonitor from former test_network_monitors_stimulus.py."""

from __future__ import annotations

from tests.network_monitors_stimulus_support import *  # noqa: F403

class TestStateMonitor:
    def test_records_voltage(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        mon = StateMonitor(pop, variables=["v"])
        assert "v" in mon._data

    def test_default_variable_is_v(self):
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        mon = StateMonitor(pop)
        assert mon.variables == ["v"]
