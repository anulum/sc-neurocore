# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveCheckpointInterval from former test_twinsync.py

"""Focused suite: TestAdaptiveCheckpointInterval from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403


class TestAdaptiveCheckpointInterval:
    def test_default(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        assert aci.current_interval == 1000

    def test_increases_on_low_rollbacks(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        aci.update(0, 1000)  # 0 rollbacks / 1000 events = 0%
        assert aci.current_interval >= 1000

    def test_decreases_on_high_rollbacks(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        aci.update(100, 1000)  # 10% rollback rate
        assert aci.current_interval < 1000

    def test_clamps_to_min(self):
        aci = AdaptiveCheckpointInterval(base_interval=200, min_interval=100)
        for _ in range(10):
            aci.update(999, 100)
        assert aci.current_interval >= 100

    def test_update_zero_events_keeps_interval(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        assert aci.update(5, 0) == 1000

    def test_is_aggressive_near_minimum(self):
        aci = AdaptiveCheckpointInterval(base_interval=200, min_interval=100)
        aci.update(100, 100)  # rollback rate 1.0 -> halve to the floor of 100
        assert aci.current_interval == 100
        assert aci.is_aggressive is True
