# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveCheckpointInterval from former test_twinsync.py

"""Focused suite: TestAdaptiveCheckpointInterval from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    AdaptiveCheckpointInterval,
)


class TestAdaptiveCheckpointInterval:
    """Adapting the checkpoint interval to rollbacks."""

    def test_default(self) -> None:
        """The interval starts at the base interval."""
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        assert aci.current_interval == 1000

    def test_increases_on_low_rollbacks(self) -> None:
        """No rollbacks keep the interval at or above the base."""
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        aci.update(0, 1000)  # 0 rollbacks / 1000 events = 0%
        assert aci.current_interval >= 1000

    def test_decreases_on_high_rollbacks(self) -> None:
        """Frequent rollbacks shorten the interval."""
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        aci.update(100, 1000)  # 10% rollback rate
        assert aci.current_interval < 1000

    def test_clamps_to_min(self) -> None:
        """The interval never falls below its minimum."""
        aci = AdaptiveCheckpointInterval(base_interval=200, min_interval=100)
        for _ in range(10):
            aci.update(999, 100)
        assert aci.current_interval >= 100

    def test_update_zero_events_keeps_interval(self) -> None:
        """An update over zero events keeps the interval."""
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        assert aci.update(5, 0) == 1000

    def test_is_aggressive_near_minimum(self) -> None:
        """An interval at its minimum is reported aggressive."""
        aci = AdaptiveCheckpointInterval(base_interval=200, min_interval=100)
        aci.update(100, 100)  # rollback rate 1.0 -> halve to the floor of 100
        assert aci.current_interval == 100
        assert aci.is_aggressive is True
