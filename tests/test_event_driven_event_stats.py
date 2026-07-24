# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventStats from former test_event_driven.py

"""Focused suite: TestEventStats from former test_event_driven.py."""

from __future__ import annotations

from tests.event_driven_support import *  # noqa: F403


class TestEventStats:
    def test_summary(self):
        s = EventStats(total_events_processed=100, total_spikes_generated=10, max_queue_size=50)
        assert "100 events" in s.summary()
