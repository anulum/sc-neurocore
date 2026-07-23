# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeEvent from former test_event_driven.py

"""Focused suite: TestSpikeEvent from former test_event_driven.py."""

from __future__ import annotations

from tests.event_driven_support import *  # noqa: F403

class TestSpikeEvent:
    def test_ordering(self):
        e1 = SpikeEvent(time=1.0, source_id=0, target_id=1)
        e2 = SpikeEvent(time=2.0, source_id=0, target_id=1)
        assert e1 < e2
