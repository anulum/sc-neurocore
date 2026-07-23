# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTelemetryRing from former test_tinysc_ports.py

"""Focused suite: TestTelemetryRing from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403

class TestTelemetryRing:
    def test_empty_ring_defaults(self):
        ring = TelemetryRing(0)
        assert ring.capacity == 1
        assert ring.count == 0
        assert ring.mean() == 0.0
        assert ring.last() == 0

    def test_push_and_mean(self):
        ring = TelemetryRing(4)
        for v in [10, 20, 30, 40]:
            ring.push(v)
        assert ring.mean() == 25.0

    def test_last(self):
        ring = TelemetryRing(4)
        ring.push(42)
        assert ring.last() == 42

    def test_overflow(self):
        ring = TelemetryRing(2)
        for v in [1, 2, 3, 4, 5]:
            ring.push(v)
        assert ring.count == 2
        assert ring.last() == 5
