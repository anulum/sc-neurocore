# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSeuScrubberFallback from former test_intelligence_digital_twin.py

"""Focused suite: TestSeuScrubberFallback from former test_intelligence_digital_twin.py."""

from __future__ import annotations

from tests.intelligence_digital_twin_support import *  # noqa: F403

class TestSeuScrubberFallback:
    """No configuration bits means no expected upsets, so the scrub interval
    falls back to the daily cadence rather than dividing by zero."""

    def test_zero_config_bits_uses_daily_fallback_interval(self):
        from sc_neurocore.compiler.intelligence import schedule_seu_scrubbing

        s = schedule_seu_scrubbing(0, orbit_altitude_km=400)
        assert s.interval_ms == round(24.0 * 3_600_000, 2)
        assert s.expected_seu_rate == 0.0
