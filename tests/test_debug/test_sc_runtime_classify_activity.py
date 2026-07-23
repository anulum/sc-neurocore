# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestClassifyActivity from former test_sc_runtime.py

"""Focused suite: TestClassifyActivity from former test_sc_runtime.py."""

from __future__ import annotations

from sc_runtime_support import *  # noqa: F403

class TestClassifyActivity:
    def test_idle(self):
        assert classify_activity(0.005) == ActivityZone.IDLE

    def test_low(self):
        assert classify_activity(0.03) == ActivityZone.LOW

    def test_normal(self):
        assert classify_activity(0.3) == ActivityZone.NORMAL

    def test_high(self):
        assert classify_activity(0.8) == ActivityZone.HIGH

    def test_burst(self):
        assert classify_activity(0.99) == ActivityZone.BURST

    def test_boundary_idle_low(self):
        assert classify_activity(0.01) == ActivityZone.LOW

    def test_boundary_low_normal(self):
        assert classify_activity(0.05) == ActivityZone.NORMAL
