# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHealthCheck from former test_wave4.py

"""Focused suite: TestHealthCheck from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestHealthCheck:
    def test_healthy(self):
        h = check_health(100, 10, 50, 1000)
        assert h.status == "healthy"
        assert h.events_per_sec == pytest.approx(10.0)

    def test_buffer_pressure(self):
        h = check_health(100, 10, 999, 1000)
        assert h.status == "buffer_pressure"
