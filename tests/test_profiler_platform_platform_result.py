# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlatformResult from former test_profiler_platform.py

"""Focused suite: TestPlatformResult from former test_profiler_platform.py."""

from __future__ import annotations

from tests.profiler_platform_support import *  # noqa: F403

class TestPlatformResult:
    def test_fields(self):
        r = PlatformResult(
            platform="python",
            latency_ms=10.0,
            throughput_inf_per_s=100.0,
            power_mw=10000.0,
            energy_per_inf_nj=100000.0,
        )
        assert r.available is True
        assert r.notes == ""

    def test_unavailable(self):
        r = PlatformResult(
            platform="custom",
            latency_ms=0,
            throughput_inf_per_s=0,
            power_mw=0,
            energy_per_inf_nj=0,
            available=False,
            notes="Not installed",
        )
        assert r.available is False
