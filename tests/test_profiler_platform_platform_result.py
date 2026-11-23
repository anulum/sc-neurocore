# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlatformResult from former test_profiler_platform.py

"""Focused suite: TestPlatformResult from former test_profiler_platform.py."""

from __future__ import annotations

from sc_neurocore.profiler import (
    PlatformResult,
)


class TestPlatformResult:
    """The platform comparison result record."""

    def test_fields(self) -> None:
        """A result is available with empty notes by default."""
        r = PlatformResult(
            platform="python",
            latency_ms=10.0,
            throughput_inf_per_s=100.0,
            power_mw=10000.0,
            energy_per_inf_nj=100000.0,
        )
        assert r.available is True
        assert r.notes == ""

    def test_unavailable(self) -> None:
        """A result can be marked unavailable."""
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
