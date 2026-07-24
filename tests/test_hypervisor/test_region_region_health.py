# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegionHealth from former test_region.py

"""Focused suite: TestRegionHealth from former test_region.py."""

from __future__ import annotations

from region_support import *  # noqa: F403


class TestRegionHealth:
    def test_healthy_default(self) -> None:
        rh = RegionHealth(region_id=0)
        assert rh.health_score == pytest.approx(1.0)
        assert not rh.is_degraded

    def test_degraded_by_errors(self) -> None:
        rh = RegionHealth(region_id=0, error_count=5)
        assert rh.health_score < 0.8
        assert rh.is_degraded

    def test_temperature_penalty(self) -> None:
        rh = RegionHealth(region_id=0, temperature_c=100.0)
        assert rh.health_score < 1.0

    def test_record_error(self) -> None:
        rh = RegionHealth(region_id=0)
        rh.record_error()
        assert rh.error_count == 1
