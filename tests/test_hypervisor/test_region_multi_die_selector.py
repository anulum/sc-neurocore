# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiDieSelector from former test_region.py

"""Focused suite: TestMultiDieSelector from former test_region.py."""

from __future__ import annotations

from sc_neurocore.hypervisor.region import (
    HWRegion,
    select_region_multi_die,
)


class TestMultiDieSelector:
    """Region selection across dies."""

    def test_prefers_die(self) -> None:
        """The selector prefers a region on the requested die."""
        regions = {
            0: HWRegion(0, 1024, 0, 0x1000, 0x1000, die_id=0),
            1: HWRegion(1, 1024, 0, 0x2000, 0x1000, die_id=1),
        }
        rid = select_region_multi_die(regions, 512, preferred_die=1)
        assert rid == 1

    def test_fallback_any_die(self) -> None:
        """Without a fit on the requested die the selector takes another."""
        regions = {
            0: HWRegion(0, 1024, 0, 0x1000, 0x1000, die_id=0),
        }
        rid = select_region_multi_die(regions, 512, preferred_die=5)
        assert rid == 0

    def test_no_fit(self) -> None:
        """No region that fits gives no selection."""
        regions = {
            0: HWRegion(0, 64, 0, 0x1000, 0x1000, die_id=0),
        }
        assert select_region_multi_die(regions, 512) is None
