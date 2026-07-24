# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHWRegion from former test_region.py

"""Focused suite: TestHWRegion from former test_region.py."""

from __future__ import annotations

from region_support import *  # noqa: F403


class TestHWRegion:
    def test_defaults(self) -> None:
        r = _region()
        assert r.is_free
        assert r.state == RegionState.FREE

    def test_axi_end_addr(self) -> None:
        r = _region(base=0x1000)
        assert r.axi_end_addr == 0x1000 + 0x1000

    def test_contains_addr(self) -> None:
        r = _region(base=0x1000)
        assert r.contains_addr(0x1000) is True
        assert r.contains_addr(0x1FFF) is True
        assert r.contains_addr(0x2000) is False
        assert r.contains_addr(0x0FFF) is False
