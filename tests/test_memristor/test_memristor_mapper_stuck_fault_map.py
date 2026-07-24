# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStuckFaultMap from former test_memristor_mapper.py

"""Focused suite: TestStuckFaultMap from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403


class TestStuckFaultMap:
    def test_generate_faults(self) -> None:
        fm = StuckFaultMap.generate(100, 100, fault_rate=0.01, seed=42)
        assert fm.num_faults > 0
        assert fm.fault_rate > 0

    def test_is_stuck(self) -> None:
        fm = StuckFaultMap(10, 10, stuck_on=[(0, 0)], stuck_off=[(1, 1)])
        assert fm.is_stuck(0, 0) == "on"
        assert fm.is_stuck(1, 1) == "off"
        assert fm.is_stuck(5, 5) is None

    def test_zero_rate_no_faults(self) -> None:
        fm = StuckFaultMap.generate(10, 10, fault_rate=0.0, seed=42)
        assert fm.num_faults == 0

    def test_fault_rate_property(self) -> None:
        fm = StuckFaultMap(10, 10, stuck_on=[(0, 0)], stuck_off=[(1, 1)])
        assert fm.fault_rate == pytest.approx(0.02)
