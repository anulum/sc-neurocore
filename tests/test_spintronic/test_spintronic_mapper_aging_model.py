# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAgingModel from former test_spintronic_mapper.py

"""Focused suite: TestAgingModel from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403

class TestAgingModel:
    def test_no_degradation_initially(self):
        am = AgingModel()
        assert am.tmr_degradation(1.5, 10**12) == 1.5

    def test_degradation_with_cycles(self):
        am = AgingModel(cycles_written=10**12)
        tmr = am.tmr_degradation(1.5, 10**12)
        assert tmr < 1.5

    def test_write_increments(self):
        am = AgingModel()
        am.write(100)
        assert am.cycles_written == 100

    def test_tmr_degradation_zero_endurance_is_identity(self):
        assert AgingModel(cycles_written=10).tmr_degradation(1.5, 0) == 1.5

    def test_stability_degradation_zero_endurance_is_identity(self):
        assert AgingModel(cycles_written=10).stability_degradation(2.0, 0) == 2.0

    def test_stability_degradation_with_cycles(self):
        degraded = AgingModel(cycles_written=10**12).stability_degradation(2.0, 10**12)
        assert degraded < 2.0

    def test_is_worn_out_flag(self):
        am = AgingModel(cycles_written=10)
        assert isinstance(am.is_worn_out, bool)
