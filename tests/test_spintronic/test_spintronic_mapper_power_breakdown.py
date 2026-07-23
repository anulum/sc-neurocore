# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerBreakdown from former test_spintronic_mapper.py

"""Focused suite: TestPowerBreakdown from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403

class TestPowerBreakdown:
    def test_power_breakdown_keys(self):
        arr = SpintronicArray(2, 2)
        pb = arr.power_breakdown(bitstream_length=128)
        assert "switching_fj" in pb
        assert "leakage_fj" in pb
        assert "total_fj" in pb

    def test_total_equals_sum(self):
        arr = SpintronicArray(4, 4)
        pb = arr.power_breakdown(256)
        assert abs(pb["total_fj"] - pb["switching_fj"] - pb["leakage_fj"]) < 1e-6

    def test_longer_bitstream_more_energy(self):
        arr = SpintronicArray(4, 4)
        pb_short = arr.power_breakdown(128)
        pb_long = arr.power_breakdown(512)
        assert pb_long["total_fj"] > pb_short["total_fj"]

    def test_energy_positive(self):
        arr = SpintronicArray(2, 2)
        pb = arr.power_breakdown(256)
        assert pb["switching_fj"] > 0
        assert pb["leakage_fj"] > 0
