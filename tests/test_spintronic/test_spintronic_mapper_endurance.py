# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEndurance from former test_spintronic_mapper.py

"""Focused suite: TestEndurance from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    SpintronicDeviceConfig,
    SpintronicTech,
)


class TestEndurance:
    """Write endurance of the device technologies."""

    def test_endurance_positive(self) -> None:
        """Every technology has a positive endurance."""
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.endurance_cycles > 0

    def test_sot_higher_than_stt(self) -> None:
        """SOT-MRAM endures at least as many cycles as STT-MRAM."""
        sot = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        stt = SpintronicDeviceConfig.from_tech(SpintronicTech.STT_MTJ)
        assert sot.endurance_cycles >= stt.endurance_cycles
