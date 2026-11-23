# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpintronicCell from former test_spintronic_mapper.py

"""Focused suite: TestSpintronicCell from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    SpintronicCell,
    SpintronicDeviceConfig,
    SpintronicTech,
)


class TestSpintronicCell:
    """Resistance of a single spintronic cell."""

    def test_resistance_p(self) -> None:
        """A parallel cell has the device's parallel resistance."""
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev, state=0)
        r_p = cell.resistance_ohm
        assert r_p == dev.parallel_resistance_ohm

    def test_resistance_ap(self) -> None:
        """An antiparallel cell exceeds 5000 ohm."""
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        cell = SpintronicCell(0, 0, dev, state=1)
        r_ap = cell.resistance_ohm
        assert r_ap > 5000.0

    def test_tmr_ratio(self) -> None:
        """The cell's resistances give the device TMR ratio."""
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        p = SpintronicCell(0, 0, dev, state=0)
        ap = SpintronicCell(0, 0, dev, state=1)
        ratio = (ap.resistance_ohm - p.resistance_ohm) / p.resistance_ohm
        assert abs(ratio - dev.tmr_ratio) < 0.01

    def test_resistance_uses_device_parallel_resistance(self) -> None:
        """Both states scale from the device's parallel resistance."""
        dev = SpintronicDeviceConfig(
            parallel_resistance_ohm=7_500.0,
            tmr_ratio=2.0,
        )
        p = SpintronicCell(0, 0, dev, state=0)
        ap = SpintronicCell(0, 0, dev, state=1)
        assert p.resistance_ohm == 7_500.0
        assert ap.resistance_ohm == 22_500.0
