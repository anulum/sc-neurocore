# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReadDisturb from former test_spintronic_mapper.py

"""Focused suite: TestReadDisturb from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    SpintronicDeviceConfig,
    SpintronicTech,
)


class TestReadDisturb:
    """The probability that a read flips a cell."""

    def test_read_disturb_low(self) -> None:
        """The read-disturb probability is below 1."""
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        assert cfg.read_disturb_probability < 1.0

    def test_read_disturb_nonnegative(self) -> None:
        """The read-disturb probability is not negative."""
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.read_disturb_probability >= 0.0
