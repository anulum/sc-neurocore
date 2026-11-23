# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDefectMap from former test_spintronic_mapper.py

"""Focused suite: TestDefectMap from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    DefectMap,
)


class TestDefectMap:
    """Recording and remapping defective cells."""

    def test_add_and_count(self) -> None:
        """An added defect is counted and reported defective."""
        dm = DefectMap()
        dm.add_defect(0, 3, "stuck_p")
        assert dm.defect_count == 1
        assert dm.is_defective(0, 3)

    def test_remap(self) -> None:
        """A remapped defect resolves to its spare address."""
        dm = DefectMap()
        dm.add_defect(0, 3, "stuck_p")
        dm.add_remap((0, 3), (7, 0))
        assert dm.effective_address(0, 3) == (7, 0)

    def test_defect_rate(self) -> None:
        """One defect in 100 cells is a rate of 0.01."""
        dm = DefectMap()
        dm.add_defect(0, 0, "open")
        assert dm.defect_rate(100) == 0.01

    def test_defect_rate_zero_cells(self) -> None:
        """An array of zero cells has a defect rate of 0."""
        dm = DefectMap()
        dm.add_defect(0, 0, "open")
        assert dm.defect_rate(0) == 0.0
