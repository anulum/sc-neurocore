# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiWellPlate from former test_experiment.py

"""Focused suite: TestMultiWellPlate from former test_experiment.py."""

from __future__ import annotations

from sc_neurocore.bioware.bioware import (
    MEAConfig,
    MultiWellPlate,
    WellConfig,
)


class TestMultiWellPlate:
    """Multi-well culture plates."""

    def test_standard_6_well(self) -> None:
        """The standard plate has six wells."""
        plate = MultiWellPlate.standard_6_well()
        assert plate.num_wells == 6

    def test_get_well(self) -> None:
        """A well is found by its id."""
        plate = MultiWellPlate.standard_6_well()
        w = plate.get_well("W1")
        assert w is not None
        assert w.well_id == "W1"

    def test_well_label(self) -> None:
        """A well label joins its id, tissue and age."""
        w = WellConfig(
            well_id="W1", mea_config=MEAConfig(), culture_type="hippocampal", passage_number=3
        )
        assert w.label == "W1_hippocampal_P3"

    def test_get_missing_well(self) -> None:
        """An unknown well id finds nothing."""
        plate = MultiWellPlate.standard_6_well()
        assert plate.get_well("W99") is None
