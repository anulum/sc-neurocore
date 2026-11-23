# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompensationLUT from former test_memristor_mapper.py

"""Focused suite: TestCompensationLUT from former test_memristor_mapper.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.memristor.memristor_mapper import (
    CompensationLUT,
    ConductanceModel,
    MemristorTechnology,
)


class TestCompensationLUT:
    """Per-device threshold compensation tables."""

    def test_build_nominal(self) -> None:
        """A table holds one threshold per conductance level."""
        m = ConductanceModel(MemristorTechnology.GENERIC)
        lut = CompensationLUT.build((0, 0), m)
        assert len(lut.compensated_thresholds) == m.num_levels

    def test_nominal_no_compensation(self) -> None:
        """A nominal device needs almost no compensation."""
        m = ConductanceModel(MemristorTechnology.GENERIC)
        lut = CompensationLUT.build((0, 0), m)
        assert lut.max_compensation < 0.01

    def test_measured_applies_compensation(self) -> None:
        """A measured deviation is compensated."""
        m = ConductanceModel(MemristorTechnology.GENERIC)
        measured = np.array([m.target_conductance(i) * 0.9 for i in range(m.num_levels)])
        lut = CompensationLUT.build((0, 0), m, measured)
        assert lut.max_compensation > 0.05

    def test_device_id_stored(self) -> None:
        """The table keeps its device position."""
        m = ConductanceModel(MemristorTechnology.GENERIC)
        lut = CompensationLUT.build((3, 7), m)
        assert lut.device_id == (3, 7)
