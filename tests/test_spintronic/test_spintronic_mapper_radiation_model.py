# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadiationModel from former test_spintronic_mapper.py

"""Focused suite: TestRadiationModel from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    RadiationModel,
)


class TestRadiationModel:
    """Radiation effects on the spintronic array."""

    def test_is_rad_hard(self) -> None:
        """The default radiation model is radiation hard."""
        rm = RadiationModel()
        assert rm.is_rad_hard

    def test_seu_rate(self) -> None:
        """The single-event upset rate is positive."""
        rm = RadiationModel()
        rate = rm.seu_rate(1e4, 1000)  # LEO flux
        assert rate > 0

    def test_tid_degradation(self) -> None:
        """The dose factor is 1 at 0 krad and 0.5 at 1000 krad."""
        rm = RadiationModel()
        assert rm.tid_degradation(0.0) == 1.0
        assert rm.tid_degradation(1000.0) == 0.5
