# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadiationModel from former test_spintronic_mapper.py

"""Focused suite: TestRadiationModel from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403

class TestRadiationModel:
    def test_is_rad_hard(self):
        rm = RadiationModel()
        assert rm.is_rad_hard

    def test_seu_rate(self):
        rm = RadiationModel()
        rate = rm.seu_rate(1e4, 1000)  # LEO flux
        assert rate > 0

    def test_tid_degradation(self):
        rm = RadiationModel()
        assert rm.tid_degradation(0.0) == 1.0
        assert rm.tid_degradation(1000.0) == 0.5
