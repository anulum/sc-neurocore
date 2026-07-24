# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDegradationModel from former test_bridges_dna_mapper.py

"""Focused suite: TestDegradationModel from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403


class TestDegradationModel:
    def test_predict_concentration_decays(self) -> None:
        model = DegradationModel(half_life_hr=2.0)
        c0 = 100.0
        c1 = model.predict_concentration(c0, strand_length=20, time_hr=2.0)
        assert c1 < c0
        c2 = model.predict_concentration(c0, strand_length=20, time_hr=0.0)
        assert c2 == pytest.approx(c0)
