# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalShift from former test_memristor_mapper.py

"""Focused suite: TestThermalShift from former test_memristor_mapper.py."""

from __future__ import annotations

from memristor_mapper_support import *  # noqa: F403

class TestThermalShift:
    def test_higher_temp_shifts(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        g_hot = m.thermal_shift(g0, temp_c=85.0)
        assert g_hot != g0

    def test_ref_temp_no_shift(self) -> None:
        m = ConductanceModel(MemristorTechnology.GENERIC)
        g0 = 50e-6
        g_ref = m.thermal_shift(g0, temp_c=25.0)
        assert g_ref == pytest.approx(g0)
