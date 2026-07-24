# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalConstraints from former test_intelligence_power_and_thermal.py

"""Focused suite: TestThermalConstraints from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403


class TestThermalConstraints:
    """Tests for thermal constraint generation."""

    def test_basic_constraints(self):
        """Thermal constraints include derated clock."""
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        t = thermal_analysis(100.0, 500.0)
        xdc = generate_thermal_constraints("sc_lif", t)
        assert "create_clock" in xdc
        assert "Derated frequency" in xdc

    def test_hotspot_constraints(self):
        """High hotspot risk adds DSP spreading."""
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        t = thermal_analysis(100.0, 500.0, mul_count=25, dsp_columns=1)
        xdc = generate_thermal_constraints("sc_hh", t)
        assert "DSP spreading" in xdc

    def test_unsafe_warning(self):
        """Unsafe temperature adds warning."""
        from sc_neurocore.compiler.intelligence import (
            thermal_analysis,
            generate_thermal_constraints,
        )

        t = thermal_analysis(50000.0, 500.0)
        xdc = generate_thermal_constraints("sc_lif", t)
        assert "WARNING" in xdc
