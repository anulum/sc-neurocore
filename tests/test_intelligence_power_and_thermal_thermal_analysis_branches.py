# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalAnalysisBranches from former test_intelligence_power_and_thermal.py

"""Focused suite: TestThermalAnalysisBranches from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

import pytest


class TestThermalAnalysisBranches:
    """Cover the thermal branches the existing cases leave untouched.

    These are the per-process-node derating tiers, the medium and low hotspot
    bands, and the finite-input guard.
    """

    def test_16nm_node_derates_less_than_7nm(self) -> None:
        """A 16 nm node derates less than 7 nm."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        # 16 nm takes the >7, <=16 derating tier (x0.99), distinct from 7 nm (x0.98),
        # so with identical load the 16 nm result keeps a higher derated frequency.
        t16 = thermal_analysis(100.0, 500.0, process_nm=16)
        t7 = thermal_analysis(100.0, 500.0, process_nm=7)
        assert t16.derated_freq_mhz > t7.derated_freq_mhz

    def test_medium_hotspot_band(self) -> None:
        """A medium power density is a medium hotspot risk."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        # 150 muls across 10 columns -> 15 per column, inside the (10, 20] band.
        t = thermal_analysis(100.0, 500.0, mul_count=150, dsp_columns=10)
        assert t.hotspot_risk == "medium"

    def test_low_hotspot_band(self) -> None:
        """A low power density is a low hotspot risk."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        # 50 muls across 10 columns -> 5 per column, inside the (4, 10] band.
        t = thermal_analysis(100.0, 500.0, mul_count=50, dsp_columns=10)
        assert t.hotspot_risk == "low"

    def test_non_finite_ambient_is_rejected(self) -> None:
        """A non-finite ambient temperature is rejected."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        with pytest.raises(ValueError, match="t_ambient_c must be finite"):
            thermal_analysis(100.0, 500.0, t_ambient_c=float("nan"))
