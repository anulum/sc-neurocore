# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalAnalysis from former test_intelligence_power_and_thermal.py

"""Focused suite: TestThermalAnalysis from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403

class TestThermalAnalysis:
    """Tests for thermal estimation and derating."""

    def test_basic_thermal(self):
        """Basic thermal analysis returns valid fields."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(100.0, 500.0)
        assert t.junction_temp_c > 25.0
        assert t.derated_freq_mhz > 0
        assert t.thermal_safe
        assert t.hotspot_risk in ("none", "low", "medium", "high")

    def test_low_power_safe(self):
        """Low power design is thermally safe."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(0.1, 100.0)
        assert t.thermal_safe
        assert t.delta_t_c < 1.0

    def test_high_power_derating(self):
        """High power causes frequency derating."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(10000.0, 500.0)  # 10W
        assert t.junction_temp_c > 85.0
        assert t.derated_freq_mhz < 500.0

    def test_extreme_power_unsafe(self):
        """Extreme power exceeds junction limit."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(50000.0, 500.0)  # 50W
        assert not t.thermal_safe

    def test_dsp_hotspot(self):
        """Many DSPs in one column → high hotspot risk."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=1)
        assert t.hotspot_risk == "high"

    def test_hotspot_concentration_derates_frequency(self):
        """Concentrated DSP hotspots should affect timing, not only labels."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        spread = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=10)
        concentrated = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=1)

        assert concentrated.hotspot_risk == "high"
        assert concentrated.derated_freq_mhz < spread.derated_freq_mhz

    def test_dsp_hotspot_adds_local_junction_rise(self):
        """DSP hotspot power should increase junction temperature, not only risk labels."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        spread = thermal_analysis(
            1000.0,
            500.0,
            mul_count=32,
            dsp_columns=8,
            dsp_power_mw=320.0,
            theta_spreading=12.0,
        )
        concentrated = thermal_analysis(
            1000.0,
            500.0,
            mul_count=32,
            dsp_columns=1,
            dsp_power_mw=320.0,
            theta_spreading=12.0,
        )

        assert concentrated.hotspot_delta_t_c > spread.hotspot_delta_t_c
        assert concentrated.junction_temp_c > spread.junction_temp_c

    def test_dsp_spread(self):
        """DSPs spread across columns → lower risk."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t = thermal_analysis(100.0, 500.0, mul_count=30, dsp_columns=10)
        assert t.hotspot_risk in ("none", "low")

    def test_small_process_more_derating(self):
        """7nm process derates more than 28nm."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        t7 = thermal_analysis(5000.0, 500.0, process_nm=7)
        t28 = thermal_analysis(5000.0, 500.0, process_nm=28)
        assert t7.derated_freq_mhz < t28.derated_freq_mhz

    def test_rejects_invalid_physical_inputs(self):
        """Thermal analysis must reject non-physical parameters."""
        from sc_neurocore.compiler.intelligence import thermal_analysis

        invalid_cases = [
            ({"estimated_power_mw": -1.0, "target_freq_mhz": 500.0}, "estimated_power_mw"),
            ({"estimated_power_mw": 1.0, "target_freq_mhz": 0.0}, "target_freq_mhz"),
            ({"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "theta_ja": 0.0}, "theta_ja"),
            (
                {"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "process_nm": 0},
                "process_nm",
            ),
            (
                {"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "mul_count": -1},
                "mul_count",
            ),
            (
                {"estimated_power_mw": 1.0, "target_freq_mhz": 500.0, "dsp_columns": 0},
                "dsp_columns",
            ),
        ]
        for kwargs, message in invalid_cases:
            with pytest.raises(ValueError, match=message):
                thermal_analysis(**kwargs)
