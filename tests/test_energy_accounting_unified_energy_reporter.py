# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUnifiedEnergyReporter from former test_energy_accounting.py

"""Focused suite: TestUnifiedEnergyReporter from former test_energy_accounting.py."""

from __future__ import annotations

from tests.energy_accounting_support import *  # noqa: F403

class TestUnifiedEnergyReporter:
    def test_summary_includes_asic_line_conditionally(self):
        no_asic = UnifiedEnergyReport(total_power_mw=5.0, carbon_g_co2=0.01, junction_temp_c=30.0)
        with_asic = UnifiedEnergyReport(
            total_power_mw=7.0,
            carbon_g_co2=0.02,
            junction_temp_c=31.0,
            asic_power_mw=2.0,
        )
        assert "ASIC power" not in no_asic.summary()
        assert "ASIC power" in with_asic.summary()

    def test_analyse_adds_layer_and_asic_power(self):
        reporter = UnifiedEnergyReporter(asic_power_mw=3.0)
        report = reporter.analyze(
            layer_configs=[{"power_mw": 2.0}, {"power_mw": 1.0}],
            total_power_mw=4.0,
            inference_time_s=0.5,
        )
        assert report.total_power_mw == pytest.approx(10.0)
        assert report.grid_region
        assert isinstance(report.thermal_safe, bool)

    def test_analyse_without_layer_configs_uses_total_and_asic_only(self):
        reporter = UnifiedEnergyReporter(asic_power_mw=2.5)
        report = reporter.analyze(
            layer_configs=None,
            total_power_mw=5.5,
            inference_time_s=1.0,
        )
        assert report.total_power_mw == pytest.approx(8.0)
        assert report.asic_power_mw == pytest.approx(2.5)
        assert report.grid_region
