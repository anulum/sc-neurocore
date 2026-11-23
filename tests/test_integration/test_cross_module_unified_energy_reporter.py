# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUnifiedEnergyReporter from former test_cross_module.py

"""Focused suite: TestUnifiedEnergyReporter from former test_cross_module.py."""

from __future__ import annotations


class TestUnifiedEnergyReporter:
    """Verify Sustainability ↔ Profiling integration."""

    def test_basic_analysis(self) -> None:
        """A one-layer analysis reports at least its layer power under the unified title."""
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter
        from sc_neurocore.energy_accounting.sustainability_profiler import GridRegion

        reporter = UnifiedEnergyReporter(region=GridRegion.EU)
        report = reporter.analyze(
            layer_configs=[{"name": "L0", "power_mw": 10.0}],
            inference_time_s=0.001,
        )
        assert report.summary().startswith("Unified Energy Report")
        assert report.total_power_mw >= 10.0

    def test_asic_power_included(self) -> None:
        """The ASIC power is reported and counted in the total."""
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter

        reporter = UnifiedEnergyReporter(asic_power_mw=500.0)
        report = reporter.analyze(inference_time_s=0.001)
        assert report.asic_power_mw == 500.0
        assert report.total_power_mw >= 500.0

    def test_thermal_check(self) -> None:
        """A 100 mW total gives a junction above ambient that stays thermally safe."""
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter

        reporter = UnifiedEnergyReporter()
        report = reporter.analyze(total_power_mw=100.0)
        assert report.junction_temp_c > 25.0
        assert report.thermal_safe
