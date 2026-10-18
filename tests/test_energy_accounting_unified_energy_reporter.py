# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUnifiedEnergyReporter from former test_energy_accounting.py

"""Focused suite: TestUnifiedEnergyReporter from former test_energy_accounting.py."""

from __future__ import annotations

import doctest
import inspect
import textwrap

import pytest

from sc_neurocore.energy_accounting.sustainability_profiler import GridRegion
from sc_neurocore.energy_accounting.unified_reporter import (
    UnifiedEnergyReport,
    UnifiedEnergyReporter,
)


class TestUnifiedEnergyReporter:
    """Exercise public report formatting, power composition and documented usage."""

    def test_summary_includes_asic_line_conditionally(self) -> None:
        """Include the ASIC summary line only when an ASIC contribution exists."""
        no_asic = UnifiedEnergyReport(total_power_mw=5.0, carbon_g_co2=0.01, junction_temp_c=30.0)
        with_asic = UnifiedEnergyReport(
            total_power_mw=7.0,
            carbon_g_co2=0.02,
            junction_temp_c=31.0,
            asic_power_mw=2.0,
        )
        assert "ASIC power" not in no_asic.summary()
        assert "ASIC power" in with_asic.summary()

    def test_analyse_adds_layer_and_asic_power(self) -> None:
        """Retain base, layer and ASIC contributions in the combined report."""
        reporter = UnifiedEnergyReporter(asic_power_mw=3.0)
        report = reporter.analyze(
            layer_configs=[{"power_mw": 2.0}, {"power_mw": 1.0}],
            total_power_mw=4.0,
            inference_time_s=0.5,
        )
        assert report.total_power_mw == pytest.approx(10.0)
        assert report.grid_region
        assert isinstance(report.thermal_safe, bool)

    def test_analyse_without_layer_configs_uses_total_and_asic_only(self) -> None:
        """Accept omitted layer configuration while retaining the ASIC contribution."""
        reporter = UnifiedEnergyReporter(asic_power_mw=2.5)
        report = reporter.analyze(
            layer_configs=None,
            total_power_mw=5.5,
            inference_time_s=1.0,
        )
        assert report.total_power_mw == pytest.approx(8.0)
        assert report.asic_power_mw == pytest.approx(2.5)
        assert report.grid_region

    def test_documented_usage_runs_against_public_reporter(self) -> None:
        """Run the actual class documentation example through the public interface."""
        doc = inspect.getdoc(UnifiedEnergyReporter)
        assert doc is not None
        usage = textwrap.dedent(doc.split("Usage::\n\n", 1)[1]).strip()
        prompted = "\n".join(">>> " + line for line in usage.splitlines() if line.strip())
        example = doctest.DocTestParser().get_doctest(
            prompted,
            {"UnifiedEnergyReporter": UnifiedEnergyReporter, "GridRegion": GridRegion},
            "UnifiedEnergyReporter.usage",
            inspect.getfile(UnifiedEnergyReporter),
            0,
        )

        result = doctest.DebugRunner().run(example, clear_globs=False)

        assert result.failed == 0
        assert result.attempted >= 2
        report = example.globs["report"]
        assert isinstance(report, UnifiedEnergyReport)
        assert report.total_power_mw == pytest.approx(150.0)
        assert report.grid_region == GridRegion.EU.value
