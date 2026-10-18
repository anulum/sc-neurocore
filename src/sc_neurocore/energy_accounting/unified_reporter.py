# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Unified Energy Reporter

"""Combine runtime power, carbon estimates and thermal analysis in one report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from sc_neurocore.energy_accounting.sustainability_profiler import (
    CarbonModel,
    GridRegion,
    ThermalModel,
)


@dataclass
class UnifiedEnergyReport:
    """Combined report from runtime profiling + sustainability analysis."""

    total_power_mw: float = 0.0
    carbon_g_co2: float = 0.0
    junction_temp_c: float = 0.0
    ambient_temp_c: float = 25.0
    thermal_safe: bool = True
    asic_power_mw: float = 0.0
    grid_region: str = ""

    def summary(self) -> str:
        """Format the report's power, carbon and temperature values.

        Returns
        -------
        str
            Multiline summary, including ASIC power when its contribution is
            positive. Units are shown beside each reported value.
        """
        lines = [
            "Unified Energy Report",
            f"  Total power: {self.total_power_mw:.2f} mW",
            f"  Carbon: {self.carbon_g_co2:.6f} g CO₂",
            f"  Junction temp: {self.junction_temp_c:.1f} °C (safe: {self.thermal_safe})",
        ]
        if self.asic_power_mw > 0:
            lines.append(f"  ASIC power: {self.asic_power_mw:.2f} mW")
        return "\n".join(lines)


class UnifiedEnergyReporter:
    """Orchestrates power estimation → carbon accounting → thermal analysis.

    Parameters
    ----------
    region : GridRegion, optional
        Grid region used by the carbon model; defaults to the global profile.
    ambient_temp_c : float, optional
        Ambient temperature in degrees Celsius, defaulting to 25.
    asic_power_mw : float, optional
        ASIC power contribution in milliwatts, added to every analysis.

    The public analysis time argument is in seconds. The example accounts for
    one hour by passing 3600 seconds.

    Usage::

        from sc_neurocore.energy_accounting.sustainability_profiler import GridRegion
        from sc_neurocore.energy_accounting.unified_reporter import UnifiedEnergyReporter

        reporter = UnifiedEnergyReporter(region=GridRegion.EU)
        report = reporter.analyze(total_power_mw=150.0, inference_time_s=3600.0)
    """

    def __init__(
        self,
        region: GridRegion = GridRegion.GLOBAL,
        ambient_temp_c: float = 25.0,
        asic_power_mw: float = 0.0,
    ) -> None:
        """Configure the carbon and thermal models used by each analysis.

        Parameters
        ----------
        region : GridRegion, optional
            Grid region for the carbon model, defaulting to the global profile.
        ambient_temp_c : float, optional
            Ambient temperature in degrees Celsius, defaulting to 25.
        asic_power_mw : float, optional
            ASIC power contribution in milliwatts, defaulting to zero.
        """
        self.carbon_model = CarbonModel(region=region)
        self.thermal_model = ThermalModel(ambient_c=ambient_temp_c)
        self.asic_power_mw = asic_power_mw
        self.region = region

    def analyze(
        self,
        layer_configs: List[Dict[str, Any]] | None = None,
        total_power_mw: float = 0.0,
        inference_time_s: float = 0.001,
    ) -> UnifiedEnergyReport:
        """Combine supplied power contributions for the requested elapsed time.

        Parameters
        ----------
        layer_configs : list of dict, optional
            Layer records whose optional ``power_mw`` values are added to the
            base power. Omitted or empty records contribute no layer power.
        total_power_mw : float, optional
            Base power in milliwatts, before layer and ASIC contributions.
        inference_time_s : float, optional
            Elapsed time in seconds for carbon accounting; defaults to 0.001.
            An hour is represented by 3600 seconds.

        Returns
        -------
        UnifiedEnergyReport
            Combined power in milliwatts, carbon in grams of CO2, junction and
            ambient temperatures in degrees Celsius, and thermal-limit status.
        """
        if layer_configs:
            total_power_mw += sum(cfg.get("power_mw", 0.0) for cfg in layer_configs)
        total_power_mw += self.asic_power_mw

        duration_h = inference_time_s / 3600.0
        carbon_g = self.carbon_model.compute(total_power_mw, duration_h)
        junction_c = self.thermal_model.junction_temp(total_power_mw)
        safe = self.thermal_model.is_safe(total_power_mw)

        return UnifiedEnergyReport(
            total_power_mw=total_power_mw,
            carbon_g_co2=carbon_g,
            junction_temp_c=junction_c,
            ambient_temp_c=self.thermal_model.ambient_c,
            thermal_safe=safe,
            asic_power_mw=self.asic_power_mw,
            grid_region=self.region.value,
        )
