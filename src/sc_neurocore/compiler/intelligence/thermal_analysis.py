# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Thermal analysis

"""Thermal impact estimation and temperature-aware timing derating."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ThermalEstimate:
    """Thermal analysis result for a compiled neuron.

    Attributes
    ----------
    power_mw : float
        Estimated total power in milliwatts.
    delta_t_c : float
        Estimated temperature rise in °C.
    junction_temp_c : float
        Estimated junction temperature.
    hotspot_delta_t_c : float
        Local temperature rise from concentrated DSP power.
    derated_freq_mhz : float
        Frequency after thermal derating.
    thermal_safe : bool
        True if junction temp is within limits.
    hotspot_risk : str
        ``"none"``, ``"low"``, ``"medium"``, ``"high"``.
    """

    power_mw: float
    delta_t_c: float
    junction_temp_c: float
    hotspot_delta_t_c: float
    derated_freq_mhz: float
    thermal_safe: bool
    hotspot_risk: str


@dataclass
class ThermalEnvelopeEstimate:
    """Junction temperature estimate.

    Attributes
    ----------
    power_mw : float
        Estimated power dissipation (mW).
    theta_ja : float
        Junction-to-ambient thermal resistance (°C/W).
    t_ambient : float
        Ambient temperature (°C).
    t_junction : float
        Estimated junction temperature (°C).
    thermal_margin : float
        Margin to max T_j (°C).
    pass_fail : str
        ``"PASS"`` or ``"FAIL"``.
    """

    power_mw: float
    theta_ja: float
    t_ambient: float
    t_junction: float
    thermal_margin: float
    pass_fail: str


def thermal_analysis(
    estimated_power_mw: float,
    target_freq_mhz: float,
    *,
    theta_ja: float = 11.5,
    t_ambient_c: float = 25.0,
    t_junction_max_c: float = 100.0,
    process_nm: int = 28,
    mul_count: int = 0,
    dsp_columns: int = 1,
    dsp_power_mw: float | None = None,
    theta_spreading: float = 0.0,
) -> ThermalEstimate:
    """Estimate thermal impact and frequency derating."""
    _require_finite_non_negative(estimated_power_mw, "estimated_power_mw")
    _require_finite_positive(target_freq_mhz, "target_freq_mhz")
    _require_finite_positive(theta_ja, "theta_ja")
    _require_finite(t_ambient_c, "t_ambient_c")
    _require_finite_positive(t_junction_max_c, "t_junction_max_c")
    if not isinstance(process_nm, int) or process_nm <= 0:
        raise ValueError("process_nm must be a positive integer")
    if not isinstance(mul_count, int) or mul_count < 0:
        raise ValueError("mul_count must be a non-negative integer")
    if not isinstance(dsp_columns, int) or dsp_columns <= 0:
        raise ValueError("dsp_columns must be a positive integer")
    if dsp_power_mw is not None:
        _require_finite_non_negative(dsp_power_mw, "dsp_power_mw")
    _require_finite_non_negative(theta_spreading, "theta_spreading")

    power_w = estimated_power_mw / 1000.0
    delta_t = power_w * theta_ja
    hotspot_delta_t = 0.0
    if dsp_power_mw is not None and mul_count > 0 and theta_spreading > 0.0:
        dsp_power_per_column_w = (dsp_power_mw / 1000.0) / dsp_columns
        hotspot_delta_t = dsp_power_per_column_w * theta_spreading
    t_junction = t_ambient_c + delta_t + hotspot_delta_t

    if t_junction > 85.0:
        derate_factor = 1.0 - (t_junction - 85.0) * 0.001
        derate_factor = max(0.7, derate_factor)
    else:
        derate_factor = 1.0

    if process_nm <= 7:
        derate_factor *= 0.98
    elif process_nm <= 16:
        derate_factor *= 0.99

    muls_per_column = mul_count / max(1, dsp_columns)
    if muls_per_column > 20:
        hotspot = "high"
        derate_factor *= 0.94
    elif muls_per_column > 10:
        hotspot = "medium"
        derate_factor *= 0.97
    elif muls_per_column > 4:
        hotspot = "low"
        derate_factor *= 0.99
    else:
        hotspot = "none"

    thermal_safe = t_junction < t_junction_max_c
    derated_freq = target_freq_mhz * derate_factor

    return ThermalEstimate(
        power_mw=estimated_power_mw,
        delta_t_c=round(delta_t, 2),
        junction_temp_c=round(t_junction, 1),
        hotspot_delta_t_c=round(hotspot_delta_t, 2),
        derated_freq_mhz=round(derated_freq, 1),
        thermal_safe=thermal_safe,
        hotspot_risk=hotspot,
    )


def estimate_thermal_envelope(
    *,
    power_mw: float = 100.0,
    theta_ja: float = 25.0,
    t_ambient: float = 25.0,
    t_junction_max: float = 125.0,
) -> ThermalEnvelopeEstimate:
    """Predict junction temperature from power dissipation."""
    power_w = power_mw / 1000.0
    t_j = t_ambient + power_w * theta_ja
    margin = t_junction_max - t_j
    status = "PASS" if margin > 0 else "FAIL"

    return ThermalEnvelopeEstimate(
        power_mw=power_mw,
        theta_ja=theta_ja,
        t_ambient=t_ambient,
        t_junction=round(t_j, 2),
        thermal_margin=round(margin, 2),
        pass_fail=status,
    )


def generate_thermal_constraints(
    module_name: str,
    analysis: ThermalEstimate,
    *,
    dsp_columns: int = 2,
) -> str:
    """Generate XDC constraints for thermal-aware DSP placement."""
    period_ns = 1000.0 / analysis.derated_freq_mhz
    lines = [
        f"# Thermal-aware constraints for {module_name}",
        "# SC-NeuroCore thermal compilation",
        f"# Junction temp: {analysis.junction_temp_c}°C, Hotspot risk: {analysis.hotspot_risk}",
        f"# Derated frequency: {analysis.derated_freq_mhz} MHz",
        "",
        "# Use derated clock period",
        f"create_clock -period {period_ns:.3f} -name clk [get_ports clk]",
        "",
    ]

    if analysis.hotspot_risk in ("medium", "high"):
        lines.extend(
            [
                f"# DSP spreading across {dsp_columns} columns to reduce hotspots",
                "set_property LOC DSP48E2_X0Y0 "
                "[get_cells -hier -filter {REF_NAME =~ DSP*} -limit 1]",
                "",
                "# Soft placement constraint: spread DSPs",
                "set_property C_REG 1 [get_cells -hier -filter {REF_NAME =~ DSP*}]",
                "",
            ]
        )

    if not analysis.thermal_safe:
        lines.extend(
            [
                f"# WARNING: Junction temperature {analysis.junction_temp_c}°C exceeds limit!",
                "# Consider: reduce clock, add heatsink, or reduce neuron count.",
                "",
            ]
        )

    return "\n".join(lines)


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
