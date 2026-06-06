# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aging and reliability predictor

"""Aging and reliability prediction models for semiconductor degradation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ReliabilityEstimate:
    """Mean time to failure estimate.

    Attributes
    ----------
    mttf_hours : float
        Estimated MTTF in hours.
    mttf_years : float
        Estimated MTTF in years.
    failure_mode : str
        Dominant failure mechanism.
    voltage_stress : float
        Normalised voltage stress factor.
    temp_accel : float
        Arrhenius temperature acceleration factor.
    mechanism_mttf_hours : dict[str, float]
        Per-mechanism MTTF estimates for NBTI, HCI, and TDDB.
    """

    mttf_hours: float
    mttf_years: float
    failure_mode: str
    voltage_stress: float
    temp_accel: float
    mechanism_mttf_hours: dict[str, float]


@dataclass
class AgingPrediction:
    """Transistor aging prediction.

    Attributes
    ----------
    initial_fmax_mhz : float
    degraded_fmax_mhz : float
    degradation_pct : float
    recommended_derating : float
    dominant_mechanism : str
    nbti_degradation_pct : float
    hci_degradation_pct : float
    """

    initial_fmax_mhz: float
    degraded_fmax_mhz: float
    degradation_pct: float
    recommended_derating: float
    dominant_mechanism: str
    nbti_degradation_pct: float
    hci_degradation_pct: float


def predict_reliability(
    *,
    voltage_v: float = 0.9,
    temperature_c: float = 85.0,
    node_nm: int = 7,
    base_mttf_hours: float = 1e6,
) -> ReliabilityEstimate:
    """Predict MTTF from voltage, temperature, and technology node."""
    _require_finite_positive(voltage_v, "voltage_v")
    _require_celsius_above_absolute_zero(temperature_c, "temperature_c")
    if not isinstance(node_nm, int) or node_nm <= 0:
        raise ValueError("node_nm must be a positive integer")
    _require_finite_positive(base_mttf_hours, "base_mttf_hours")

    k = 8.617e-5  # Boltzmann constant (eV/K)
    t_ref = 25.0 + 273.15
    t_op = temperature_c + 273.15

    nbti_temp_accel = math.exp(0.55 / k * (1 / t_ref - 1 / t_op))
    hci_temp_accel = math.exp(0.20 / k * (1 / t_ref - 1 / t_op))
    tddb_temp_accel = math.exp(0.70 / k * (1 / t_ref - 1 / t_op))
    nbti_voltage_accel = (voltage_v / 0.9) ** 2.0
    hci_voltage_accel = (voltage_v / 0.9) ** 4.0
    tddb_voltage_accel = math.exp(4.0 * ((voltage_v / 0.9) - 1.0))
    v_stress = max(nbti_voltage_accel, hci_voltage_accel, tddb_voltage_accel)
    temp_accel = max(nbti_temp_accel, hci_temp_accel, tddb_temp_accel)
    node_factor = max(0.5, node_nm / 28.0)  # smaller nodes degrade faster

    mechanism_accels = {
        "NBTI": nbti_temp_accel * nbti_voltage_accel,
        "HCI": hci_temp_accel * hci_voltage_accel,
        "TDDB": tddb_temp_accel * tddb_voltage_accel,
    }
    mechanism_mttf = {
        name: base_mttf_hours / accel * node_factor for name, accel in mechanism_accels.items()
    }
    failure = min(mechanism_mttf, key=lambda name: mechanism_mttf[name])
    mttf = mechanism_mttf[failure]

    return ReliabilityEstimate(
        mttf_hours=round(mttf, 1),
        mttf_years=round(mttf / 8760, 2),
        failure_mode=failure,
        voltage_stress=round(v_stress, 3),
        temp_accel=round(temp_accel, 3),
        mechanism_mttf_hours={k: round(v, 1) for k, v in mechanism_mttf.items()},
    )


def predict_aging(
    initial_fmax_mhz: float,
    *,
    voltage_v: float = 0.9,
    temperature_c: float = 85.0,
    years: float = 10.0,
) -> AgingPrediction:
    """Predict end-of-life Fmax after transistor aging."""
    _require_finite_positive(initial_fmax_mhz, "initial_fmax_mhz")
    _require_finite_positive(voltage_v, "voltage_v")
    _require_celsius_above_absolute_zero(temperature_c, "temperature_c")
    _require_finite_non_negative(years, "years")

    k_b_ev = 8.617333262e-5
    t_ref_k = 25.0 + 273.15
    t_op_k = temperature_c + 273.15
    time_years = max(years, 0.0)

    nbti_temp_accel = math.exp(0.55 / k_b_ev * (1.0 / t_ref_k - 1.0 / t_op_k))
    hci_temp_accel = math.exp(0.20 / k_b_ev * (1.0 / t_ref_k - 1.0 / t_op_k))
    nbti_time_factor = (time_years / 10.0) ** 0.25 if time_years > 0.0 else 0.0
    hci_time_factor = (time_years / 10.0) ** 0.50 if time_years > 0.0 else 0.0
    nbti_voltage_factor = (voltage_v / 0.9) ** 2.0
    hci_voltage_factor = (voltage_v / 0.9) ** 4.0

    nbti_pct = 3.0 * nbti_time_factor * nbti_temp_accel * nbti_voltage_factor
    hci_pct = 1.5 * hci_time_factor * hci_temp_accel * hci_voltage_factor

    total_degradation = min(nbti_pct + hci_pct, 50.0)
    degraded = initial_fmax_mhz * (1 - total_degradation / 100)
    dominant = "NBTI" if nbti_pct > hci_pct else "HCI"

    return AgingPrediction(
        initial_fmax_mhz=initial_fmax_mhz,
        degraded_fmax_mhz=round(degraded, 1),
        degradation_pct=round(total_degradation, 2),
        recommended_derating=round(1.0 + total_degradation / 100, 3),
        dominant_mechanism=dominant,
        nbti_degradation_pct=round(nbti_pct, 2),
        hci_degradation_pct=round(hci_pct, 2),
    )


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_celsius_above_absolute_zero(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= -273.15:
        raise ValueError(f"{name} must be finite and above absolute zero")
