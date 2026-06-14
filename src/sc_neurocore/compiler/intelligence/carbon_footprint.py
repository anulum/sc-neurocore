# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Carbon footprint estimator

"""Lifecycle carbon footprint estimation for compilation targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarbonEstimate:
    """Carbon footprint estimate per compilation target.

    Attributes
    ----------
    profile_name : str
        Target profile.
    manufacturing_kg_co2 : float
        Estimated manufacturing CO₂ (kg).
    operation_kg_co2_per_year : float
        Estimated annual operation CO₂ (kg).
    total_5yr_kg_co2 : float
        Total 5-year lifecycle CO₂ (kg).
    energy_mix : str
        Assumed energy source.
    """

    profile_name: str
    manufacturing_kg_co2: float
    operation_kg_co2_per_year: float
    total_5yr_kg_co2: float
    energy_mix: str


# Approximate manufacturing CO2 per process node (kg CO2 per die)
_MFG_CO2: dict[str, float] = {
    "fpga": 8.0,
    "asic": 12.0,
    "neuromorphic": 6.0,
    "photonic": 10.0,
    "in_memory": 5.0,
    "accelerator": 15.0,
    "edge_mcu": 0.5,
    "biological": 0.1,
    "wetware": 0.1,
    "simulation": 0.0,
    "superconducting": 20.0,
    "quantum_neuro": 25.0,
    "rram": 3.0,
    "sram_cim": 4.0,
    "electrochemical": 2.0,
}


def estimate_carbon_footprint(
    profile_name: str,
    *,
    power_mw: float = 100.0,
    hours_per_day: float = 24.0,
    grid_carbon_g_per_kwh: float = 400.0,
) -> CarbonEstimate:
    """Estimate carbon footprint for a compilation target."""
    from ..platforms import get_profile

    p = get_profile(profile_name)

    mfg = _MFG_CO2.get(p.platform_class, 5.0)
    kwh_per_year = (power_mw / 1e6) * hours_per_day * 365
    op_kg = kwh_per_year * grid_carbon_g_per_kwh / 1000
    total = mfg + op_kg * 5

    return CarbonEstimate(
        profile_name=profile_name,
        manufacturing_kg_co2=round(mfg, 2),
        operation_kg_co2_per_year=round(op_kg, 4),
        total_5yr_kg_co2=round(total, 2),
        energy_mix="grid_average",
    )
