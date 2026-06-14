# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Supply chain risk scorer

"""Supply chain risk assessment for hardware profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SupplyChainRisk:
    """Supply chain risk assessment for a hardware profile.

    Attributes
    ----------
    profile_name : str
        Assessed profile.
    risk_score : float
        Risk score 0-100 (higher = riskier).
    risk_factors : list[str]
        Individual risk factor descriptions.
    alternatives : list[str]
        Suggested alternative profiles.
    export_control : str
        Export control classification.
    """

    profile_name: str
    risk_score: float
    risk_factors: list[str]
    alternatives: list[str]
    export_control: str


# Geography risk mapping
_GEO_RISK: dict[str, float] = {
    "TSMC": 35,
    "MediaTek": 30,  # Taiwan concentration
    "Samsung": 20,
    "SK Hynix": 20,  # South Korea
    "NIST": 5,
    "Northrop Grumman": 5,  # US defence
    "Intel": 10,
    "AMD": 10,
    "Qualcomm": 10,
    "Xilinx": 10,
    "Lattice": 10,
    "Microchip": 10,
    "Research": 50,  # Research-only, no commercial supply
    "FinalSpark": 60,
    "Cortical Labs": 60,  # Pre-commercial
    "Stanford": 60,  # Academic
    "Tachyum": 45,  # Pre-production
}


def score_supply_chain_risk(
    profile_name: str,
) -> SupplyChainRisk:
    """Assess supply chain risk for a hardware profile."""
    from ..platforms import get_profile

    p = get_profile(profile_name)
    score = 0.0
    factors = []

    # Geographic risk
    geo = _GEO_RISK.get(p.vendor, 15)
    score += geo
    if geo >= 30:
        factors.append(f"Geographic concentration: {p.vendor}")

    # Sole-source risk
    if p.platform_class in ("biological", "wetware", "superconducting", "electrochemical"):
        score += 20
        factors.append("Emerging tech, limited vendors")

    # Export control
    export = "EAR99"
    if p.platform_class in ("fpga",) and "rad" in p.name.lower():
        export = "ITAR"
        score += 15
        factors.append("ITAR-controlled radiation-hardened")
    elif p.platform_class == "superconducting":
        export = "EAR-controlled"
        score += 10
        factors.append("Export-controlled superconducting tech")

    if not factors:
        factors.append("Standard commercial supply")

    # Alternatives
    from ..platforms import list_profile_names

    alts = [
        n
        for n in list_profile_names()
        if n != profile_name and get_profile(n).platform_class == p.platform_class
    ][:3]

    return SupplyChainRisk(
        profile_name=profile_name,
        risk_score=min(100, round(score, 1)),
        risk_factors=factors,
        alternatives=alts,
        export_control=export,
    )
