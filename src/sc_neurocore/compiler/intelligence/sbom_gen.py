# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SBOM generator

"""Software/Hardware Bill of Materials (SBOM/HBOM) generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SBOM:
    """Software/Hardware Bill of Materials.

    Attributes
    ----------
    format : str
    components : list[dict]
    total_components : int
    """

    format: str
    components: list[dict]
    total_components: int


def generate_sbom(
    module_name: str,
    profile_name: str,
    *,
    dependencies: dict[str, str] | None = None,
    sbom_format: str = "CycloneDX",
) -> SBOM:
    """Generate SBOM/HBOM for IP core compliance."""
    from ...platforms import get_profile

    p = get_profile(profile_name)

    components = [
        {
            "type": "library",
            "name": "sc-neurocore",
            "version": "3.15.7",
            "license": "AGPL-3.0-or-later",
        },
        {"type": "hardware", "name": profile_name, "vendor": p.vendor, "family": p.family},
        {"type": "module", "name": module_name, "target": profile_name},
    ]
    if dependencies:
        for name, version in dependencies.items():
            components.append({"type": "library", "name": name, "version": version})

    return SBOM(
        format=sbom_format,
        components=components,
        total_components=len(components),
    )
