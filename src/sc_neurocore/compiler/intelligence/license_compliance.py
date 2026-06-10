# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — License compliance checker

"""IP core license compatibility checking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LicenseCheck:
    """IP core license compatibility result.

    Attributes
    ----------
    compatible : bool
    conflicts : list[str]
    licenses_found : list[str]
    """

    compatible: bool
    conflicts: list[str]
    licenses_found: list[str]


# Compatibility matrix: {project_license: [allowed_deps]}
_COMPAT: dict[str, set[str]] = {
    "AGPL-3.0": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC", "AGPL-3.0"},
    "GPL-3.0": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC", "GPL-3.0", "LGPL-3.0"},
    "Apache-2.0": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC"},
    "MIT": {"MIT", "BSD-2", "BSD-3", "ISC"},
    "proprietary": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC"},
}


def check_license_compliance(
    project_license: str,
    dependencies: dict[str, str],
) -> LicenseCheck:
    """Verify IP core licensing compatibility."""
    allowed = _COMPAT.get(project_license, set())
    conflicts = []
    licenses = []

    for dep, lic in dependencies.items():
        licenses.append(lic)
        if lic not in allowed:
            conflicts.append(f"{dep} ({lic}) incompatible with {project_license}")

    return LicenseCheck(
        compatible=len(conflicts) == 0,
        conflicts=conflicts,
        licenses_found=licenses,
    )
