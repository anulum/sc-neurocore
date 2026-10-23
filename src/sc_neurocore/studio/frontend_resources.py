# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio interface resource discovery

"""Locate packaged or checkout UI assets without importing the scientific API."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def studio_frontend_candidates(app_module_file: str) -> tuple[Path, ...]:
    """Return packaged assets before the application module's checkout candidates.

    Parameters
    ----------
    app_module_file : str
        Application module path in the selected installation.

    Returns
    -------
    tuple of pathlib.Path
        Packaged interface, source-layout build and flat-layout build candidates.
    """
    anchor = Path(app_module_file).resolve().parent
    return (
        Path(__file__).resolve().parent / "frontend_dist",
        anchor.parents[2] / "studio" / "frontend" / "dist",
        anchor.parents[3] / "studio" / "frontend" / "dist",
    )


def studio_frontend_dir(candidates: Sequence[Path]) -> Path | None:
    """Return the first candidate containing an interface entry file.

    Parameters
    ----------
    candidates : sequence of pathlib.Path
        Ordered installation or checkout locations.

    Returns
    -------
    pathlib.Path or None
        First built interface, or no interface when every candidate is absent.
    """
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return None
