# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio hardware registration discovery

"""Expose hardware registrations without performing a hardware operation."""

from __future__ import annotations

from fastapi import APIRouter

from sc_neurocore.studio.hardware_catalogue import HardwareCatalogue, hardware_catalogue


def build_hardware_router() -> APIRouter:
    """Build the read-only hardware discovery router.

    Returns
    -------
    APIRouter
        Public registration metadata. No device probe, job or vendor call.
    """
    router = APIRouter()

    @router.get("/api/hardware/catalogue", response_model=HardwareCatalogue)
    def api_hardware_catalogue() -> HardwareCatalogue:
        """Return registered paths without attesting device execution."""
        return hardware_catalogue()

    return router
