# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio responsibility router registry

"""Build the ordered Studio responsibility-router set."""

from __future__ import annotations

from fastapi import APIRouter

from sc_neurocore.studio.api.adaptive_precision import build_adaptive_precision_router
from sc_neurocore.studio.api.audit import build_audit_router
from sc_neurocore.studio.api.catalogue import build_catalogue_router
from sc_neurocore.studio.api.compiler import build_compiler_router
from sc_neurocore.studio.api.cosim import build_cosim_router
from sc_neurocore.studio.api.deploy import build_deploy_router
from sc_neurocore.studio.api.design import build_design_router
from sc_neurocore.studio.api.export import build_export_router
from sc_neurocore.studio.api.identity import build_identity_router
from sc_neurocore.studio.api.jobs import build_jobs_router
from sc_neurocore.studio.api.presets import build_presets_router
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.simulation import build_simulation_router
from sc_neurocore.studio.api.synthesis import build_synthesis_router
from sc_neurocore.studio.api.system import build_system_router
from sc_neurocore.studio.api.training import build_training_router
from sc_neurocore.studio.api.training_weights import build_training_weights_router


def build_studio_routers(context: StudioApiContext) -> tuple[APIRouter, ...]:
    """Build responsibility routers in deterministic compatibility order.

    Parameters
    ----------
    context:
        Shared runtime state used by all route adapters.

    Returns
    -------
    tuple[APIRouter, ...]
        Routers ready for application inclusion.
    """
    return (
        build_system_router(context),
        build_jobs_router(context),
        build_audit_router(context),
        build_training_weights_router(context),
        build_identity_router(context),
        build_catalogue_router(context),
        build_presets_router(context),
        build_simulation_router(context),
        build_adaptive_precision_router(context),
        build_compiler_router(context),
        build_cosim_router(context),
        build_synthesis_router(context),
        build_design_router(context),
        build_deploy_router(context),
        build_training_router(context),
        build_export_router(context),
    )
