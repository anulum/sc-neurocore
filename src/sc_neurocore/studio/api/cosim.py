# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio co-simulation routes

"""Expose Python-to-SystemVerilog trace parity through a focused adapter."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import PrecisionRequest
from sc_neurocore.studio.compiler import cosim_traces


def build_cosim_router(context: StudioApiContext) -> APIRouter:
    """Build the co-simulation router over shared Studio runtime state."""
    router = APIRouter()

    @router.post("/api/ir/cosim")
    def api_ir_cosim(req: PrecisionRequest) -> Any:
        return _safe(
            lambda: cosim_traces(
                equations=req.equations,
                threshold=req.threshold,
                reset=req.reset,
                params=req.params,
                init=req.init,
                dt=req.dt,
                duration=req.duration,
                current=req.current,
            )
        )

    return router
