# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FastAPI backend for Visual SNN Design Studio

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sc_neurocore.studio.simulation import fi_curve, simulate
from sc_neurocore.studio.templates import get_template, list_templates


class SimulateRequest(BaseModel):
    equations: list[str]
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = Field(default=0.1, gt=0)
    duration: float = Field(default=100.0, gt=0)
    current: float = 0.0
    protocol: str = "constant"


class FICurveRequest(BaseModel):
    equations: list[str]
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = Field(default=0.1, gt=0)
    duration: float = Field(default=200.0, gt=0)
    i_min: float = 0.0
    i_max: float = 50.0
    i_steps: int = Field(default=20, ge=2, le=100)


def create_app() -> FastAPI:
    app = FastAPI(title="SC-NeuroCore Studio", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/templates")
    def templates_list():
        return list_templates()

    @app.get("/api/templates/{name}")
    def template_detail(name: str):
        t = get_template(name)
        if t is None:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        return t

    @app.post("/api/simulate")
    def run_simulation(req: SimulateRequest):
        try:
            result = simulate(
                equations=req.equations,
                threshold=req.threshold,
                reset=req.reset,
                params=req.params,
                init=req.init,
                dt=req.dt,
                duration=req.duration,
                current=req.current,
                protocol=req.protocol,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return result

    @app.post("/api/fi-curve")
    def run_fi_curve(req: FICurveRequest):
        try:
            result = fi_curve(
                equations=req.equations,
                threshold=req.threshold,
                reset=req.reset,
                params=req.params,
                init=req.init,
                dt=req.dt,
                duration=req.duration,
                i_min=req.i_min,
                i_max=req.i_max,
                i_steps=req.i_steps,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return result

    return app


app = create_app()
