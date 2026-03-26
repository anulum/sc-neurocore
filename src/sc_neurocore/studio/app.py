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

from sc_neurocore.studio.models import get_model_detail, list_models, simulate_model
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


class ModelSimulateRequest(BaseModel):
    name: str
    params: dict[str, float] | None = None
    dt: float | None = None
    duration: float = Field(default=100.0, gt=0)
    current: float = 10.0
    protocol: str = "constant"


class FICurveRequest(BaseModel):
    equations: list[str] | None = None
    model_name: str | None = None
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = Field(default=0.1, gt=0)
    duration: float = Field(default=200.0, gt=0)
    i_min: float = 0.0
    i_max: float = 50.0
    i_steps: int = Field(default=25, ge=2, le=100)


class CompileRequest(BaseModel):
    equations: list[str]
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    module_name: str = "sc_neuron"


def create_app() -> FastAPI:
    app = FastAPI(title="SC-NeuroCore Studio", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # --- ODE equation templates (5 built-in) ---

    @app.get("/api/templates")
    def templates_list():
        return list_templates()

    @app.get("/api/templates/{name}")
    def template_detail(name: str):
        t = get_template(name)
        if t is None:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        return t

    # --- Model library (118 neuron models) ---

    @app.get("/api/models")
    def models_list():
        return list_models()

    @app.get("/api/models/{name}")
    def model_detail(name: str):
        m = get_model_detail(name)
        if m is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        return m

    @app.post("/api/models/simulate")
    def model_simulate(req: ModelSimulateRequest):
        try:
            return simulate_model(
                name=req.name,
                param_overrides=req.params,
                dt=req.dt,
                duration=req.duration,
                current=req.current,
                protocol=req.protocol,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # --- Custom ODE simulation ---

    @app.post("/api/simulate")
    def run_simulation(req: SimulateRequest):
        try:
            return simulate(
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

    # --- f-I curve (works with both ODE and model) ---

    @app.post("/api/fi-curve")
    def run_fi_curve(req: FICurveRequest):
        try:
            if req.model_name:
                currents = []
                rates = []
                import numpy as np
                for I_val in np.linspace(req.i_min, req.i_max, req.i_steps):
                    r = simulate_model(
                        name=req.model_name,
                        param_overrides=req.params,
                        dt=req.dt if req.dt != 0.1 else None,
                        duration=req.duration,
                        current=float(I_val),
                        protocol="constant",
                    )
                    currents.append(float(I_val))
                    rates.append(r["stats"]["rate_hz"])
                return {"currents": currents, "rates": rates}
            elif req.equations:
                return fi_curve(
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
            else:
                raise ValueError("Either equations or model_name required")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # --- Verilog compilation ---

    @app.post("/api/compile")
    def compile_verilog(req: CompileRequest):
        try:
            from sc_neurocore.compiler.equation_compiler import equation_to_fpga
            ode_str = req.equations[0] if len(req.equations) == 1 else req.equations[0]
            _, verilog = equation_to_fpga(
                ode_str,
                threshold=req.threshold,
                reset=req.reset,
                params=req.params,
                init=req.init,
                module_name=req.module_name,
            )
            return {"verilog": verilog, "module_name": req.module_name, "chars": len(verilog)}
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app


app = create_app()
