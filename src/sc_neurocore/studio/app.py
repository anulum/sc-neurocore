# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FastAPI backend for Visual SNN Design Studio

from __future__ import annotations

import traceback

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


def _safe(fn, detail_prefix: str = ""):
    """Wrap any callable so exceptions become 422 with traceback detail, never 500."""
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc().split("\n")[-3:]
        msg = f"{detail_prefix}{e}" if detail_prefix else str(e)
        raise HTTPException(status_code=422, detail=f"{msg}\n{''.join(tb)}") from e


def create_app() -> FastAPI:
    app = FastAPI(title="SC-NeuroCore Studio", version="0.2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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

    @app.get("/api/models")
    def models_list():
        return _safe(list_models)

    @app.get("/api/models/{name}")
    def model_detail(name: str):
        def fn():
            m = get_model_detail(name)
            if m is None:
                raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
            return m
        return _safe(fn)

    @app.post("/api/models/simulate")
    def model_simulate(req: ModelSimulateRequest):
        return _safe(lambda: simulate_model(
            name=req.name, param_overrides=req.params, dt=req.dt,
            duration=req.duration, current=req.current, protocol=req.protocol,
        ), f"Model '{req.name}': ")

    @app.post("/api/simulate")
    def run_simulation(req: SimulateRequest):
        return _safe(lambda: simulate(
            equations=req.equations, threshold=req.threshold, reset=req.reset,
            params=req.params, init=req.init, dt=req.dt,
            duration=req.duration, current=req.current, protocol=req.protocol,
        ))

    @app.post("/api/fi-curve")
    def run_fi_curve(req: FICurveRequest):
        def fn():
            if req.model_name:
                import numpy as np
                currents, rates = [], []
                for I_val in np.linspace(req.i_min, req.i_max, req.i_steps):
                    r = simulate_model(
                        name=req.model_name, param_overrides=req.params,
                        dt=req.dt if req.dt != 0.1 else None,
                        duration=req.duration, current=float(I_val), protocol="constant",
                    )
                    currents.append(float(I_val))
                    rates.append(r["stats"]["rate_hz"])
                return {"currents": currents, "rates": rates}
            elif req.equations:
                return fi_curve(
                    equations=req.equations, threshold=req.threshold, reset=req.reset,
                    params=req.params, init=req.init, dt=req.dt,
                    duration=req.duration, i_min=req.i_min, i_max=req.i_max, i_steps=req.i_steps,
                )
            raise ValueError("Either equations or model_name required")
        return _safe(fn)

    @app.post("/api/compile")
    def compile_verilog(req: CompileRequest):
        def fn():
            from sc_neurocore.compiler.equation_compiler import equation_to_fpga
            _, verilog = equation_to_fpga(
                req.equations[0], threshold=req.threshold, reset=req.reset,
                params=req.params, init=req.init, module_name=req.module_name,
            )
            return {"verilog": verilog, "module_name": req.module_name, "chars": len(verilog)}
        return _safe(fn)

    return app


app = create_app()
