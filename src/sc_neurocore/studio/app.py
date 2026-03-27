# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FastAPI backend for Visual SNN Design Studio

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sc_neurocore.studio.analysis import (
    bifurcation_sweep,
    frequency_response,
    heatmap_2d,
    nullclines_2d,
    precision_compare,
    sensitivity_analysis,
)
from sc_neurocore.studio.characterize import characterize_model
from sc_neurocore.studio.model_scan import scan_all_models
from sc_neurocore.studio.network import simulate_ei_network
from sc_neurocore.studio.codegen import (
    classify_firing_pattern,
    generate_model_script,
    generate_ode_script,
    generate_oneliner,
)
from sc_neurocore.studio.compiler import (
    build_ir_from_equation,
    cosim_traces,
    emit_sv_from_equation,
    emit_systemverilog,
    verify_ir,
)
from sc_neurocore.studio.synthesis import (
    check_tools,
    estimate_resources,
    multi_target_synthesis,
    run_pnr,
    run_synthesis,
)
from sc_neurocore.studio.project import (
    delete_project,
    list_projects,
    load_project,
    run_pipeline,
    save_project,
)
from sc_neurocore.studio.network_graph import (
    available_models as graph_available_models,
    create_population,
    create_projection,
    graph_to_nir,
    nir_to_graph,
    simulate_graph,
    validate_graph,
)
from sc_neurocore.studio.training import (
    get_training_status,
    list_cell_types,
    list_jobs,
    list_surrogates,
    start_training,
    stop_training,
    stream_metrics,
)
from sc_neurocore.studio.models import get_model_detail, list_models, simulate_model
from sc_neurocore.studio.presets import get_preset, list_presets
from sc_neurocore.studio.simulation import simulate
from sc_neurocore.studio.templates import get_template, list_templates


# --- Request schemas ---


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


class BifurcationRequest(BaseModel):
    equations: list[str] | None = None
    model_name: str | None = None
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = 0.1
    duration: float = 200.0
    current: float = 10.0
    sweep_param: str
    sweep_min: float
    sweep_max: float
    sweep_steps: int = Field(default=30, ge=5, le=80)


class SensitivityRequest(BaseModel):
    equations: list[str] | None = None
    model_name: str | None = None
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = 0.1
    duration: float = 200.0
    current: float = 10.0


class NullclineRequest(BaseModel):
    equations: list[str]
    params: dict[str, float]
    var_names: list[str]
    ranges: dict[str, list[float]]
    grid_size: int = Field(default=60, ge=20, le=150)


class PrecisionRequest(BaseModel):
    equations: list[str]
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = 0.1
    duration: float = 200.0
    current: float = 10.0


class CompareRequest(BaseModel):
    config_a: dict
    config_b: dict


class FreqResponseRequest(BaseModel):
    equations: list[str] | None = None
    model_name: str | None = None
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = 0.1
    duration: float = 200.0
    amplitude: float = 10.0
    freq_min: float = 1.0
    freq_max: float = 100.0
    n_freqs: int = Field(default=15, ge=3, le=50)


class HeatmapRequest(BaseModel):
    equations: list[str] | None = None
    model_name: str | None = None
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = 0.1
    duration: float = 100.0
    current: float = 10.0
    param_x: str
    x_min: float
    x_max: float
    x_steps: int = Field(default=15, ge=3, le=30)
    param_y: str
    y_min: float
    y_max: float
    y_steps: int = Field(default=15, ge=3, le=30)


class NetworkRequest(BaseModel):
    n_exc: int = Field(default=80, ge=10, le=500)
    n_inh: int = Field(default=20, ge=5, le=200)
    w_ee: float = 0.1
    w_ei: float = 0.4
    w_ie: float = 0.1
    w_ii: float = 0.4
    p_conn: float = Field(default=0.2, ge=0.01, le=1.0)
    ext_rate: float = 5.0
    duration: float = Field(default=200.0, gt=0, le=2000)
    dt: float = Field(default=0.1, gt=0)


class CodegenRequest(BaseModel):
    mode: str = "model"
    model_name: str | None = None
    equations: list[str] | None = None
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = 0.1
    duration: float = 100.0
    current: float = 10.0


class _SimCache:
    """LRU cache for simulation results keyed by JSON hash."""

    def __init__(self, maxsize: int = 64):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _key(self, data: dict) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    def get(self, params: dict):
        k = self._key(params)
        if k in self._cache:
            self.hits += 1
            self._cache.move_to_end(k)
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, params: dict, result: dict):
        k = self._key(params)
        self._cache[k] = result
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


_cache = _SimCache()


def _safe(fn, detail_prefix: str = ""):
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:
        msg = f"{detail_prefix}{e}" if detail_prefix else str(e)
        raise HTTPException(status_code=422, detail=msg) from e


def _make_simulate_fn(req_dict: dict):
    """Build a simulate callable from request params (ODE or model)."""
    if req_dict.get("model_name"):

        def fn(**overrides):
            cfg = {
                "name": req_dict["model_name"],
                "param_overrides": overrides.get("params", req_dict.get("params")),
                "dt": overrides.get("dt", req_dict.get("dt")),
                "duration": overrides.get("duration", req_dict.get("duration", 200)),
                "current": overrides.get("current", req_dict.get("current", 10)),
                "protocol": overrides.get("protocol", req_dict.get("protocol", "constant")),
            }
            return simulate_model(**cfg)

        return fn
    else:

        def fn(**overrides):
            return simulate(
                equations=req_dict.get("equations", []),
                threshold=req_dict.get("threshold"),
                reset=req_dict.get("reset"),
                params=overrides.get("params", req_dict.get("params")),
                init=overrides.get("init", req_dict.get("init")),
                dt=overrides.get("dt", req_dict.get("dt", 0.1)),
                duration=overrides.get("duration", req_dict.get("duration", 200)),
                current=overrides.get("current", req_dict.get("current", 10)),
                protocol=overrides.get("protocol", req_dict.get("protocol", "constant")),
            )

        return fn


def create_app() -> FastAPI:
    app = FastAPI(title="SC-NeuroCore Studio", version="0.3.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # --- Health ---
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # --- Templates & Models ---
    @app.get("/api/templates")
    def api_templates():
        return list_templates()

    @app.get("/api/templates/{name}")
    def api_template(name: str):
        t = get_template(name)
        if not t:
            raise HTTPException(404, f"Template '{name}' not found")
        return t

    @app.get("/api/models")
    def api_models():
        return _safe(list_models)

    # --- Model scan (behavior classification) — must precede /api/models/{name} ---
    @app.get("/api/models/scan")
    def api_model_scan():
        return _safe(lambda: scan_all_models(current=10.0, duration=100.0))

    @app.get("/api/models/{name}")
    def api_model(name: str):
        return _safe(
            lambda: (
                get_model_detail(name)
                or (_ for _ in ()).throw(HTTPException(404, f"Model '{name}' not found"))
            )
        )

    # --- Presets (#3) ---
    @app.get("/api/presets")
    def api_presets():
        return list_presets()

    @app.get("/api/presets/{preset_id}")
    def api_preset(preset_id: str):
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        return p

    # --- Simulation (with auto-classification + cache) ---
    @app.post("/api/simulate")
    def api_simulate(req: SimulateRequest):
        cache_key = {"_type": "ode", **req.model_dump()}
        cached = _cache.get(cache_key)
        if cached:
            return cached

        def fn():
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
            result["pattern"] = classify_firing_pattern(
                result["spikes"], result["n_steps"], result["dt"]
            )
            _cache.put(cache_key, result)
            return result

        return _safe(fn)

    @app.post("/api/models/simulate")
    def api_model_simulate(req: ModelSimulateRequest):
        cache_key = {"_type": "model", **req.model_dump()}
        cached = _cache.get(cache_key)
        if cached:
            return cached

        def fn():
            result = simulate_model(
                name=req.name,
                param_overrides=req.params,
                dt=req.dt,
                duration=req.duration,
                current=req.current,
                protocol=req.protocol,
            )
            result["pattern"] = classify_firing_pattern(
                result["spikes"], result["n_steps"], result["dt"]
            )
            _cache.put(cache_key, result)
            return result

        return _safe(fn, f"Model '{req.name}': ")

    @app.get("/api/cache/stats")
    def api_cache_stats():
        return {"hits": _cache.hits, "misses": _cache.misses, "size": len(_cache._cache)}

    # --- Comparison (#1) ---
    @app.post("/api/compare")
    def api_compare(req: CompareRequest):
        def fn():
            sim_a = _make_simulate_fn(req.config_a)
            sim_b = _make_simulate_fn(req.config_b)
            return {"a": sim_a(), "b": sim_b()}

        return _safe(fn)

    # --- f-I Curve ---
    @app.post("/api/fi-curve")
    def api_fi_curve(req: FICurveRequest):
        def fn():
            import numpy as np

            sim_fn = _make_simulate_fn(req.model_dump())
            currents = np.linspace(req.i_min, req.i_max, req.i_steps).tolist()
            rates = [sim_fn(current=float(I))["stats"]["rate_hz"] for I in currents]
            return {"currents": currents, "rates": rates}

        return _safe(fn)

    # --- Bifurcation (#2) ---
    @app.post("/api/bifurcation")
    def api_bifurcation(req: BifurcationRequest):
        def fn():
            sim_fn = _make_simulate_fn(req.model_dump())
            base_cfg = {
                "params": req.params,
                "init": req.init,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.current,
                "protocol": "constant",
            }
            return bifurcation_sweep(
                sim_fn, base_cfg, req.sweep_param, req.sweep_min, req.sweep_max, req.sweep_steps
            )

        return _safe(fn)

    # --- Sensitivity (#8) ---
    @app.post("/api/sensitivity")
    def api_sensitivity(req: SensitivityRequest):
        def fn():
            sim_fn = _make_simulate_fn(req.model_dump())
            param_names = list((req.params or {}).keys())
            base_cfg = {
                "params": req.params,
                "init": req.init,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.current,
                "protocol": "constant",
            }
            return sensitivity_analysis(sim_fn, base_cfg, param_names)

        return _safe(fn)

    # --- Nullclines (#9) ---
    @app.post("/api/nullclines")
    def api_nullclines(req: NullclineRequest):
        def fn():
            ranges = {k: tuple(v) for k, v in req.ranges.items()}
            return nullclines_2d(req.equations, req.params, req.var_names, ranges, req.grid_size)

        return _safe(fn)

    # --- Precision Compare (#5) ---
    @app.post("/api/precision")
    def api_precision(req: PrecisionRequest):
        return _safe(
            lambda: precision_compare(
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

    # --- Compile (#5 adjacent) ---
    @app.post("/api/compile")
    def api_compile(req: CompileRequest):
        def fn():
            from sc_neurocore.compiler.equation_compiler import equation_to_fpga

            _, verilog = equation_to_fpga(
                req.equations[0],
                threshold=req.threshold,
                reset=req.reset,
                params=req.params,
                init=req.init,
                module_name=req.module_name,
            )
            return {"verilog": verilog, "module_name": req.module_name, "chars": len(verilog)}

        return _safe(fn)

    # --- Frequency Response (#11) ---
    @app.post("/api/freq-response")
    def api_freq_response(req: FreqResponseRequest):
        def fn():
            sim_fn = _make_simulate_fn(req.model_dump())
            base_cfg = {
                "params": req.params,
                "init": req.init,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.amplitude,
                "protocol": "constant",
            }
            return frequency_response(
                sim_fn, base_cfg, req.freq_min, req.freq_max, req.n_freqs, req.amplitude
            )

        return _safe(fn)

    # --- 2D Heatmap ---
    @app.post("/api/heatmap")
    def api_heatmap(req: HeatmapRequest):
        def fn():
            sim_fn = _make_simulate_fn(req.model_dump())
            base_cfg = {
                "params": req.params,
                "init": req.init,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.current,
                "protocol": "constant",
            }
            return heatmap_2d(
                sim_fn,
                base_cfg,
                req.param_x,
                req.x_min,
                req.x_max,
                req.x_steps,
                req.param_y,
                req.y_min,
                req.y_max,
                req.y_steps,
            )

        return _safe(fn)

    # --- Code Generation ---
    @app.post("/api/codegen")
    def api_codegen(req: CodegenRequest):
        if req.mode == "model" and req.model_name:
            script = generate_model_script(
                req.model_name, req.params, req.duration, req.current, req.dt
            )
            oneliner = generate_oneliner(req.model_name, req.params, req.current)
        else:
            script = generate_ode_script(
                req.equations or [],
                req.threshold,
                req.reset,
                req.params,
                req.init,
                req.duration,
                req.current,
                req.dt,
            )
            oneliner = ""
        return {"script": script, "oneliner": oneliner}

    # --- Firing Pattern Classification ---
    @app.post("/api/classify")
    def api_classify(req: SimulateRequest):
        def fn():
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
            pattern = classify_firing_pattern(result["spikes"], result["n_steps"], result["dt"])
            return {**result, "pattern": pattern}

        return _safe(fn)

    # --- One-click Characterisation ---
    @app.post("/api/characterize")
    def api_characterize(req: ModelSimulateRequest):
        def fn():
            sim_fn = _make_simulate_fn(
                {
                    "model_name": req.name,
                    "params": req.params,
                    "dt": req.dt,
                    "duration": req.duration,
                    "current": req.current,
                    "protocol": "constant",
                }
            )
            base_cfg = {
                "params": req.params,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.current,
                "protocol": "constant",
            }
            return characterize_model(sim_fn, base_cfg)

        return _safe(fn, f"Characterize '{req.name}': ")

    # --- Multi-model Overlay ---
    @app.post("/api/multi-simulate")
    def api_multi_simulate(configs: list[ModelSimulateRequest]):
        def fn():
            results = []
            for cfg in configs[:4]:
                sim_fn = _make_simulate_fn(
                    {
                        "model_name": cfg.name,
                        "params": cfg.params,
                        "dt": cfg.dt,
                        "duration": cfg.duration,
                        "current": cfg.current,
                        "protocol": cfg.protocol,
                    }
                )
                r = sim_fn()
                r["pattern"] = classify_firing_pattern(r["spikes"], r["n_steps"], r["dt"])
                results.append(r)
            return results

        return _safe(fn)

    # --- Data Import (CSV voltage trace) ---
    @app.post("/api/import-trace")
    def api_import_trace(data: dict):
        """Accept a voltage trace as JSON array for overlay comparison."""
        voltage = data.get("voltage", [])
        dt = data.get("dt", 0.1)
        if not voltage or not isinstance(voltage, list):
            raise HTTPException(422, "Expected {voltage: [...], dt: float}")
        import numpy as np

        v = np.array(voltage, dtype=float)
        time = (np.arange(len(v)) * dt).tolist()
        # Detect spikes (threshold crossings)
        threshold = np.mean(v) + 2 * np.std(v)
        crossings = np.where(np.diff(np.sign(v - threshold)) > 0)[0]
        return {
            "time": time,
            "voltage": v.tolist(),
            "spikes": crossings.tolist(),
            "spike_count": len(crossings),
            "dt": dt,
            "stats": {
                "mean": round(float(np.mean(v)), 2),
                "std": round(float(np.std(v)), 2),
                "min": round(float(np.min(v)), 2),
                "max": round(float(np.max(v)), 2),
                "threshold_estimate": round(float(threshold), 2),
            },
        }

    # --- E-I Network Simulation ---
    @app.post("/api/network/ei")
    def api_network_ei(req: NetworkRequest):
        return _safe(
            lambda: simulate_ei_network(
                n_exc=req.n_exc,
                n_inh=req.n_inh,
                w_ee=req.w_ee,
                w_ei=req.w_ei,
                w_ie=req.w_ie,
                w_ii=req.w_ii,
                p_conn=req.p_conn,
                ext_rate=req.ext_rate,
                duration=req.duration,
                dt=req.dt,
            )
        )

    # --- Compiler Inspector (Block 2) ---
    @app.post("/api/ir/build")
    def api_ir_build(req: SimulateRequest):
        return _safe(
            lambda: build_ir_from_equation(
                equations=req.equations,
                params=req.params,
                threshold=req.threshold,
                reset=req.reset,
                dt=req.dt,
            )
        )

    @app.post("/api/ir/verify")
    def api_ir_verify(data: dict):
        ir_text = data.get("ir_text", "")
        if not ir_text:
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: verify_ir(ir_text))

    @app.post("/api/ir/emit-sv")
    def api_ir_emit_sv(data: dict):
        ir_text = data.get("ir_text", "")
        if not ir_text:
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: emit_systemverilog(ir_text))

    @app.post("/api/ir/emit-sv-direct")
    def api_ir_emit_sv_direct(req: SimulateRequest):
        return _safe(
            lambda: emit_sv_from_equation(
                equations=req.equations,
                params=req.params,
                threshold=req.threshold,
                reset=req.reset,
            )
        )

    @app.post("/api/ir/cosim")
    def api_ir_cosim(req: PrecisionRequest):
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

    # --- Synthesis Dashboard (Block 3) ---
    @app.get("/api/synth/tools-status")
    def api_synth_tools():
        return check_tools()

    @app.post("/api/synth/run")
    def api_synth_run(data: dict):
        verilog = data.get("verilog", "")
        target = data.get("target", "ice40")
        if not verilog:
            raise HTTPException(422, "verilog source required")
        return _safe(lambda: run_synthesis(verilog, target))

    @app.post("/api/synth/multi-target")
    def api_synth_multi(data: dict):
        verilog = data.get("verilog", "")
        if not verilog:
            raise HTTPException(422, "verilog source required")
        return _safe(lambda: multi_target_synthesis(verilog))

    @app.post("/api/synth/estimate")
    def api_synth_estimate(data: dict):
        ir_op_count = data.get("ir_op_count", 0)
        target = data.get("target", "ice40")
        if ir_op_count < 1:
            raise HTTPException(422, "ir_op_count must be >= 1")
        return estimate_resources(ir_op_count, target)

    @app.post("/api/synth/pnr")
    def api_synth_pnr(data: dict):
        json_path = data.get("json_path", "")
        target = data.get("target", "ice40")
        if not json_path:
            raise HTTPException(422, "json_path required")
        return _safe(lambda: run_pnr(json_path, target))

    # --- Integration (Block 6) ---
    @app.post("/api/project/save")
    def api_project_save(data: dict):
        name = data.get("name", "")
        state = data.get("state", {})
        if not name:
            raise HTTPException(422, "Project name required")
        return save_project(name, state)

    @app.get("/api/project/list")
    def api_project_list():
        return list_projects()

    @app.get("/api/project/load/{name}")
    def api_project_load(name: str):
        result = load_project(name)
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @app.delete("/api/project/{name}")
    def api_project_delete(name: str):
        result = delete_project(name)
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @app.post("/api/pipeline/run")
    def api_pipeline_run(data: dict):
        graph = data.get("graph", {})
        target = data.get("target", "ice40")
        return _safe(lambda: run_pipeline(graph, target))

    # --- Network Canvas (Block 5) ---
    @app.get("/api/graph/models")
    def api_graph_models():
        return graph_available_models()

    @app.post("/api/graph/population")
    def api_create_population(data: dict):
        return create_population(
            **{
                k: v
                for k, v in data.items()
                if k in ("label", "model", "count", "neuron_type", "x", "y")
            }
        )

    @app.post("/api/graph/projection")
    def api_create_projection(data: dict):
        return _safe(
            lambda: create_projection(
                **{
                    k: v
                    for k, v in data.items()
                    if k in ("source_id", "target_id", "weight", "delay", "probability")
                }
            )
        )

    @app.post("/api/graph/validate")
    def api_validate_graph(data: dict):
        errors = validate_graph(data)
        return {"valid": len(errors) == 0, "errors": errors}

    @app.post("/api/graph/simulate")
    def api_simulate_graph(data: dict):
        return _safe(lambda: simulate_graph(data))

    @app.post("/api/graph/export-nir")
    def api_export_nir(data: dict):
        return graph_to_nir(data)

    @app.post("/api/graph/import-nir")
    def api_import_nir(data: dict):
        return nir_to_graph(data)

    # --- Training Monitor (Block 4) ---
    @app.get("/api/training/surrogates")
    def api_surrogates():
        return list_surrogates()

    @app.get("/api/training/cell-types")
    def api_cell_types():
        return list_cell_types()

    @app.post("/api/training/start")
    def api_training_start(data: dict):
        return _safe(lambda: start_training(data))

    @app.post("/api/training/stop")
    def api_training_stop(data: dict):
        job_id = data.get("job_id", "")
        if not job_id:
            raise HTTPException(422, "job_id required")
        return stop_training(job_id)

    @app.get("/api/training/jobs")
    def api_training_jobs():
        return list_jobs()

    @app.get("/api/training/status/{job_id}")
    def api_training_status(job_id: str):
        result = get_training_status(job_id)
        if result.get("error") and "job_id" not in result:
            raise HTTPException(404, result["error"])
        return result

    @app.get("/api/training/stream/{job_id}")
    def api_training_stream(job_id: str):
        from starlette.responses import StreamingResponse

        return StreamingResponse(
            stream_metrics(job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Static file serving for production mode ---
    import os

    dist_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "studio", "frontend", "dist"
    )
    if not os.path.isdir(dist_dir):
        dist_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "studio", "frontend", "dist"
        )
    if os.path.isdir(dist_dir):
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        @app.get("/")
        def serve_index():
            return FileResponse(os.path.join(dist_dir, "index.html"))

        app.mount("/", StaticFiles(directory=dist_dir), name="static")

    return app


app = create_app()
