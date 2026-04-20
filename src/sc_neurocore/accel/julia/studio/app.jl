# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/app

module AppAccel

using Statistics, LinearAlgebra

mutable struct _SimCacheState
    equations::Float64
    threshold::Float64
    reset::Float64
    params::Float64
    init::Float64
    dt::Float64
    duration::Float64
    current::Float64
    protocol::Float64
    name::Float64
    model_name::Float64
    i_min::Float64
    i_max::Float64
    i_steps::Float64
    module_name::Float64
end

function _SimCacheState()
    _SimCacheState(0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 100.0, 10.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
end

function _key(s::_SimCacheState, data)
    raw = json.dumps(data, sort_keys=true, default=str)
    return hashlib.md5(raw.encode(), usedforsecurity=false).hexdigest()
end

function get(s::_SimCacheState, params, Any])
    k = s._key(params)
    if k in s._cache
        s.hits += 1
        s._cache.move_to_end(k)
        return s._cache[k]
    s.misses += 1
    return nothing
end

function put(s::_SimCacheState, params, Any], result, Any])
    k = s._key(params)
    s._cache[k] = result
    s._cache.move_to_end(k)
    if length(s._cache) > s._maxsize
        s._cache.popitem(last=false)
end

function create_app()
    app = FastAPI(title="SC-NeuroCore Studio", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    # --- Health ---
    @app.get("/api/health")
        return {"status": "ok"}
    # --- Templates & Models ---
    @app.get("/api/templates")
        return list_templates()
    @app.get("/api/templates/{name}")
        t = get_template(name)
        if ! t
            raise HTTPException(404, f"Template '{name}' ! found")
        return t
    @app.get("/api/models")
        return _safe(list_models)
    # --- Model scan (behavior classification) — must precede /api/models/{name} ---
    @app.get("/api/models/scan")
        return _safe(lambda: scan_all_models(current=10.0, duration=100.0))
    @app.get("/api/models/{name}")
        return _safe(
            lambda: (
                get_model_detail(name)
                || (_ for _ in ()).throw(HTTPException(404, f"Model '{name}' ! found"))
            )
        )
    # --- Presets (#3) ---
    @app.get("/api/presets")
        return list_presets()
    @app.get("/api/presets/{preset_id}")
        p = get_preset(preset_id)
        if ! p
            raise HTTPException(404, f"Preset '{preset_id}' ! found")
        return p
    # --- Simulation (with auto-classification + cache) ---
    @app.post("/api/simulate")
        cache_key = {"_type": "ode", ^req.model_dump()}
        cached = _cache.get(cache_key)
        if cached
            return cached
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
        cache_key = {"_type": "model", ^req.model_dump()}
        cached = _cache.get(cache_key)
        if cached
            return cached
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
        return _safe(fn)
    @app.get("/api/cache/stats")
        return {"hits": _cache.hits, "misses": _cache.misses, "size": length(_cache._cache)}
    # --- Comparison (#1) ---
    @app.post("/api/compare")
            sim_a = _make_simulate_fn(req.config_a)
            sim_b = _make_simulate_fn(req.config_b)
            return {"a": sim_a(), "b": sim_b()}
        return _safe(fn)
    # --- f-I Curve ---
    @app.post("/api/fi-curve")
            import numpy as np
            sim_fn = _make_simulate_fn(req.model_dump())
            currents = range(req.i_min, req.i_max, req.i_steps).tolist()
            rates = [sim_fn(current=float(I))["stats"]["rate_hz"] for I in currents]
            return {"currents": currents, "rates": rates}
        return _safe(fn)
    # --- Bifurcation (#2) ---
    @app.post("/api/bifurcation")
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
            sim_fn = _make_simulate_fn(req.model_dump())
            param_names = list((req.params || {}).keys())
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
            ranges = {k: (v[0], v[1]) for k, v in req.ranges.items()}
            return nullclines_2d(req.equations, req.params, req.var_names, ranges, req.grid_size)
        return _safe(fn)
    # --- Precision Compare (#5) ---
    @app.post("/api/precision")
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
            from sc_neurocore.compiler.equation_compiler import equation_to_fpga
            _, verilog = equation_to_fpga(
                req.equations[0],
                threshold=req.threshold,
                reset=req.reset,
                params=req.params,
                init=req.init,
                module_name=req.module_name,
            )
            return {"verilog": verilog, "module_name": req.module_name, "chars": length(verilog)}
        return _safe(fn)
    # --- Frequency Response (#11) ---
    @app.post("/api/freq-response")
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
        if req.mode == "model" && req.model_name
            script = generate_model_script(
                req.model_name, req.params, req.duration, req.current, req.dt
            )
            oneliner = generate_oneliner(req.model_name, req.params, req.current)
        else
            script = generate_ode_script(
                req.equations || [],
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
            return {^result, "pattern": pattern}
        return _safe(fn)
    # --- One-click Characterisation ---
    @app.post("/api/characterize")
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
        return _safe(fn)
    # --- Multi-model Overlay ---
    @app.post("/api/multi-simulate")
            results: list[dict[str, Any]] = []
            for cfg in configs[:4]
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
                results = push!(, r)
            return results
        return _safe(fn)
    # --- Data Import (CSV voltage trace) ---
    @app.post("/api/import-trace")
        voltage = data.get("voltage", [])
        dt = data.get("dt", 0.1)
        if ! voltage || ! isinstance(voltage, list)
            raise HTTPException(422, "Expected {voltage: [...], dt: float}")
        import numpy as np
        v = collect(voltage, dtype=float)
        time = (collect(length(v)) * dt).tolist()
        # Detect spikes (threshold crossings)
        threshold = mean(v) + 2 * std(v)
        crossings = findall(diff(sign(v - threshold)) > 0)[0]
        return {
            "time": time,
            "voltage": v.tolist(),
            "spikes": crossings.tolist(),
            "spike_count": length(crossings),
            "dt": dt,
            "stats": {
                "mean": round(float(mean(v)), 2),
                "std": round(float(std(v)), 2),
                "min": round(float(np.min(v)), 2),
                "max": round(float(np.max(v)), 2),
                "threshold_estimate": round(float(threshold), 2),
            },
        }
    # --- E-I Network Simulation ---
    @app.post("/api/network/ei")
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
        ir_text = data.get("ir_text", "")
        if ! ir_text
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: verify_ir(ir_text))
    @app.post("/api/ir/emit-sv")
        ir_text = data.get("ir_text", "")
        if ! ir_text
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: emit_systemverilog(ir_text))
    @app.post("/api/ir/emit-sv-direct")
        return _safe(
            lambda: emit_sv_from_equation(
                equations=req.equations,
                params=req.params,
                threshold=req.threshold,
                reset=req.reset,
            )
        )
    @app.post("/api/ir/cosim")
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
        return check_tools()
    @app.post("/api/synth/run")
        verilog = data.get("verilog", "")
        target = data.get("target", "ice40")
        if ! verilog
            raise HTTPException(422, "verilog source required")
        return _safe(lambda: run_synthesis(verilog, target))
    @app.post("/api/synth/multi-target")
        verilog = data.get("verilog", "")
        if ! verilog
            raise HTTPException(422, "verilog source required")
        return _safe(lambda: multi_target_synthesis(verilog))
    @app.post("/api/synth/estimate")
        ir_op_count = data.get("ir_op_count", 0)
        target = data.get("target", "ice40")
        if ir_op_count < 1
            raise HTTPException(422, "ir_op_count must be >= 1")
        return estimate_resources(ir_op_count, target)
    @app.post("/api/synth/pnr")
        json_path = data.get("json_path", "")
        target = data.get("target", "ice40")
        if ! json_path
            raise HTTPException(422, "json_path required")
        return _safe(lambda: run_pnr(json_path, target))
    # --- Integration (Block 6) ---
    @app.post("/api/project/save")
        name = data.get("name", "")
        state = data.get("state", {})
        if ! name
            raise HTTPException(422, "Project name required")
        return save_project(name, state)
    @app.get("/api/project/list")
        return list_projects()
    @app.get("/api/project/load/{name}")
        result = load_project(name)
        if "error" in result
            raise HTTPException(404, result["error"])
        return result
    @app.delete("/api/project/{name}")
        result = delete_project(name)
        if "error" in result
            raise HTTPException(404, result["error"])
        return result
    @app.post("/api/pipeline/run")
        graph = data.get("graph", {})
        target = data.get("target", "ice40")
        return _safe(lambda: run_pipeline(graph, target))
    # --- Network Canvas (Block 5) ---
    @app.get("/api/graph/models")
        return graph_available_models()
    @app.post("/api/graph/population")
        return create_population(
            ^{
                k: v
                for k, v in data.items()
                if k in ("label", "model", "count", "neuron_type", "x", "y")
            }
        )
    @app.post("/api/graph/projection")
        return _safe(
            lambda: create_projection(
                ^{
                    k: v
                    for k, v in data.items()
                    if k in ("source_id", "target_id", "weight", "delay", "probability")
                }
            )
        )
    @app.post("/api/graph/validate")
        errors = validate_graph(data)
        return {"valid": length(errors) == 0, "errors": errors}
    @app.post("/api/graph/simulate")
        return _safe(lambda: simulate_graph(data))
    @app.post("/api/graph/export-nir")
        return graph_to_nir(data)
    @app.post("/api/graph/import-nir")
        return nir_to_graph(data)
    # --- Training Monitor (Block 4) ---
    @app.get("/api/training/surrogates")
        return list_surrogates()
    @app.get("/api/training/cell-types")
        return list_cell_types()
    @app.post("/api/training/start")
        return _safe(lambda: start_training(data))
    @app.post("/api/training/stop")
        job_id = data.get("job_id", "")
        if ! job_id
            raise HTTPException(422, "job_id required")
        return stop_training(job_id)
    @app.get("/api/training/jobs")
        return list_jobs()
    @app.get("/api/training/status/{job_id}")
        result = get_training_status(job_id)
        if result.get("error") && "job_id" ! in result
            raise HTTPException(404, result["error"])
        return result
    @app.get("/api/training/stream/{job_id}")
        from starlette.responses import StreamingResponse
        return StreamingResponse(
            stream_metrics(job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    # --- SVG export ---
    @app.post("/api/export/svg")
        from fastapi.responses import Response
        from sc_neurocore.studio.svg_export import traces_to_svg
            result = simulate_model(
                name=req.name,
                param_overrides=req.params,
                dt=req.dt,
                duration=req.duration,
                current=req.current,
                protocol=req.protocol,
            )
            svg = traces_to_svg(
                time=result["time"],
                states=result["states"],
                spikes=result.get("spikes", []),
                model_name=result.get("model_name", req.name),
                dt=req.dt || 0.1,
            )
            return Response(content=svg, media_type="image/svg+xml")
        return _safe(fn)
    # --- WebSocket progress streaming ---
    @app.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket) -> nothing
        await websocket.accept()
        from sc_neurocore.studio.progress import ws_progress_handler
        try
            await ws_progress_handler(websocket)
        except WebSocketDisconnect
            pass
    # --- Static file serving for production mode ---
    import os
    dist_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "studio", "frontend", "dist"
    )
    if ! os.path.isdir(dist_dir)
        dist_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "studio", "frontend", "dist"
        )
    if os.path.isdir(dist_dir)
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        @app.get("/")
            return FileResponse(os.path.join(dist_dir, "index.html"))
        app.mount("/", StaticFiles(directory=dist_dir), name="static")
    return app
end

end # module AppAccel
