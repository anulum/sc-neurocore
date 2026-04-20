# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for app

fn _safe(fn: Int) -> Int:
    var __safe_line = 'try:'
    return 0  # return fn()
    var __safe_line = 'except HTTPException:'
    var __safe_line = 'raise'
    var __safe_line = 'except (ValueError, TypeError, KeyError):'
    var __safe_line = 'raise HTTPException(status_code=422, detail="Invalid input")'
    var __safe_line = 'except Exception:'
    var __safe_line = 'raise HTTPException(status_code=500, detail="Internal error"'

fn _make_simulate_fn(req_dict: Int) -> Int:
    var __make_simulate_fn_line = 'if req_dict.get("model_name"):'
    var __make_simulate_fn_line = 'cfg = {'
    var __make_simulate_fn_line = '"name": req_dict["model_name"],'
    var __make_simulate_fn_line = '"param_overrides": overrides.get("params", req_dict.get("par'
    var __make_simulate_fn_line = '"dt": overrides.get("dt", req_dict.get("dt")),'
    var __make_simulate_fn_line = '"duration": overrides.get("duration", req_dict.get("duration'
    var __make_simulate_fn_line = '"current": overrides.get("current", req_dict.get("current", '
    var __make_simulate_fn_line = '"protocol": overrides.get("protocol", req_dict.get("protocol'
    var __make_simulate_fn_line = '}'
    return 0  # return simulate_model(**cfg)
    return 0  # return fn
    var __make_simulate_fn_line = 'else:'
    return 0  # return simulate(
    var __make_simulate_fn_line = 'equations=req_dict.get("equations", []),'
    var __make_simulate_fn_line = 'threshold=req_dict.get("threshold"),'
    var __make_simulate_fn_line = 'reset=req_dict.get("reset"),'
    var __make_simulate_fn_line = 'params=overrides.get("params", req_dict.get("params")),'
    var __make_simulate_fn_line = 'init=overrides.get("init", req_dict.get("init")),'
    var __make_simulate_fn_line = 'dt=overrides.get("dt", req_dict.get("dt", 0.1)),'
    var __make_simulate_fn_line = 'duration=overrides.get("duration", req_dict.get("duration", '
    var __make_simulate_fn_line = 'current=overrides.get("current", req_dict.get("current", 10)'
    var __make_simulate_fn_line = 'protocol=overrides.get("protocol", req_dict.get("protocol", '
    var __make_simulate_fn_line = ')'
    return 0  # return fn

fn create_app() -> Int:
    var _create_app_line = 'app = FastAPI(title="SC-NeuroCore Studio", version="1.0.0")'
    var _create_app_line = 'app.add_middleware('
    var _create_app_line = 'CORSMiddleware, allow_origins=["*"], allow_methods=["*"], al'
    var _create_app_line = ')'
    var _create_app_line = '# --- Health ---'
    var _create_app_line = '@app.get("/api/health")'
    return 0  # return {"status": "ok"}
    var _create_app_line = '# --- Templates & Models ---'
    var _create_app_line = '@app.get("/api/templates")'
    return 0  # return list_templates()
    var _create_app_line = '@app.get("/api/templates/{name}")'
    var _create_app_line = 't = get_template(name)'
    var _create_app_line = 'if not t:'
    var _create_app_line = 'raise HTTPException(404, f"Template \'{name}\' not found")'
    return 0  # return t
    var _create_app_line = '@app.get("/api/models")'
    return 0  # return _safe(list_models)
    var _create_app_line = '# --- Model scan (behavior classification) — must precede /a'
    var _create_app_line = '@app.get("/api/models/scan")'
    return 0  # return _safe(lambda: scan_all_models(current=10.0,
    var _create_app_line = '@app.get("/api/models/{name}")'
    return 0  # return _safe(
    var _create_app_line = 'lambda: ('
    var _create_app_line = 'get_model_detail(name)'
    var _create_app_line = 'or (_ for _ in ()).throw(HTTPException(404, f"Model \'{name}\''
    var _create_app_line = ')'
    var _create_app_line = ')'
    var _create_app_line = '# --- Presets (#3) ---'
    var _create_app_line = '@app.get("/api/presets")'
    return 0  # return list_presets()
    var _create_app_line = '@app.get("/api/presets/{preset_id}")'
    var _create_app_line = 'p = get_preset(preset_id)'
    var _create_app_line = 'if not p:'
    var _create_app_line = 'raise HTTPException(404, f"Preset \'{preset_id}\' not found")'
    return 0  # return p
    var _create_app_line = '# --- Simulation (with auto-classification + cache) ---'
    var _create_app_line = '@app.post("/api/simulate")'
    var _create_app_line = 'cache_key = {"_type": "ode", **req.model_dump()}'
    var _create_app_line = 'cached = _cache.get(cache_key)'
    var _create_app_line = 'if cached:'
    return 0  # return cached
    var _create_app_line = 'result = simulate('
    var _create_app_line = 'equations=req.equations,'
    var _create_app_line = 'threshold=req.threshold,'
    var _create_app_line = 'reset=req.reset,'
    var _create_app_line = 'params=req.params,'
    var _create_app_line = 'init=req.init,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = 'duration=req.duration,'
    var _create_app_line = 'current=req.current,'
    var _create_app_line = 'protocol=req.protocol,'
    var _create_app_line = ')'
    var _create_app_line = 'result["pattern"] = classify_firing_pattern('
    var _create_app_line = 'result["spikes"], result["n_steps"], result["dt"]'
    var _create_app_line = ')'
    var _create_app_line = '_cache.put(cache_key, result)'
    return 0  # return result
    return 0  # return _safe(fn)
    var _create_app_line = '@app.post("/api/models/simulate")'
    var _create_app_line = 'cache_key = {"_type": "model", **req.model_dump()}'
    var _create_app_line = 'cached = _cache.get(cache_key)'
    var _create_app_line = 'if cached:'
    return 0  # return cached
    var _create_app_line = 'result = simulate_model('
    var _create_app_line = 'name=req.name,'
    var _create_app_line = 'param_overrides=req.params,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = 'duration=req.duration,'
    var _create_app_line = 'current=req.current,'
    var _create_app_line = 'protocol=req.protocol,'
    var _create_app_line = ')'
    var _create_app_line = 'result["pattern"] = classify_firing_pattern('
    var _create_app_line = 'result["spikes"], result["n_steps"], result["dt"]'
    var _create_app_line = ')'
    var _create_app_line = '_cache.put(cache_key, result)'
    return 0  # return result
    return 0  # return _safe(fn)
    var _create_app_line = '@app.get("/api/cache/stats")'
    return 0  # return {"hits": _cache.hits, "misses": _cache.miss
    var _create_app_line = '# --- Comparison (#1) ---'
    var _create_app_line = '@app.post("/api/compare")'
    var _create_app_line = 'sim_a = _make_simulate_fn(req.config_a)'
    var _create_app_line = 'sim_b = _make_simulate_fn(req.config_b)'
    return 0  # return {"a": sim_a(), "b": sim_b()}
    return 0  # return _safe(fn)
    var _create_app_line = '# --- f-I Curve ---'
    var _create_app_line = '@app.post("/api/fi-curve")'
    var _create_app_line = 'import numpy as np'
    var _create_app_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _create_app_line = 'currents = linspace(req.i_min, req.i_max, req.i_steps).tolis'
    var _create_app_line = 'rates = [sim_fn(current=float(I))["stats"]["rate_hz"] for I '
    return 0  # return {"currents": currents, "rates": rates}
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Bifurcation (#2) ---'
    var _create_app_line = '@app.post("/api/bifurcation")'
    var _create_app_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _create_app_line = 'base_cfg = {'
    var _create_app_line = '"params": req.params,'
    var _create_app_line = '"init": req.init,'
    var _create_app_line = '"dt": req.dt,'
    var _create_app_line = '"duration": req.duration,'
    var _create_app_line = '"current": req.current,'
    var _create_app_line = '"protocol": "constant",'
    var _create_app_line = '}'
    return 0  # return bifurcation_sweep(
    var _create_app_line = 'sim_fn, base_cfg, req.sweep_param, req.sweep_min, req.sweep_'
    var _create_app_line = ')'
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Sensitivity (#8) ---'
    var _create_app_line = '@app.post("/api/sensitivity")'
    var _create_app_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _create_app_line = 'param_names = list((req.params or {}).keys())'
    var _create_app_line = 'base_cfg = {'
    var _create_app_line = '"params": req.params,'
    var _create_app_line = '"init": req.init,'
    var _create_app_line = '"dt": req.dt,'
    var _create_app_line = '"duration": req.duration,'
    var _create_app_line = '"current": req.current,'
    var _create_app_line = '"protocol": "constant",'
    var _create_app_line = '}'
    return 0  # return sensitivity_analysis(sim_fn, base_cfg, para
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Nullclines (#9) ---'
    var _create_app_line = '@app.post("/api/nullclines")'
    var _create_app_line = 'ranges = {k: (v[0], v[1]) for k, v in req.ranges.items()}'
    return 0  # return nullclines_2d(req.equations, req.params, re
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Precision Compare (#5) ---'
    var _create_app_line = '@app.post("/api/precision")'
    return 0  # return _safe(
    var _create_app_line = 'lambda: precision_compare('
    var _create_app_line = 'equations=req.equations,'
    var _create_app_line = 'threshold=req.threshold,'
    var _create_app_line = 'reset=req.reset,'
    var _create_app_line = 'params=req.params,'
    var _create_app_line = 'init=req.init,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = 'duration=req.duration,'
    var _create_app_line = 'current=req.current,'
    var _create_app_line = ')'
    var _create_app_line = ')'
    var _create_app_line = '# --- Compile (#5 adjacent) ---'
    var _create_app_line = '@app.post("/api/compile")'
    var _create_app_line = 'from sc_neurocore.compiler.equation_compiler import equation'
    var _create_app_line = '_, verilog = equation_to_fpga('
    var _create_app_line = 'req.equations[0],'
    var _create_app_line = 'threshold=req.threshold,'
    var _create_app_line = 'reset=req.reset,'
    var _create_app_line = 'params=req.params,'
    var _create_app_line = 'init=req.init,'
    var _create_app_line = 'module_name=req.module_name,'
    var _create_app_line = ')'
    return 0  # return {"verilog": verilog, "module_name": req.mod
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Frequency Response (#11) ---'
    var _create_app_line = '@app.post("/api/freq-response")'
    var _create_app_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _create_app_line = 'base_cfg = {'
    var _create_app_line = '"params": req.params,'
    var _create_app_line = '"init": req.init,'
    var _create_app_line = '"dt": req.dt,'
    var _create_app_line = '"duration": req.duration,'
    var _create_app_line = '"current": req.amplitude,'
    var _create_app_line = '"protocol": "constant",'
    var _create_app_line = '}'
    return 0  # return frequency_response(
    var _create_app_line = 'sim_fn, base_cfg, req.freq_min, req.freq_max, req.n_freqs, r'
    var _create_app_line = ')'
    return 0  # return _safe(fn)
    var _create_app_line = '# --- 2D Heatmap ---'
    var _create_app_line = '@app.post("/api/heatmap")'
    var _create_app_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _create_app_line = 'base_cfg = {'
    var _create_app_line = '"params": req.params,'
    var _create_app_line = '"init": req.init,'
    var _create_app_line = '"dt": req.dt,'
    var _create_app_line = '"duration": req.duration,'
    var _create_app_line = '"current": req.current,'
    var _create_app_line = '"protocol": "constant",'
    var _create_app_line = '}'
    return 0  # return heatmap_2d(
    var _create_app_line = 'sim_fn,'
    var _create_app_line = 'base_cfg,'
    var _create_app_line = 'req.param_x,'
    var _create_app_line = 'req.x_min,'
    var _create_app_line = 'req.x_max,'
    var _create_app_line = 'req.x_steps,'
    var _create_app_line = 'req.param_y,'
    var _create_app_line = 'req.y_min,'
    var _create_app_line = 'req.y_max,'
    var _create_app_line = 'req.y_steps,'
    var _create_app_line = ')'
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Code Generation ---'
    var _create_app_line = '@app.post("/api/codegen")'
    var _create_app_line = 'if req.mode == "model" and req.model_name:'
    var _create_app_line = 'script = generate_model_script('
    var _create_app_line = 'req.model_name, req.params, req.duration, req.current, req.d'
    var _create_app_line = ')'
    var _create_app_line = 'oneliner = generate_oneliner(req.model_name, req.params, req'
    var _create_app_line = 'else:'
    var _create_app_line = 'script = generate_ode_script('
    var _create_app_line = 'req.equations or [],'
    var _create_app_line = 'req.threshold,'
    var _create_app_line = 'req.reset,'
    var _create_app_line = 'req.params,'
    var _create_app_line = 'req.init,'
    var _create_app_line = 'req.duration,'
    var _create_app_line = 'req.current,'
    var _create_app_line = 'req.dt,'
    var _create_app_line = ')'
    var _create_app_line = 'oneliner = ""'
    return 0  # return {"script": script, "oneliner": oneliner}
    var _create_app_line = '# --- Firing Pattern Classification ---'
    var _create_app_line = '@app.post("/api/classify")'
    var _create_app_line = 'result = simulate('
    var _create_app_line = 'equations=req.equations,'
    var _create_app_line = 'threshold=req.threshold,'
    var _create_app_line = 'reset=req.reset,'
    var _create_app_line = 'params=req.params,'
    var _create_app_line = 'init=req.init,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = 'duration=req.duration,'
    var _create_app_line = 'current=req.current,'
    var _create_app_line = 'protocol=req.protocol,'
    var _create_app_line = ')'
    var _create_app_line = 'pattern = classify_firing_pattern(result["spikes"], result["'
    return 0  # return {**result, "pattern": pattern}
    return 0  # return _safe(fn)
    var _create_app_line = '# --- One-click Characterisation ---'
    var _create_app_line = '@app.post("/api/characterize")'
    var _create_app_line = 'sim_fn = _make_simulate_fn('
    var _create_app_line = '{'
    var _create_app_line = '"model_name": req.name,'
    var _create_app_line = '"params": req.params,'
    var _create_app_line = '"dt": req.dt,'
    var _create_app_line = '"duration": req.duration,'
    var _create_app_line = '"current": req.current,'
    var _create_app_line = '"protocol": "constant",'
    var _create_app_line = '}'
    var _create_app_line = ')'
    var _create_app_line = 'base_cfg = {'
    var _create_app_line = '"params": req.params,'
    var _create_app_line = '"dt": req.dt,'
    var _create_app_line = '"duration": req.duration,'
    var _create_app_line = '"current": req.current,'
    var _create_app_line = '"protocol": "constant",'
    var _create_app_line = '}'
    return 0  # return characterize_model(sim_fn, base_cfg)
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Multi-model Overlay ---'
    var _create_app_line = '@app.post("/api/multi-simulate")'
    var _create_app_line = 'results: list[dict[str, Any]] = []'
    var _create_app_line = 'for cfg in configs[:4]:'
    var _create_app_line = 'sim_fn = _make_simulate_fn('
    var _create_app_line = '{'
    var _create_app_line = '"model_name": cfg.name,'
    var _create_app_line = '"params": cfg.params,'
    var _create_app_line = '"dt": cfg.dt,'
    var _create_app_line = '"duration": cfg.duration,'
    var _create_app_line = '"current": cfg.current,'
    var _create_app_line = '"protocol": cfg.protocol,'
    var _create_app_line = '}'
    var _create_app_line = ')'
    var _create_app_line = 'r = sim_fn()'
    var _create_app_line = 'r["pattern"] = classify_firing_pattern(r["spikes"], r["n_ste'
    var _create_app_line = 'results.append(r)'
    return 0  # return results
    return 0  # return _safe(fn)
    var _create_app_line = '# --- Data Import (CSV voltage trace) ---'
    var _create_app_line = '@app.post("/api/import-trace")'
    var _create_app_line = 'voltage = data.get("voltage", [])'
    var _create_app_line = 'dt = data.get("dt", 0.1)'
    var _create_app_line = 'if not voltage or not isinstance(voltage, list):'
    var _create_app_line = 'raise HTTPException(422, "Expected {voltage: [...], dt: floa'
    var _create_app_line = 'import numpy as np'
    var _create_app_line = 'v = array(voltage, dtype=float)'
    var _create_app_line = 'time = (arange(len(v)) * dt).tolist()'
    var _create_app_line = '# Detect spikes (threshold crossings)'
    var _create_app_line = 'threshold = mean(v) + 2 * std(v)'
    var _create_app_line = 'crossings = where(diff(sign(v - threshold)) > 0)[0]'
    return 0  # return {
    var _create_app_line = '"time": time,'
    var _create_app_line = '"voltage": v.tolist(),'
    var _create_app_line = '"spikes": crossings.tolist(),'
    var _create_app_line = '"spike_count": len(crossings),'
    var _create_app_line = '"dt": dt,'
    var _create_app_line = '"stats": {'
    var _create_app_line = '"mean": round(float(mean(v)), 2),'
    var _create_app_line = '"std": round(float(std(v)), 2),'
    var _create_app_line = '"min": round(float(min(v)), 2),'
    var _create_app_line = '"max": round(float(max(v)), 2),'
    var _create_app_line = '"threshold_estimate": round(float(threshold), 2),'
    var _create_app_line = '},'
    var _create_app_line = '}'
    var _create_app_line = '# --- E-I Network Simulation ---'
    var _create_app_line = '@app.post("/api/network/ei")'
    return 0  # return _safe(
    var _create_app_line = 'lambda: simulate_ei_network('
    var _create_app_line = 'n_exc=req.n_exc,'
    var _create_app_line = 'n_inh=req.n_inh,'
    var _create_app_line = 'w_ee=req.w_ee,'
    var _create_app_line = 'w_ei=req.w_ei,'
    var _create_app_line = 'w_ie=req.w_ie,'
    var _create_app_line = 'w_ii=req.w_ii,'
    var _create_app_line = 'p_conn=req.p_conn,'
    var _create_app_line = 'ext_rate=req.ext_rate,'
    var _create_app_line = 'duration=req.duration,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = ')'
    var _create_app_line = ')'
    var _create_app_line = '# --- Compiler Inspector (Block 2) ---'
    var _create_app_line = '@app.post("/api/ir/build")'
    return 0  # return _safe(
    var _create_app_line = 'lambda: build_ir_from_equation('
    var _create_app_line = 'equations=req.equations,'
    var _create_app_line = 'params=req.params,'
    var _create_app_line = 'threshold=req.threshold,'
    var _create_app_line = 'reset=req.reset,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = ')'
    var _create_app_line = ')'
    var _create_app_line = '@app.post("/api/ir/verify")'
    var _create_app_line = 'ir_text = data.get("ir_text", "")'
    var _create_app_line = 'if not ir_text:'
    var _create_app_line = 'raise HTTPException(422, "ir_text required")'
    return 0  # return _safe(lambda: verify_ir(ir_text))
    var _create_app_line = '@app.post("/api/ir/emit-sv")'
    var _create_app_line = 'ir_text = data.get("ir_text", "")'
    var _create_app_line = 'if not ir_text:'
    var _create_app_line = 'raise HTTPException(422, "ir_text required")'
    return 0  # return _safe(lambda: emit_systemverilog(ir_text))
    var _create_app_line = '@app.post("/api/ir/emit-sv-direct")'
    return 0  # return _safe(
    var _create_app_line = 'lambda: emit_sv_from_equation('
    var _create_app_line = 'equations=req.equations,'
    var _create_app_line = 'params=req.params,'
    var _create_app_line = 'threshold=req.threshold,'
    var _create_app_line = 'reset=req.reset,'
    var _create_app_line = ')'
    var _create_app_line = ')'
    var _create_app_line = '@app.post("/api/ir/cosim")'
    return 0  # return _safe(
    var _create_app_line = 'lambda: cosim_traces('
    var _create_app_line = 'equations=req.equations,'
    var _create_app_line = 'threshold=req.threshold,'
    var _create_app_line = 'reset=req.reset,'
    var _create_app_line = 'params=req.params,'
    var _create_app_line = 'init=req.init,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = 'duration=req.duration,'
    var _create_app_line = 'current=req.current,'
    var _create_app_line = ')'
    var _create_app_line = ')'
    var _create_app_line = '# --- Synthesis Dashboard (Block 3) ---'
    var _create_app_line = '@app.get("/api/synth/tools-status")'
    return 0  # return check_tools()
    var _create_app_line = '@app.post("/api/synth/run")'
    var _create_app_line = 'verilog = data.get("verilog", "")'
    var _create_app_line = 'target = data.get("target", "ice40")'
    var _create_app_line = 'if not verilog:'
    var _create_app_line = 'raise HTTPException(422, "verilog source required")'
    return 0  # return _safe(lambda: run_synthesis(verilog, target
    var _create_app_line = '@app.post("/api/synth/multi-target")'
    var _create_app_line = 'verilog = data.get("verilog", "")'
    var _create_app_line = 'if not verilog:'
    var _create_app_line = 'raise HTTPException(422, "verilog source required")'
    return 0  # return _safe(lambda: multi_target_synthesis(verilo
    var _create_app_line = '@app.post("/api/synth/estimate")'
    var _create_app_line = 'ir_op_count = data.get("ir_op_count", 0)'
    var _create_app_line = 'target = data.get("target", "ice40")'
    var _create_app_line = 'if ir_op_count < 1:'
    var _create_app_line = 'raise HTTPException(422, "ir_op_count must be >= 1")'
    return 0  # return estimate_resources(ir_op_count, target)
    var _create_app_line = '@app.post("/api/synth/pnr")'
    var _create_app_line = 'json_path = data.get("json_path", "")'
    var _create_app_line = 'target = data.get("target", "ice40")'
    var _create_app_line = 'if not json_path:'
    var _create_app_line = 'raise HTTPException(422, "json_path required")'
    return 0  # return _safe(lambda: run_pnr(json_path, target))
    var _create_app_line = '# --- Integration (Block 6) ---'
    var _create_app_line = '@app.post("/api/project/save")'
    var _create_app_line = 'name = data.get("name", "")'
    var _create_app_line = 'state = data.get("state", {})'
    var _create_app_line = 'if not name:'
    var _create_app_line = 'raise HTTPException(422, "Project name required")'
    return 0  # return save_project(name, state)
    var _create_app_line = '@app.get("/api/project/list")'
    return 0  # return list_projects()
    var _create_app_line = '@app.get("/api/project/load/{name}")'
    var _create_app_line = 'result = load_project(name)'
    var _create_app_line = 'if "error" in result:'
    var _create_app_line = 'raise HTTPException(404, result["error"])'
    return 0  # return result
    var _create_app_line = '@app.delete("/api/project/{name}")'
    var _create_app_line = 'result = delete_project(name)'
    var _create_app_line = 'if "error" in result:'
    var _create_app_line = 'raise HTTPException(404, result["error"])'
    return 0  # return result
    var _create_app_line = '@app.post("/api/pipeline/run")'
    var _create_app_line = 'graph = data.get("graph", {})'
    var _create_app_line = 'target = data.get("target", "ice40")'
    return 0  # return _safe(lambda: run_pipeline(graph, target))
    var _create_app_line = '# --- Network Canvas (Block 5) ---'
    var _create_app_line = '@app.get("/api/graph/models")'
    return 0  # return graph_available_models()
    var _create_app_line = '@app.post("/api/graph/population")'
    return 0  # return create_population(
    var _create_app_line = '**{'
    var _create_app_line = 'k: v'
    var _create_app_line = 'for k, v in data.items()'
    var _create_app_line = 'if k in ("label", "model", "count", "neuron_type", "x", "y")'
    var _create_app_line = '}'
    var _create_app_line = ')'
    var _create_app_line = '@app.post("/api/graph/projection")'
    return 0  # return _safe(
    var _create_app_line = 'lambda: create_projection('
    var _create_app_line = '**{'
    var _create_app_line = 'k: v'
    var _create_app_line = 'for k, v in data.items()'
    var _create_app_line = 'if k in ("source_id", "target_id", "weight", "delay", "proba'
    var _create_app_line = '}'
    var _create_app_line = ')'
    var _create_app_line = ')'
    var _create_app_line = '@app.post("/api/graph/validate")'
    var _create_app_line = 'errors = validate_graph(data)'
    return 0  # return {"valid": len(errors) == 0, "errors": error
    var _create_app_line = '@app.post("/api/graph/simulate")'
    return 0  # return _safe(lambda: simulate_graph(data))
    var _create_app_line = '@app.post("/api/graph/export-nir")'
    return 0  # return graph_to_nir(data)
    var _create_app_line = '@app.post("/api/graph/import-nir")'
    return 0  # return nir_to_graph(data)
    var _create_app_line = '# --- Training Monitor (Block 4) ---'
    var _create_app_line = '@app.get("/api/training/surrogates")'
    return 0  # return list_surrogates()
    var _create_app_line = '@app.get("/api/training/cell-types")'
    return 0  # return list_cell_types()
    var _create_app_line = '@app.post("/api/training/start")'
    return 0  # return _safe(lambda: start_training(data))
    var _create_app_line = '@app.post("/api/training/stop")'
    var _create_app_line = 'job_id = data.get("job_id", "")'
    var _create_app_line = 'if not job_id:'
    var _create_app_line = 'raise HTTPException(422, "job_id required")'
    return 0  # return stop_training(job_id)
    var _create_app_line = '@app.get("/api/training/jobs")'
    return 0  # return list_jobs()
    var _create_app_line = '@app.get("/api/training/status/{job_id}")'
    var _create_app_line = 'result = get_training_status(job_id)'
    var _create_app_line = 'if result.get("error") and "job_id" not in result:'
    var _create_app_line = 'raise HTTPException(404, result["error"])'
    return 0  # return result
    var _create_app_line = '@app.get("/api/training/stream/{job_id}")'
    var _create_app_line = 'from starlette.responses import StreamingResponse'
    return 0  # return StreamingResponse(
    var _create_app_line = 'stream_metrics(job_id),'
    var _create_app_line = 'media_type="text/event-stream",'
    var _create_app_line = 'headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "'
    var _create_app_line = ')'
    var _create_app_line = '# --- SVG export ---'
    var _create_app_line = '@app.post("/api/export/svg")'
    var _create_app_line = 'from fastapi.responses import Response'
    var _create_app_line = 'from sc_neurocore.studio.svg_export import traces_to_svg'
    var _create_app_line = 'result = simulate_model('
    var _create_app_line = 'name=req.name,'
    var _create_app_line = 'param_overrides=req.params,'
    var _create_app_line = 'dt=req.dt,'
    var _create_app_line = 'duration=req.duration,'
    var _create_app_line = 'current=req.current,'
    var _create_app_line = 'protocol=req.protocol,'
    var _create_app_line = ')'
    var _create_app_line = 'svg = traces_to_svg('
    var _create_app_line = 'time=result["time"],'
    var _create_app_line = 'states=result["states"],'
    var _create_app_line = 'spikes=result.get("spikes", []),'
    var _create_app_line = 'model_name=result.get("model_name", req.name),'
    var _create_app_line = 'dt=req.dt or 0.1,'
    var _create_app_line = ')'
    return 0  # return Response(content=svg, media_type="image/svg
    return 0  # return _safe(fn)
    var _create_app_line = '# --- WebSocket progress streaming ---'
    var _create_app_line = '@app.websocket("/ws/progress")'
    var _create_app_line = 'async def ws_progress(websocket: WebSocket) -> 0:'
    var _create_app_line = 'await websocket.accept()'
    var _create_app_line = 'from sc_neurocore.studio.progress import ws_progress_handler'
    var _create_app_line = 'try:'
    var _create_app_line = 'await ws_progress_handler(websocket)'
    var _create_app_line = 'except WebSocketDisconnect:'
    var _create_app_line = 'pass'
    var _create_app_line = '# --- Static file serving for production mode ---'
    var _create_app_line = 'import os'
    var _create_app_line = 'dist_dir = os.path.join('
    var _create_app_line = 'os.path.dirname(__file__), "..", "..", "..", "studio", "fron'
    var _create_app_line = ')'
    var _create_app_line = 'if not os.path.isdir(dist_dir):'
    var _create_app_line = 'dist_dir = os.path.join('
    var _create_app_line = 'os.path.dirname(__file__), "..", "..", "..", "..", "studio",'
    var _create_app_line = ')'
    var _create_app_line = 'if os.path.isdir(dist_dir):'
    var _create_app_line = 'from fastapi.staticfiles import StaticFiles'
    var _create_app_line = 'from fastapi.responses import FileResponse'
    var _create_app_line = '@app.get("/")'
    return 0  # return FileResponse(os.path.join(dist_dir, "index.
    var _create_app_line = 'app.mount("/", StaticFiles(directory=dist_dir), name="static'
    return 0  # return app

fn _key(data: Int) -> Int:
    var __key_line = 'raw = json.dumps(data, sort_keys=True, default=str)'
    return 0  # return hashlib.md5(raw.encode(), usedforsecurity=F

fn get(params: Int) -> Int:
    var _get_line = 'k = _key(params)'
    var _get_line = 'if k in _cache:'
    var _get_line = 'hits += 1'
    var _get_line = '_cache.move_to_end(k)'
    return 0  # return _cache[k]
    var _get_line = 'misses += 1'
    return 0  # return 0

fn put(params: Int, result: Int) -> Int:
    var _put_line = 'k = _key(params)'
    var _put_line = '_cache[k] = result'
    var _put_line = '_cache.move_to_end(k)'
    var _put_line = 'if len(_cache) > _maxsize:'
    var _put_line = '_cache.popitem(last=False)'
    return 0

fn health() -> Int:
    return 0  # return {"status": "ok"}

fn api_templates() -> Int:
    return 0  # return list_templates()

fn api_template(name: Int) -> Int:
    var _api_template_line = 't = get_template(name)'
    var _api_template_line = 'if not t:'
    var _api_template_line = 'raise HTTPException(404, f"Template \'{name}\' not found")'
    return 0  # return t

fn api_models() -> Int:
    return 0  # return _safe(list_models)

fn api_model_scan() -> Int:
    return 0  # return _safe(lambda: scan_all_models(current=10.0,

fn api_model(name: Int) -> Int:
    return 0  # return _safe(
    var _api_model_line = 'lambda: ('
    var _api_model_line = 'get_model_detail(name)'
    var _api_model_line = 'or (_ for _ in ()).throw(HTTPException(404, f"Model \'{name}\''
    var _api_model_line = ')'
    var _api_model_line = ')'

fn api_presets() -> Int:
    return 0  # return list_presets()

fn api_preset(preset_id: Int) -> Int:
    var _api_preset_line = 'p = get_preset(preset_id)'
    var _api_preset_line = 'if not p:'
    var _api_preset_line = 'raise HTTPException(404, f"Preset \'{preset_id}\' not found")'
    return 0  # return p

fn api_simulate(req: Int) -> Int:
    var _api_simulate_line = 'cache_key = {"_type": "ode", **req.model_dump()}'
    var _api_simulate_line = 'cached = _cache.get(cache_key)'
    var _api_simulate_line = 'if cached:'
    return 0  # return cached
    var _api_simulate_line = 'result = simulate('
    var _api_simulate_line = 'equations=req.equations,'
    var _api_simulate_line = 'threshold=req.threshold,'
    var _api_simulate_line = 'reset=req.reset,'
    var _api_simulate_line = 'params=req.params,'
    var _api_simulate_line = 'init=req.init,'
    var _api_simulate_line = 'dt=req.dt,'
    var _api_simulate_line = 'duration=req.duration,'
    var _api_simulate_line = 'current=req.current,'
    var _api_simulate_line = 'protocol=req.protocol,'
    var _api_simulate_line = ')'
    var _api_simulate_line = 'result["pattern"] = classify_firing_pattern('
    var _api_simulate_line = 'result["spikes"], result["n_steps"], result["dt"]'
    var _api_simulate_line = ')'
    var _api_simulate_line = '_cache.put(cache_key, result)'
    return 0  # return result
    return 0  # return _safe(fn)

fn api_model_simulate(req: Int) -> Int:
    var _api_model_simulate_line = 'cache_key = {"_type": "model", **req.model_dump()}'
    var _api_model_simulate_line = 'cached = _cache.get(cache_key)'
    var _api_model_simulate_line = 'if cached:'
    return 0  # return cached
    var _api_model_simulate_line = 'result = simulate_model('
    var _api_model_simulate_line = 'name=req.name,'
    var _api_model_simulate_line = 'param_overrides=req.params,'
    var _api_model_simulate_line = 'dt=req.dt,'
    var _api_model_simulate_line = 'duration=req.duration,'
    var _api_model_simulate_line = 'current=req.current,'
    var _api_model_simulate_line = 'protocol=req.protocol,'
    var _api_model_simulate_line = ')'
    var _api_model_simulate_line = 'result["pattern"] = classify_firing_pattern('
    var _api_model_simulate_line = 'result["spikes"], result["n_steps"], result["dt"]'
    var _api_model_simulate_line = ')'
    var _api_model_simulate_line = '_cache.put(cache_key, result)'
    return 0  # return result
    return 0  # return _safe(fn)

fn api_cache_stats() -> Int:
    return 0  # return {"hits": _cache.hits, "misses": _cache.miss

fn api_compare(req: Int) -> Int:
    var _api_compare_line = 'sim_a = _make_simulate_fn(req.config_a)'
    var _api_compare_line = 'sim_b = _make_simulate_fn(req.config_b)'
    return 0  # return {"a": sim_a(), "b": sim_b()}
    return 0  # return _safe(fn)

fn api_fi_curve(req: Int) -> Int:
    var _api_fi_curve_line = 'import numpy as np'
    var _api_fi_curve_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _api_fi_curve_line = 'currents = linspace(req.i_min, req.i_max, req.i_steps).tolis'
    var _api_fi_curve_line = 'rates = [sim_fn(current=float(I))["stats"]["rate_hz"] for I '
    return 0  # return {"currents": currents, "rates": rates}
    return 0  # return _safe(fn)

fn api_bifurcation(req: Int) -> Int:
    var _api_bifurcation_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _api_bifurcation_line = 'base_cfg = {'
    var _api_bifurcation_line = '"params": req.params,'
    var _api_bifurcation_line = '"init": req.init,'
    var _api_bifurcation_line = '"dt": req.dt,'
    var _api_bifurcation_line = '"duration": req.duration,'
    var _api_bifurcation_line = '"current": req.current,'
    var _api_bifurcation_line = '"protocol": "constant",'
    var _api_bifurcation_line = '}'
    return 0  # return bifurcation_sweep(
    var _api_bifurcation_line = 'sim_fn, base_cfg, req.sweep_param, req.sweep_min, req.sweep_'
    var _api_bifurcation_line = ')'
    return 0  # return _safe(fn)

fn api_sensitivity(req: Int) -> Int:
    var _api_sensitivity_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _api_sensitivity_line = 'param_names = list((req.params or {}).keys())'
    var _api_sensitivity_line = 'base_cfg = {'
    var _api_sensitivity_line = '"params": req.params,'
    var _api_sensitivity_line = '"init": req.init,'
    var _api_sensitivity_line = '"dt": req.dt,'
    var _api_sensitivity_line = '"duration": req.duration,'
    var _api_sensitivity_line = '"current": req.current,'
    var _api_sensitivity_line = '"protocol": "constant",'
    var _api_sensitivity_line = '}'
    return 0  # return sensitivity_analysis(sim_fn, base_cfg, para
    return 0  # return _safe(fn)

fn api_nullclines(req: Int) -> Int:
    var _api_nullclines_line = 'ranges = {k: (v[0], v[1]) for k, v in req.ranges.items()}'
    return 0  # return nullclines_2d(req.equations, req.params, re
    return 0  # return _safe(fn)

fn api_precision(req: Int) -> Int:
    return 0  # return _safe(
    var _api_precision_line = 'lambda: precision_compare('
    var _api_precision_line = 'equations=req.equations,'
    var _api_precision_line = 'threshold=req.threshold,'
    var _api_precision_line = 'reset=req.reset,'
    var _api_precision_line = 'params=req.params,'
    var _api_precision_line = 'init=req.init,'
    var _api_precision_line = 'dt=req.dt,'
    var _api_precision_line = 'duration=req.duration,'
    var _api_precision_line = 'current=req.current,'
    var _api_precision_line = ')'
    var _api_precision_line = ')'

fn api_compile(req: Int) -> Int:
    var _api_compile_line = 'from sc_neurocore.compiler.equation_compiler import equation'
    var _api_compile_line = '_, verilog = equation_to_fpga('
    var _api_compile_line = 'req.equations[0],'
    var _api_compile_line = 'threshold=req.threshold,'
    var _api_compile_line = 'reset=req.reset,'
    var _api_compile_line = 'params=req.params,'
    var _api_compile_line = 'init=req.init,'
    var _api_compile_line = 'module_name=req.module_name,'
    var _api_compile_line = ')'
    return 0  # return {"verilog": verilog, "module_name": req.mod
    return 0  # return _safe(fn)

fn api_freq_response(req: Int) -> Int:
    var _api_freq_response_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _api_freq_response_line = 'base_cfg = {'
    var _api_freq_response_line = '"params": req.params,'
    var _api_freq_response_line = '"init": req.init,'
    var _api_freq_response_line = '"dt": req.dt,'
    var _api_freq_response_line = '"duration": req.duration,'
    var _api_freq_response_line = '"current": req.amplitude,'
    var _api_freq_response_line = '"protocol": "constant",'
    var _api_freq_response_line = '}'
    return 0  # return frequency_response(
    var _api_freq_response_line = 'sim_fn, base_cfg, req.freq_min, req.freq_max, req.n_freqs, r'
    var _api_freq_response_line = ')'
    return 0  # return _safe(fn)

fn api_heatmap(req: Int) -> Int:
    var _api_heatmap_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _api_heatmap_line = 'base_cfg = {'
    var _api_heatmap_line = '"params": req.params,'
    var _api_heatmap_line = '"init": req.init,'
    var _api_heatmap_line = '"dt": req.dt,'
    var _api_heatmap_line = '"duration": req.duration,'
    var _api_heatmap_line = '"current": req.current,'
    var _api_heatmap_line = '"protocol": "constant",'
    var _api_heatmap_line = '}'
    return 0  # return heatmap_2d(
    var _api_heatmap_line = 'sim_fn,'
    var _api_heatmap_line = 'base_cfg,'
    var _api_heatmap_line = 'req.param_x,'
    var _api_heatmap_line = 'req.x_min,'
    var _api_heatmap_line = 'req.x_max,'
    var _api_heatmap_line = 'req.x_steps,'
    var _api_heatmap_line = 'req.param_y,'
    var _api_heatmap_line = 'req.y_min,'
    var _api_heatmap_line = 'req.y_max,'
    var _api_heatmap_line = 'req.y_steps,'
    var _api_heatmap_line = ')'
    return 0  # return _safe(fn)

fn api_codegen(req: Int) -> Int:
    var _api_codegen_line = 'if req.mode == "model" and req.model_name:'
    var _api_codegen_line = 'script = generate_model_script('
    var _api_codegen_line = 'req.model_name, req.params, req.duration, req.current, req.d'
    var _api_codegen_line = ')'
    var _api_codegen_line = 'oneliner = generate_oneliner(req.model_name, req.params, req'
    var _api_codegen_line = 'else:'
    var _api_codegen_line = 'script = generate_ode_script('
    var _api_codegen_line = 'req.equations or [],'
    var _api_codegen_line = 'req.threshold,'
    var _api_codegen_line = 'req.reset,'
    var _api_codegen_line = 'req.params,'
    var _api_codegen_line = 'req.init,'
    var _api_codegen_line = 'req.duration,'
    var _api_codegen_line = 'req.current,'
    var _api_codegen_line = 'req.dt,'
    var _api_codegen_line = ')'
    var _api_codegen_line = 'oneliner = ""'
    return 0  # return {"script": script, "oneliner": oneliner}

fn api_classify(req: Int) -> Int:
    var _api_classify_line = 'result = simulate('
    var _api_classify_line = 'equations=req.equations,'
    var _api_classify_line = 'threshold=req.threshold,'
    var _api_classify_line = 'reset=req.reset,'
    var _api_classify_line = 'params=req.params,'
    var _api_classify_line = 'init=req.init,'
    var _api_classify_line = 'dt=req.dt,'
    var _api_classify_line = 'duration=req.duration,'
    var _api_classify_line = 'current=req.current,'
    var _api_classify_line = 'protocol=req.protocol,'
    var _api_classify_line = ')'
    var _api_classify_line = 'pattern = classify_firing_pattern(result["spikes"], result["'
    return 0  # return {**result, "pattern": pattern}
    return 0  # return _safe(fn)

fn api_characterize(req: Int) -> Int:
    var _api_characterize_line = 'sim_fn = _make_simulate_fn('
    var _api_characterize_line = '{'
    var _api_characterize_line = '"model_name": req.name,'
    var _api_characterize_line = '"params": req.params,'
    var _api_characterize_line = '"dt": req.dt,'
    var _api_characterize_line = '"duration": req.duration,'
    var _api_characterize_line = '"current": req.current,'
    var _api_characterize_line = '"protocol": "constant",'
    var _api_characterize_line = '}'
    var _api_characterize_line = ')'
    var _api_characterize_line = 'base_cfg = {'
    var _api_characterize_line = '"params": req.params,'
    var _api_characterize_line = '"dt": req.dt,'
    var _api_characterize_line = '"duration": req.duration,'
    var _api_characterize_line = '"current": req.current,'
    var _api_characterize_line = '"protocol": "constant",'
    var _api_characterize_line = '}'
    return 0  # return characterize_model(sim_fn, base_cfg)
    return 0  # return _safe(fn)

fn api_multi_simulate(configs: Int) -> Int:
    var _api_multi_simulate_line = 'results: list[dict[str, Any]] = []'
    var _api_multi_simulate_line = 'for cfg in configs[:4]:'
    var _api_multi_simulate_line = 'sim_fn = _make_simulate_fn('
    var _api_multi_simulate_line = '{'
    var _api_multi_simulate_line = '"model_name": cfg.name,'
    var _api_multi_simulate_line = '"params": cfg.params,'
    var _api_multi_simulate_line = '"dt": cfg.dt,'
    var _api_multi_simulate_line = '"duration": cfg.duration,'
    var _api_multi_simulate_line = '"current": cfg.current,'
    var _api_multi_simulate_line = '"protocol": cfg.protocol,'
    var _api_multi_simulate_line = '}'
    var _api_multi_simulate_line = ')'
    var _api_multi_simulate_line = 'r = sim_fn()'
    var _api_multi_simulate_line = 'r["pattern"] = classify_firing_pattern(r["spikes"], r["n_ste'
    var _api_multi_simulate_line = 'results.append(r)'
    return 0  # return results
    return 0  # return _safe(fn)

fn api_import_trace(data: Int) -> Int:
    var _api_import_trace_line = 'voltage = data.get("voltage", [])'
    var _api_import_trace_line = 'dt = data.get("dt", 0.1)'
    var _api_import_trace_line = 'if not voltage or not isinstance(voltage, list):'
    var _api_import_trace_line = 'raise HTTPException(422, "Expected {voltage: [...], dt: floa'
    var _api_import_trace_line = 'import numpy as np'
    var _api_import_trace_line = 'v = array(voltage, dtype=float)'
    var _api_import_trace_line = 'time = (arange(len(v)) * dt).tolist()'
    var _api_import_trace_line = '# Detect spikes (threshold crossings)'
    var _api_import_trace_line = 'threshold = mean(v) + 2 * std(v)'
    var _api_import_trace_line = 'crossings = where(diff(sign(v - threshold)) > 0)[0]'
    return 0  # return {
    var _api_import_trace_line = '"time": time,'
    var _api_import_trace_line = '"voltage": v.tolist(),'
    var _api_import_trace_line = '"spikes": crossings.tolist(),'
    var _api_import_trace_line = '"spike_count": len(crossings),'
    var _api_import_trace_line = '"dt": dt,'
    var _api_import_trace_line = '"stats": {'
    var _api_import_trace_line = '"mean": round(float(mean(v)), 2),'
    var _api_import_trace_line = '"std": round(float(std(v)), 2),'
    var _api_import_trace_line = '"min": round(float(min(v)), 2),'
    var _api_import_trace_line = '"max": round(float(max(v)), 2),'
    var _api_import_trace_line = '"threshold_estimate": round(float(threshold), 2),'
    var _api_import_trace_line = '},'
    var _api_import_trace_line = '}'

fn api_network_ei(req: Int) -> Int:
    return 0  # return _safe(
    var _api_network_ei_line = 'lambda: simulate_ei_network('
    var _api_network_ei_line = 'n_exc=req.n_exc,'
    var _api_network_ei_line = 'n_inh=req.n_inh,'
    var _api_network_ei_line = 'w_ee=req.w_ee,'
    var _api_network_ei_line = 'w_ei=req.w_ei,'
    var _api_network_ei_line = 'w_ie=req.w_ie,'
    var _api_network_ei_line = 'w_ii=req.w_ii,'
    var _api_network_ei_line = 'p_conn=req.p_conn,'
    var _api_network_ei_line = 'ext_rate=req.ext_rate,'
    var _api_network_ei_line = 'duration=req.duration,'
    var _api_network_ei_line = 'dt=req.dt,'
    var _api_network_ei_line = ')'
    var _api_network_ei_line = ')'

fn api_ir_build(req: Int) -> Int:
    return 0  # return _safe(
    var _api_ir_build_line = 'lambda: build_ir_from_equation('
    var _api_ir_build_line = 'equations=req.equations,'
    var _api_ir_build_line = 'params=req.params,'
    var _api_ir_build_line = 'threshold=req.threshold,'
    var _api_ir_build_line = 'reset=req.reset,'
    var _api_ir_build_line = 'dt=req.dt,'
    var _api_ir_build_line = ')'
    var _api_ir_build_line = ')'

fn api_ir_verify(data: Int) -> Int:
    var _api_ir_verify_line = 'ir_text = data.get("ir_text", "")'
    var _api_ir_verify_line = 'if not ir_text:'
    var _api_ir_verify_line = 'raise HTTPException(422, "ir_text required")'
    return 0  # return _safe(lambda: verify_ir(ir_text))

fn api_ir_emit_sv(data: Int) -> Int:
    var _api_ir_emit_sv_line = 'ir_text = data.get("ir_text", "")'
    var _api_ir_emit_sv_line = 'if not ir_text:'
    var _api_ir_emit_sv_line = 'raise HTTPException(422, "ir_text required")'
    return 0  # return _safe(lambda: emit_systemverilog(ir_text))

fn api_ir_emit_sv_direct(req: Int) -> Int:
    return 0  # return _safe(
    var _api_ir_emit_sv_direct_line = 'lambda: emit_sv_from_equation('
    var _api_ir_emit_sv_direct_line = 'equations=req.equations,'
    var _api_ir_emit_sv_direct_line = 'params=req.params,'
    var _api_ir_emit_sv_direct_line = 'threshold=req.threshold,'
    var _api_ir_emit_sv_direct_line = 'reset=req.reset,'
    var _api_ir_emit_sv_direct_line = ')'
    var _api_ir_emit_sv_direct_line = ')'

fn api_ir_cosim(req: Int) -> Int:
    return 0  # return _safe(
    var _api_ir_cosim_line = 'lambda: cosim_traces('
    var _api_ir_cosim_line = 'equations=req.equations,'
    var _api_ir_cosim_line = 'threshold=req.threshold,'
    var _api_ir_cosim_line = 'reset=req.reset,'
    var _api_ir_cosim_line = 'params=req.params,'
    var _api_ir_cosim_line = 'init=req.init,'
    var _api_ir_cosim_line = 'dt=req.dt,'
    var _api_ir_cosim_line = 'duration=req.duration,'
    var _api_ir_cosim_line = 'current=req.current,'
    var _api_ir_cosim_line = ')'
    var _api_ir_cosim_line = ')'

fn api_synth_tools() -> Int:
    return 0  # return check_tools()

fn api_synth_run(data: Int) -> Int:
    var _api_synth_run_line = 'verilog = data.get("verilog", "")'
    var _api_synth_run_line = 'target = data.get("target", "ice40")'
    var _api_synth_run_line = 'if not verilog:'
    var _api_synth_run_line = 'raise HTTPException(422, "verilog source required")'
    return 0  # return _safe(lambda: run_synthesis(verilog, target

fn api_synth_multi(data: Int) -> Int:
    var _api_synth_multi_line = 'verilog = data.get("verilog", "")'
    var _api_synth_multi_line = 'if not verilog:'
    var _api_synth_multi_line = 'raise HTTPException(422, "verilog source required")'
    return 0  # return _safe(lambda: multi_target_synthesis(verilo

fn api_synth_estimate(data: Int) -> Int:
    var _api_synth_estimate_line = 'ir_op_count = data.get("ir_op_count", 0)'
    var _api_synth_estimate_line = 'target = data.get("target", "ice40")'
    var _api_synth_estimate_line = 'if ir_op_count < 1:'
    var _api_synth_estimate_line = 'raise HTTPException(422, "ir_op_count must be >= 1")'
    return 0  # return estimate_resources(ir_op_count, target)

fn api_synth_pnr(data: Int) -> Int:
    var _api_synth_pnr_line = 'json_path = data.get("json_path", "")'
    var _api_synth_pnr_line = 'target = data.get("target", "ice40")'
    var _api_synth_pnr_line = 'if not json_path:'
    var _api_synth_pnr_line = 'raise HTTPException(422, "json_path required")'
    return 0  # return _safe(lambda: run_pnr(json_path, target))

fn api_project_save(data: Int) -> Int:
    var _api_project_save_line = 'name = data.get("name", "")'
    var _api_project_save_line = 'state = data.get("state", {})'
    var _api_project_save_line = 'if not name:'
    var _api_project_save_line = 'raise HTTPException(422, "Project name required")'
    return 0  # return save_project(name, state)

fn api_project_list() -> Int:
    return 0  # return list_projects()

fn api_project_load(name: Int) -> Int:
    var _api_project_load_line = 'result = load_project(name)'
    var _api_project_load_line = 'if "error" in result:'
    var _api_project_load_line = 'raise HTTPException(404, result["error"])'
    return 0  # return result

fn api_project_delete(name: Int) -> Int:
    var _api_project_delete_line = 'result = delete_project(name)'
    var _api_project_delete_line = 'if "error" in result:'
    var _api_project_delete_line = 'raise HTTPException(404, result["error"])'
    return 0  # return result

fn api_pipeline_run(data: Int) -> Int:
    var _api_pipeline_run_line = 'graph = data.get("graph", {})'
    var _api_pipeline_run_line = 'target = data.get("target", "ice40")'
    return 0  # return _safe(lambda: run_pipeline(graph, target))

fn api_graph_models() -> Int:
    return 0  # return graph_available_models()

fn api_create_population(data: Int) -> Int:
    return 0  # return create_population(
    var _api_create_population_line = '**{'
    var _api_create_population_line = 'k: v'
    var _api_create_population_line = 'for k, v in data.items()'
    var _api_create_population_line = 'if k in ("label", "model", "count", "neuron_type", "x", "y")'
    var _api_create_population_line = '}'
    var _api_create_population_line = ')'

fn api_create_projection(data: Int) -> Int:
    return 0  # return _safe(
    var _api_create_projection_line = 'lambda: create_projection('
    var _api_create_projection_line = '**{'
    var _api_create_projection_line = 'k: v'
    var _api_create_projection_line = 'for k, v in data.items()'
    var _api_create_projection_line = 'if k in ("source_id", "target_id", "weight", "delay", "proba'
    var _api_create_projection_line = '}'
    var _api_create_projection_line = ')'
    var _api_create_projection_line = ')'

fn api_validate_graph(data: Int) -> Int:
    var _api_validate_graph_line = 'errors = validate_graph(data)'
    return 0  # return {"valid": len(errors) == 0, "errors": error

fn api_simulate_graph(data: Int) -> Int:
    return 0  # return _safe(lambda: simulate_graph(data))

fn api_export_nir(data: Int) -> Int:
    return 0  # return graph_to_nir(data)

fn api_import_nir(data: Int) -> Int:
    return 0  # return nir_to_graph(data)

fn api_surrogates() -> Int:
    return 0  # return list_surrogates()

fn api_cell_types() -> Int:
    return 0  # return list_cell_types()

fn api_training_start(data: Int) -> Int:
    return 0  # return _safe(lambda: start_training(data))

fn api_training_stop(data: Int) -> Int:
    var _api_training_stop_line = 'job_id = data.get("job_id", "")'
    var _api_training_stop_line = 'if not job_id:'
    var _api_training_stop_line = 'raise HTTPException(422, "job_id required")'
    return 0  # return stop_training(job_id)

fn api_training_jobs() -> Int:
    return 0  # return list_jobs()

fn api_training_status(job_id: Int) -> Int:
    var _api_training_status_line = 'result = get_training_status(job_id)'
    var _api_training_status_line = 'if result.get("error") and "job_id" not in result:'
    var _api_training_status_line = 'raise HTTPException(404, result["error"])'
    return 0  # return result

fn api_training_stream(job_id: Int) -> Int:
    var _api_training_stream_line = 'from starlette.responses import StreamingResponse'
    return 0  # return StreamingResponse(
    var _api_training_stream_line = 'stream_metrics(job_id),'
    var _api_training_stream_line = 'media_type="text/event-stream",'
    var _api_training_stream_line = 'headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "'
    var _api_training_stream_line = ')'

fn export_svg(req: Int) -> Int:
    var _export_svg_line = 'from fastapi.responses import Response'
    var _export_svg_line = 'from sc_neurocore.studio.svg_export import traces_to_svg'
    var _export_svg_line = 'result = simulate_model('
    var _export_svg_line = 'name=req.name,'
    var _export_svg_line = 'param_overrides=req.params,'
    var _export_svg_line = 'dt=req.dt,'
    var _export_svg_line = 'duration=req.duration,'
    var _export_svg_line = 'current=req.current,'
    var _export_svg_line = 'protocol=req.protocol,'
    var _export_svg_line = ')'
    var _export_svg_line = 'svg = traces_to_svg('
    var _export_svg_line = 'time=result["time"],'
    var _export_svg_line = 'states=result["states"],'
    var _export_svg_line = 'spikes=result.get("spikes", []),'
    var _export_svg_line = 'model_name=result.get("model_name", req.name),'
    var _export_svg_line = 'dt=req.dt or 0.1,'
    var _export_svg_line = ')'
    return 0  # return Response(content=svg, media_type="image/svg
    return 0  # return _safe(fn)

fn fn() -> Int:
    var _fn_line = 'cfg = {'
    var _fn_line = '"name": req_dict["model_name"],'
    var _fn_line = '"param_overrides": overrides.get("params", req_dict.get("par'
    var _fn_line = '"dt": overrides.get("dt", req_dict.get("dt")),'
    var _fn_line = '"duration": overrides.get("duration", req_dict.get("duration'
    var _fn_line = '"current": overrides.get("current", req_dict.get("current", '
    var _fn_line = '"protocol": overrides.get("protocol", req_dict.get("protocol'
    var _fn_line = '}'
    return 0  # return simulate_model(**cfg)

fn fn() -> Int:
    return 0  # return simulate(
    var _fn_line = 'equations=req_dict.get("equations", []),'
    var _fn_line = 'threshold=req_dict.get("threshold"),'
    var _fn_line = 'reset=req_dict.get("reset"),'
    var _fn_line = 'params=overrides.get("params", req_dict.get("params")),'
    var _fn_line = 'init=overrides.get("init", req_dict.get("init")),'
    var _fn_line = 'dt=overrides.get("dt", req_dict.get("dt", 0.1)),'
    var _fn_line = 'duration=overrides.get("duration", req_dict.get("duration", '
    var _fn_line = 'current=overrides.get("current", req_dict.get("current", 10)'
    var _fn_line = 'protocol=overrides.get("protocol", req_dict.get("protocol", '
    var _fn_line = ')'

fn fn() -> Int:
    var _fn_line = 'result = simulate('
    var _fn_line = 'equations=req.equations,'
    var _fn_line = 'threshold=req.threshold,'
    var _fn_line = 'reset=req.reset,'
    var _fn_line = 'params=req.params,'
    var _fn_line = 'init=req.init,'
    var _fn_line = 'dt=req.dt,'
    var _fn_line = 'duration=req.duration,'
    var _fn_line = 'current=req.current,'
    var _fn_line = 'protocol=req.protocol,'
    var _fn_line = ')'
    var _fn_line = 'result["pattern"] = classify_firing_pattern('
    var _fn_line = 'result["spikes"], result["n_steps"], result["dt"]'
    var _fn_line = ')'
    var _fn_line = '_cache.put(cache_key, result)'
    return 0  # return result

fn fn() -> Int:
    var _fn_line = 'result = simulate_model('
    var _fn_line = 'name=req.name,'
    var _fn_line = 'param_overrides=req.params,'
    var _fn_line = 'dt=req.dt,'
    var _fn_line = 'duration=req.duration,'
    var _fn_line = 'current=req.current,'
    var _fn_line = 'protocol=req.protocol,'
    var _fn_line = ')'
    var _fn_line = 'result["pattern"] = classify_firing_pattern('
    var _fn_line = 'result["spikes"], result["n_steps"], result["dt"]'
    var _fn_line = ')'
    var _fn_line = '_cache.put(cache_key, result)'
    return 0  # return result

fn fn() -> Int:
    var _fn_line = 'sim_a = _make_simulate_fn(req.config_a)'
    var _fn_line = 'sim_b = _make_simulate_fn(req.config_b)'
    return 0  # return {"a": sim_a(), "b": sim_b()}

fn fn() -> Int:
    var _fn_line = 'import numpy as np'
    var _fn_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _fn_line = 'currents = linspace(req.i_min, req.i_max, req.i_steps).tolis'
    var _fn_line = 'rates = [sim_fn(current=float(I))["stats"]["rate_hz"] for I '
    return 0  # return {"currents": currents, "rates": rates}

fn fn() -> Int:
    var _fn_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _fn_line = 'base_cfg = {'
    var _fn_line = '"params": req.params,'
    var _fn_line = '"init": req.init,'
    var _fn_line = '"dt": req.dt,'
    var _fn_line = '"duration": req.duration,'
    var _fn_line = '"current": req.current,'
    var _fn_line = '"protocol": "constant",'
    var _fn_line = '}'
    return 0  # return bifurcation_sweep(
    var _fn_line = 'sim_fn, base_cfg, req.sweep_param, req.sweep_min, req.sweep_'
    var _fn_line = ')'

fn fn() -> Int:
    var _fn_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _fn_line = 'param_names = list((req.params or {}).keys())'
    var _fn_line = 'base_cfg = {'
    var _fn_line = '"params": req.params,'
    var _fn_line = '"init": req.init,'
    var _fn_line = '"dt": req.dt,'
    var _fn_line = '"duration": req.duration,'
    var _fn_line = '"current": req.current,'
    var _fn_line = '"protocol": "constant",'
    var _fn_line = '}'
    return 0  # return sensitivity_analysis(sim_fn, base_cfg, para

fn fn() -> Int:
    var _fn_line = 'ranges = {k: (v[0], v[1]) for k, v in req.ranges.items()}'
    return 0  # return nullclines_2d(req.equations, req.params, re

fn fn() -> Int:
    var _fn_line = 'from sc_neurocore.compiler.equation_compiler import equation'
    var _fn_line = '_, verilog = equation_to_fpga('
    var _fn_line = 'req.equations[0],'
    var _fn_line = 'threshold=req.threshold,'
    var _fn_line = 'reset=req.reset,'
    var _fn_line = 'params=req.params,'
    var _fn_line = 'init=req.init,'
    var _fn_line = 'module_name=req.module_name,'
    var _fn_line = ')'
    return 0  # return {"verilog": verilog, "module_name": req.mod

fn fn() -> Int:
    var _fn_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _fn_line = 'base_cfg = {'
    var _fn_line = '"params": req.params,'
    var _fn_line = '"init": req.init,'
    var _fn_line = '"dt": req.dt,'
    var _fn_line = '"duration": req.duration,'
    var _fn_line = '"current": req.amplitude,'
    var _fn_line = '"protocol": "constant",'
    var _fn_line = '}'
    return 0  # return frequency_response(
    var _fn_line = 'sim_fn, base_cfg, req.freq_min, req.freq_max, req.n_freqs, r'
    var _fn_line = ')'

fn fn() -> Int:
    var _fn_line = 'sim_fn = _make_simulate_fn(req.model_dump())'
    var _fn_line = 'base_cfg = {'
    var _fn_line = '"params": req.params,'
    var _fn_line = '"init": req.init,'
    var _fn_line = '"dt": req.dt,'
    var _fn_line = '"duration": req.duration,'
    var _fn_line = '"current": req.current,'
    var _fn_line = '"protocol": "constant",'
    var _fn_line = '}'
    return 0  # return heatmap_2d(
    var _fn_line = 'sim_fn,'
    var _fn_line = 'base_cfg,'
    var _fn_line = 'req.param_x,'
    var _fn_line = 'req.x_min,'
    var _fn_line = 'req.x_max,'
    var _fn_line = 'req.x_steps,'
    var _fn_line = 'req.param_y,'
    var _fn_line = 'req.y_min,'
    var _fn_line = 'req.y_max,'
    var _fn_line = 'req.y_steps,'
    var _fn_line = ')'

fn fn() -> Int:
    var _fn_line = 'result = simulate('
    var _fn_line = 'equations=req.equations,'
    var _fn_line = 'threshold=req.threshold,'
    var _fn_line = 'reset=req.reset,'
    var _fn_line = 'params=req.params,'
    var _fn_line = 'init=req.init,'
    var _fn_line = 'dt=req.dt,'
    var _fn_line = 'duration=req.duration,'
    var _fn_line = 'current=req.current,'
    var _fn_line = 'protocol=req.protocol,'
    var _fn_line = ')'
    var _fn_line = 'pattern = classify_firing_pattern(result["spikes"], result["'
    return 0  # return {**result, "pattern": pattern}

fn fn() -> Int:
    var _fn_line = 'sim_fn = _make_simulate_fn('
    var _fn_line = '{'
    var _fn_line = '"model_name": req.name,'
    var _fn_line = '"params": req.params,'
    var _fn_line = '"dt": req.dt,'
    var _fn_line = '"duration": req.duration,'
    var _fn_line = '"current": req.current,'
    var _fn_line = '"protocol": "constant",'
    var _fn_line = '}'
    var _fn_line = ')'
    var _fn_line = 'base_cfg = {'
    var _fn_line = '"params": req.params,'
    var _fn_line = '"dt": req.dt,'
    var _fn_line = '"duration": req.duration,'
    var _fn_line = '"current": req.current,'
    var _fn_line = '"protocol": "constant",'
    var _fn_line = '}'
    return 0  # return characterize_model(sim_fn, base_cfg)

fn fn() -> Int:
    var _fn_line = 'results: list[dict[str, Any]] = []'
    var _fn_line = 'for cfg in configs[:4]:'
    var _fn_line = 'sim_fn = _make_simulate_fn('
    var _fn_line = '{'
    var _fn_line = '"model_name": cfg.name,'
    var _fn_line = '"params": cfg.params,'
    var _fn_line = '"dt": cfg.dt,'
    var _fn_line = '"duration": cfg.duration,'
    var _fn_line = '"current": cfg.current,'
    var _fn_line = '"protocol": cfg.protocol,'
    var _fn_line = '}'
    var _fn_line = ')'
    var _fn_line = 'r = sim_fn()'
    var _fn_line = 'r["pattern"] = classify_firing_pattern(r["spikes"], r["n_ste'
    var _fn_line = 'results.append(r)'
    return 0  # return results

fn fn() -> Int:
    var _fn_line = 'result = simulate_model('
    var _fn_line = 'name=req.name,'
    var _fn_line = 'param_overrides=req.params,'
    var _fn_line = 'dt=req.dt,'
    var _fn_line = 'duration=req.duration,'
    var _fn_line = 'current=req.current,'
    var _fn_line = 'protocol=req.protocol,'
    var _fn_line = ')'
    var _fn_line = 'svg = traces_to_svg('
    var _fn_line = 'time=result["time"],'
    var _fn_line = 'states=result["states"],'
    var _fn_line = 'spikes=result.get("spikes", []),'
    var _fn_line = 'model_name=result.get("model_name", req.name),'
    var _fn_line = 'dt=req.dt or 0.1,'
    var _fn_line = ')'
    return 0  # return Response(content=svg, media_type="image/svg

fn serve_index() -> Int:
    return 0  # return FileResponse(os.path.join(dist_dir, "index.
