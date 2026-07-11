# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio simulation and analysis routes

"""Adapt simulation, analysis, characterisation, and E-I network capabilities."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.analysis import (
    bifurcation_sweep,
    frequency_response,
    heatmap_2d,
    nullclines_2d,
    precision_compare,
    sensitivity_analysis,
)
from sc_neurocore.studio.analysis_manifest import attach_analysis_result_manifest
from sc_neurocore.studio.api.analysis_guards import (
    _attach_analysis_metadata,
    _config_duration_dt,
    _guard_analysis_request,
    _guard_multi_config_analysis_request,
    _guard_nullcline_grid_request,
    _make_simulate_fn,
)
from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    BifurcationRequest,
    CodegenRequest,
    CompareRequest,
    FICurveRequest,
    FreqResponseRequest,
    HeatmapRequest,
    ModelSimulateRequest,
    NetworkRequest,
    NullclineRequest,
    PrecisionRequest,
    SensitivityRequest,
    SimulateRequest,
)
from sc_neurocore.studio.characterize import characterize_model
from sc_neurocore.studio.codegen import (
    classify_firing_pattern,
    generate_model_script,
    generate_ode_script,
    generate_oneliner,
)
from sc_neurocore.studio.models import simulate_model
from sc_neurocore.studio.network import simulate_ei_network
from sc_neurocore.studio.simulation import simulate
from sc_neurocore.studio.simulation_manifest import build_simulation_run_manifest


class _SimCache:
    """LRU cache for simulation results keyed by JSON hash."""

    def __init__(self, maxsize: int = 64) -> None:
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _key(self, data: dict[str, Any]) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    def get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        k = self._key(params)
        if k in self._cache:
            self.hits += 1
            self._cache.move_to_end(k)
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, params: dict[str, Any], result: dict[str, Any]) -> None:
        k = self._key(params)
        self._cache[k] = result
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


_cache = _SimCache()


def build_simulation_router(context: StudioApiContext) -> APIRouter:
    """Build the simulation and analysis router over shared Studio runtime state."""
    router = APIRouter()
    analysis_budget = context.analysis_budget

    @router.post("/api/simulate")
    def api_simulate(req: SimulateRequest) -> Any:
        cache_key = {"_type": "ode", **req.model_dump()}
        cached = _cache.get(cache_key)
        if cached:
            return cached
        _guard_analysis_request(
            analysis_budget, simulation_count=1, duration=req.duration, dt=req.dt
        )

        def fn() -> dict[str, Any]:
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
            result["run_metadata"] = build_simulation_run_manifest(
                source="ode",
                request_payload=req.model_dump(),
                result_payload=result,
            ).to_public_dict()
            _cache.put(cache_key, result)
            return result

        return _safe(fn)

    @router.post("/api/models/simulate")
    def api_model_simulate(req: ModelSimulateRequest) -> Any:
        cache_key = {"_type": "model", **req.model_dump()}
        cached = _cache.get(cache_key)
        if cached:
            return cached
        _guard_analysis_request(
            analysis_budget, simulation_count=1, duration=req.duration, dt=req.dt
        )

        def fn() -> dict[str, Any]:
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
            result["run_metadata"] = build_simulation_run_manifest(
                source="model",
                request_payload=req.model_dump(),
                result_payload=result,
            ).to_public_dict()
            _cache.put(cache_key, result)
            return result

        return _safe(fn)

    @router.get("/api/cache/stats")
    def api_cache_stats() -> dict[str, int]:
        return {"hits": _cache.hits, "misses": _cache.misses, "size": len(_cache._cache)}

    @router.post("/api/compare")
    def api_compare(req: CompareRequest) -> Any:
        _guard_multi_config_analysis_request(
            analysis_budget,
            [_config_duration_dt(req.config_a), _config_duration_dt(req.config_b)],
        )

        def fn() -> dict[str, Any]:
            sim_a = _make_simulate_fn(req.config_a)
            sim_b = _make_simulate_fn(req.config_b)
            payload = {"a": sim_a(), "b": sim_b()}
            return attach_analysis_result_manifest(
                analysis_type="compare",
                source="mixed",
                request_payload=req.model_dump(),
                result_payload=payload,
            )

        return _safe(fn)

    @router.post("/api/fi-curve")
    def api_fi_curve(req: FICurveRequest) -> Any:
        _guard_analysis_request(
            analysis_budget,
            simulation_count=req.i_steps,
            duration=req.duration,
            dt=req.dt,
        )

        def fn() -> dict[str, Any]:
            import numpy as np

            sim_fn = _make_simulate_fn(req.model_dump())
            currents = np.linspace(req.i_min, req.i_max, req.i_steps).tolist()
            rates = [sim_fn(current=float(I))["stats"]["rate_hz"] for I in currents]
            payload = {"currents": currents, "rates": rates}
            return _attach_analysis_metadata("fi_curve", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/bifurcation")
    def api_bifurcation(req: BifurcationRequest) -> Any:
        _guard_analysis_request(
            analysis_budget,
            simulation_count=req.sweep_steps,
            duration=req.duration,
            dt=req.dt,
        )

        def fn() -> dict[str, Any]:
            sim_fn = _make_simulate_fn(req.model_dump())
            base_cfg = {
                "params": req.params,
                "init": req.init,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.current,
                "protocol": "sine",
            }
            payload = bifurcation_sweep(
                sim_fn, base_cfg, req.sweep_param, req.sweep_min, req.sweep_max, req.sweep_steps
            )
            return _attach_analysis_metadata("bifurcation", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/sensitivity")
    def api_sensitivity(req: SensitivityRequest) -> Any:
        _guard_analysis_request(
            analysis_budget,
            simulation_count=1 + 2 * len(req.params or {}),
            duration=req.duration,
            dt=req.dt,
        )

        def fn() -> dict[str, Any]:
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
            payload = sensitivity_analysis(sim_fn, base_cfg, param_names)
            return _attach_analysis_metadata("sensitivity", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/nullclines")
    def api_nullclines(req: NullclineRequest) -> Any:
        _guard_nullcline_grid_request(
            analysis_budget,
            grid_size=req.grid_size,
            equation_count=len(req.equations),
        )

        def fn() -> dict[str, Any]:
            ranges = {k: (v[0], v[1]) for k, v in req.ranges.items()}
            payload = nullclines_2d(req.equations, req.params, req.var_names, ranges, req.grid_size)
            return _attach_analysis_metadata("nullclines", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/precision")
    def api_precision(req: PrecisionRequest) -> Any:
        _guard_analysis_request(
            analysis_budget, simulation_count=2, duration=req.duration, dt=req.dt
        )

        def fn() -> dict[str, Any]:
            payload = precision_compare(
                equations=req.equations,
                threshold=req.threshold,
                reset=req.reset,
                params=req.params,
                init=req.init,
                dt=req.dt,
                duration=req.duration,
                current=req.current,
            )
            return _attach_analysis_metadata("precision", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/freq-response")
    def api_freq_response(req: FreqResponseRequest) -> Any:
        _guard_analysis_request(
            analysis_budget,
            simulation_count=req.n_freqs,
            duration=req.duration,
            dt=req.dt,
        )

        def fn() -> dict[str, Any]:
            sim_fn = _make_simulate_fn(req.model_dump())
            base_cfg = {
                "params": req.params,
                "init": req.init,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.amplitude,
                "protocol": "constant",
            }
            payload = frequency_response(
                sim_fn, base_cfg, req.freq_min, req.freq_max, req.n_freqs, req.amplitude
            )
            return _attach_analysis_metadata("frequency_response", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/heatmap")
    def api_heatmap(req: HeatmapRequest) -> Any:
        _guard_analysis_request(
            analysis_budget,
            simulation_count=req.x_steps * req.y_steps,
            duration=req.duration,
            dt=req.dt,
        )

        def fn() -> dict[str, Any]:
            sim_fn = _make_simulate_fn(req.model_dump())
            base_cfg = {
                "params": req.params,
                "init": req.init,
                "dt": req.dt,
                "duration": req.duration,
                "current": req.current,
                "protocol": "constant",
            }
            payload = heatmap_2d(
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
            return _attach_analysis_metadata("heatmap", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/codegen")
    def api_codegen(req: CodegenRequest) -> Any:
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

    @router.post("/api/classify")
    def api_classify(req: SimulateRequest) -> Any:
        _guard_analysis_request(
            analysis_budget, simulation_count=1, duration=req.duration, dt=req.dt
        )

        def fn() -> dict[str, Any]:
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

    @router.post("/api/characterize")
    def api_characterize(req: ModelSimulateRequest) -> Any:
        # characterize_model drives: 1 trace + 20-point f-I curve + 2 sims per
        # parameter for the top-15 quick-sensitivity sweep.
        _guard_analysis_request(
            analysis_budget,
            simulation_count=1 + 20 + 2 * min(15, len(req.params or {})),
            duration=req.duration,
            dt=req.dt,
        )

        def fn() -> dict[str, Any]:
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
            result = characterize_model(sim_fn, base_cfg)
            # ModelSimulateRequest is always model-driven; its dump uses ``name``
            # (not ``model_name``), so source is set explicitly rather than inferred.
            return attach_analysis_result_manifest(
                analysis_type="characterize",
                source="model",
                request_payload=req.model_dump(),
                result_payload=result,
            )

        return _safe(fn)

    @router.post("/api/multi-simulate")
    def api_multi_simulate(configs: list[ModelSimulateRequest]) -> Any:
        if configs:
            _guard_multi_config_analysis_request(
                analysis_budget,
                [(cfg.duration, cfg.dt) for cfg in configs[:4]],
            )

        def fn() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
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
                r["run_metadata"] = build_simulation_run_manifest(
                    source="model",
                    request_payload=cfg.model_dump(),
                    result_payload=r,
                ).to_public_dict()
                results.append(r)
            return results

        return _safe(fn)

    @router.post("/api/import-trace")
    def api_import_trace(data: dict[str, Any]) -> Any:
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

    @router.post("/api/network/ei")
    def api_network_ei(req: NetworkRequest) -> Any:
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

    return router
