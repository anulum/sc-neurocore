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
from typing import Any, Literal

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.analysis import (
    bifurcation_sweep,
    fi_curve_sweep,
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
from sc_neurocore.studio.api.analysis_jobs import (
    AnalysisJobValidationError,
    submit_analysis_job,
)
from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.evidence_receipt import attach_evidence_receipt
from sc_neurocore.studio.evidence_scope import simulation_scope
from sc_neurocore.studio.bit_true_execution import NativeToolUnavailable
from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    MODEL_RUN_ERROR_RESPONSES,
    PRECISION_ERROR_RESPONSES,
    AnalysisJobRequest,
    BifurcationRequest,
    ExperimentExportRequest,
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
    generate_experiment_script,
    generate_oneliner,
    generate_replay_script,
)
from sc_neurocore.studio.firing_pattern import classify_firing_pattern
from sc_neurocore.studio.experiment_spec import (
    ExperimentRejected,
    ExperimentSpec,
    resolve_experiment,
    run_experiment,
)
from sc_neurocore.studio.network import simulate_ei_network
from sc_neurocore.studio.replay_pack import build_replay_pack, pinned_request
from sc_neurocore.studio.simulation import simulate
from sc_neurocore.studio.simulation_manifest import build_simulation_run_manifest


CACHE_RAW_ELEMENT_LIMIT = 200_000


def _raw_element_count(result: dict[str, Any]) -> int:
    """Return the raw element count a custody result declares (0 when absent)."""
    raw = result.get("raw")
    if isinstance(raw, dict):
        count = raw.get("element_count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return 0


class _SimCache:
    """LRU cache for simulation results keyed by the resolved experiment digest.

    The key is the ``experiment_sha256`` of the resolved
    :class:`~sc_neurocore.studio.experiment_spec.ExperimentSpec`, which binds
    the effective inputs (model revision or equation digest, numerical
    profile, effective dt and step count, typed initial state, protocol and
    drive digest, randomness contract, backend and runtime digests), so runs
    that differ in any of them cannot share an entry. A fresh stochastic
    trial is never stored. Results whose raw block exceeds
    :data:`CACHE_RAW_ELEMENT_LIMIT` elements are returned but not retained.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(data: dict[str, Any] | str) -> str:
        if isinstance(data, str):
            return data
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, params: dict[str, Any] | str) -> dict[str, Any] | None:
        k = self._key(params)
        if k in self._cache:
            self.hits += 1
            self._cache.move_to_end(k)
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, params: dict[str, Any] | str, result: dict[str, Any]) -> None:
        if _raw_element_count(result) > CACHE_RAW_ELEMENT_LIMIT:
            return
        k = self._key(params)
        self._cache[k] = result
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


_cache = _SimCache()


def _resolve_or_422(request: dict[str, Any]) -> ExperimentSpec:
    """Resolve the effective experiment or raise the structured 422."""
    try:
        return resolve_experiment(request)
    except ExperimentRejected as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None


def _resolve_or_422_model(request: dict[str, Any]) -> ExperimentSpec:
    """Resolve a catalogue-model experiment; contract errors become the public 422."""
    try:
        return resolve_experiment(request)
    except ExperimentRejected as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None
    except ModelInputError as exc:
        raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None


def _request_payload(req: Any) -> dict[str, Any]:
    """Return an export request as the experiment contract's request mapping.

    The ``mode`` discriminator selects the schema; it is not part of the
    experiment, so it never reaches the contract or a digest.
    """
    payload = dict(req.model_dump())
    payload.pop("mode", None)
    return payload


def _cached_replay(spec: ExperimentSpec) -> dict[str, Any] | None:
    """Return the cached result of a replayable experiment, marked as a cache hit.

    The replay carries its own receipt rather than the fresh run's: the two
    responses differ in their ``cache`` block, so a receipt copied across would
    no longer seal what it accompanies. Both name the same run through
    ``scope.experiment_sha256``.
    """
    if not spec.cacheable:
        return None
    cached = _cache.get(spec.experiment_sha256)
    if cached is None:
        return None
    replay = dict(cached)
    replay["cache"] = {"hit": True, "key": spec.experiment_sha256}
    return _sealed_simulation(replay)


def _run_and_record(
    spec: ExperimentSpec, *, source: Literal["ode", "model"], request_payload: dict[str, Any]
) -> dict[str, Any]:
    """Execute a resolved experiment, attach pattern and manifest, store a replay."""
    result = run_experiment(spec)
    result["pattern"] = classify_firing_pattern(result["spikes"], result["n_steps"], result["dt"])
    result["cache"] = {"hit": False, "key": spec.experiment_sha256}
    result["run_metadata"] = build_simulation_run_manifest(
        source=source,
        request_payload=request_payload,
        result_payload=result,
    ).to_public_dict()
    if spec.cacheable:
        _cache.put(spec.experiment_sha256, result)
    return _sealed_simulation(result)


def _sealed_simulation(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the produced-evidence receipt for one simulation response."""
    return attach_evidence_receipt(
        result,
        lane="simulation",
        status="completed",
        binding="produced",
        scope=simulation_scope(result),
    )


def build_simulation_router(context: StudioApiContext) -> APIRouter:
    """Build the simulation and analysis router over shared Studio runtime state."""
    router = APIRouter()
    analysis_budget = context.analysis_budget
    studio_job_manager = context.studio_job_manager

    @router.post("/api/analysis/jobs")
    def api_analysis_job(req: AnalysisJobRequest) -> dict[str, Any]:
        """Submit a heavy analysis run as an asynchronous Studio job.

        Use this when a synchronous analysis route returns
        ``execution_mode=job_required``. The job result carries the same
        analysis payload shape as the corresponding synchronous endpoint.
        """
        try:
            return submit_analysis_job(studio_job_manager, req)
        except AnalysisJobValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None

    @router.post("/api/simulate", responses=MODEL_RUN_ERROR_RESPONSES)
    def api_simulate(req: SimulateRequest) -> Any:
        """Simulate the equation playground under the effective experiment contract.

        The request resolves into one ``experiment`` (equation digest,
        numerical profile, exact step count, typed initial state, protocol
        with drive digest, randomness contract, backend and runtime digests)
        before anything runs; the result cache is keyed by that digest and a
        fresh stochastic trial is never cached. An oversized run is refused
        with ``execution_mode = job_required`` instead of being shortened.
        """
        spec = _resolve_or_422(req.model_dump())
        cached = _cached_replay(spec)
        if cached is not None:
            return cached
        _guard_analysis_request(
            analysis_budget, simulation_count=1, duration=spec.duration_ms, dt=spec.dt
        )
        return _safe(lambda: _run_and_record(spec, source="ode", request_payload=req.model_dump()))

    @router.post("/api/models/simulate", responses=MODEL_RUN_ERROR_RESPONSES)
    def api_model_simulate(req: ModelSimulateRequest) -> Any:
        """Simulate one catalogue model under the fail-closed run contract.

        The request resolves into one ``experiment`` (model revision and
        descriptor/schema digests, numerical profile, effective dt and exact
        step count, typed initial state, protocol with drive digest,
        randomness contract, backend selection, runtime digest) before
        anything runs; the cache is keyed by that digest. A rejected request
        (HTTP 422 ``invalid_model_input`` or ``experiment_rejected``) or a
        numerical failure (HTTP 422 ``model_simulation_failed``) is never
        cached and never returns a success payload; a successful run carries
        its ``effective_inputs`` receipt, the declared ``state_layout`` with
        its custody verdict, exact initial and final snapshots, the
        full-resolution ``raw`` block and the ``display`` projection. The run
        executes on the Python custody backend so every declared variable is
        observed.
        """
        spec = _resolve_or_422_model(req.model_dump())
        cached = _cached_replay(spec)
        if cached is not None:
            return cached
        _guard_analysis_request(
            analysis_budget,
            simulation_count=1,
            duration=spec.duration_ms,
            dt=spec.dt,
            model_name=req.name,
        )
        return _safe(
            lambda: _run_and_record(spec, source="model", request_payload=req.model_dump())
        )

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
            sim_fn = _make_simulate_fn(req.model_dump())
            payload = fi_curve_sweep(sim_fn, req.i_min, req.i_max, req.i_steps)
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
                sim_fn,
                base_cfg,
                req.sweep_param,
                req.sweep_min,
                req.sweep_max,
                req.sweep_steps,
                variable=req.variable,
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
            ranges = {k: (v[0], v[1]) for k, v in req.ranges.items() if len(v) == 2}
            payload = nullclines_2d(
                req.equations,
                req.params,
                req.var_names,
                ranges,
                req.grid_size,
                current=req.current,
                held=req.held,
            )
            return _attach_analysis_metadata("nullclines", req.model_dump(), payload)

        return _safe(fn)

    @router.post("/api/precision", responses=PRECISION_ERROR_RESPONSES)
    def api_precision(req: PrecisionRequest) -> Any:
        # Three runs: float64 reference, bit-true kernel, parameter-quantised float64.
        _guard_analysis_request(
            analysis_budget, simulation_count=3, duration=req.duration, dt=req.dt
        )

        def fn() -> dict[str, Any]:
            try:
                payload = precision_compare(
                    equations=req.equations,
                    threshold=req.threshold,
                    reset=req.reset,
                    params=req.params,
                    init=req.init,
                    dt=req.dt,
                    duration=req.duration,
                    current=req.current,
                    protocol=req.protocol,
                    frequency_hz=req.frequency_hz,
                    q_format=req.q_format,
                    overflow=req.overflow,
                    rounding=req.rounding,
                )
            except NativeToolUnavailable as exc:
                raise HTTPException(status_code=503, detail=exc.to_public_detail()) from None
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

    @router.post("/api/codegen", responses=MODEL_RUN_ERROR_RESPONSES)
    def api_codegen(req: ExperimentExportRequest) -> Any:
        """Export Python that runs this exact experiment somewhere else.

        The request is resolved through the effective experiment contract
        before any code is written, so the script inherits the run's timestep,
        drive protocol, initial state, parameter overrides and randomness
        instead of restating a guess about the model's constructor. A drawn
        stochastic seed is pinned, so the exported script replays the trial it
        was exported from. The script refuses to report a result if the
        installed package resolves a different experiment. A request the
        contract rejects is refused here too (HTTP 422), never exported as
        code that would fail or, worse, quietly run something else.
        """
        payload = _request_payload(req)
        spec = _resolve_or_422_model(payload)
        sealed = pinned_request(payload, spec)
        return {
            "script": generate_experiment_script(spec, sealed),
            "oneliner": generate_oneliner(spec, sealed),
            "replay_script": generate_replay_script(),
            "experiment_sha256": spec.experiment_sha256,
            "request": sealed,
        }

    @router.post("/api/export/replay-pack", responses=MODEL_RUN_ERROR_RESPONSES)
    def api_export_replay_pack(req: ExperimentExportRequest) -> Any:
        """Export a sealed pack another installation can run and compare.

        The experiment is resolved, its randomness pinned and the run executed
        once; the pack carries the re-resolvable request, the specification,
        the scientific identity digest, the complete expectation (every spike
        event, full scalar/vector samples bound to trace digests, initial/final
        state and drive digest) and the environment that sealed it. Missing raw
        evidence or excessive pack size returns a structured HTTP 422 refusal. Replay it with
        ``python -m sc_neurocore.studio.replay_pack <pack.json>``.
        """
        payload = _request_payload(req)
        spec = _resolve_or_422_model(payload)
        _guard_analysis_request(
            analysis_budget, simulation_count=1, duration=spec.duration_ms, dt=spec.dt
        )
        return _safe(lambda: build_replay_pack(payload))

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

        specs = [_resolve_or_422_model(cfg.model_dump()) for cfg in configs[:4]]

        def fn() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for cfg, spec in zip(configs[:4], specs, strict=True):
                cached = _cached_replay(spec)
                results.append(
                    cached
                    if cached is not None
                    else _run_and_record(spec, source="model", request_payload=cfg.model_dump())
                )
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
