# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FastAPI backend for Visual SNN Design Studio

from __future__ import annotations

import hashlib
import json
import logging
import time
import tempfile
from uuid import uuid4
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.responses import Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.routing import Match, Route
import numpy as np
from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated
from sc_neurocore.compiler import (
    auto_tune_synapse_precisions,
    assign_synapse_precisions,
    write_precision_formal_evidence_bundle,
)

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
from sc_neurocore.studio.platform import (
    AuditExportValue,
    AuditSinkError,
    InMemoryAuditSink,
    JsonlAuditSink,
    PolicyGateway,
    Principal,
    StudioIdentityAuthenticator,
    StudioIdentityResult,
    StudioJobArtifactUnavailable,
    StudioJobManager,
    StudioRuntimeSettings,
    build_default_studio_capability_registry,
    build_default_studio_route_policy_registry,
    build_default_studio_runtime_settings,
    build_studio_operator_status,
    load_studio_identity_store,
)
from sc_neurocore.studio.presets import (
    get_preset,
    get_preset_action,
    get_preset_actions,
    list_preset_action_catalog,
    list_presets,
)
from sc_neurocore.studio.simulation import simulate
from sc_neurocore.studio.templates import get_template, list_templates


logger = logging.getLogger(__name__)
DEFAULT_STUDIO_JOB_KINDS = frozenset({"compiler", "synthesis", "training"})


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


class AdaptivePrecisionAutoTuneRequest(BaseModel):
    layer_weights: list[list[list[float]] | list[float]]
    layer_names: list[str] | None = None
    target_error_percent: float = Field(default=0.1, gt=0.0, le=10.0)
    min_bits: int = Field(default=4, ge=1, le=32)
    max_bits: int = Field(default=16, ge=1, le=32)
    min_length: int = Field(default=32, ge=1, le=65536)
    max_length: int = Field(default=4096, ge=1, le=262144)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class AdaptivePrecisionFormalBundleRequest(BaseModel):
    layer_weights: list[list[list[float]] | list[float]]
    layer_names: list[str] | None = None
    target_error_percent: float = Field(default=0.1, gt=0.0, le=10.0)
    min_bits: int = Field(default=4, ge=1, le=32)
    max_bits: int = Field(default=16, ge=1, le=32)
    min_length: int = Field(default=32, ge=1, le=65536)
    max_length: int = Field(default=4096, ge=1, le=262144)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    module_name: str = Field(default="adaptive_precision_plan", min_length=1, max_length=128)


class PresetActionResolveRequest(BaseModel):
    overrides: dict[str, Any] = Field(default_factory=dict)


class PresetActionsExecuteAllRequest(BaseModel):
    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowRunRequest(BaseModel):
    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowVerifyRequest(BaseModel):
    action_order: list[str]
    template_fingerprints: dict[str, Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]]
    plan_fingerprint_sha256: Annotated[str | None, StringConstraints(pattern=r"^[0-9a-f]{64}$")] = (
        None
    )


class PresetDefaultFlowGuardedRunRequest(BaseModel):
    action_order: list[str]
    template_fingerprints: dict[str, Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]]
    plan_fingerprint_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowRunFromContractRequest(BaseModel):
    contract: dict[str, Any]
    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowAttestRequest(BaseModel):
    run_result: dict[str, Any]


class PresetDefaultFlowAttestationVerifyRequest(BaseModel):
    run_result: dict[str, Any]
    attestation: dict[str, Any]


class CompareRequest(BaseModel):
    config_a: dict[str, Any]
    config_b: dict[str, Any]


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


def _safe(fn: Callable[..., Any]) -> Any:
    try:
        return fn()
    except HTTPException:
        raise
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=422, detail="Invalid input") from None
    except Exception:
        logger.exception("Studio API internal error in %r", fn)
        raise HTTPException(status_code=500, detail="Internal error") from None


def _make_simulate_fn(req_dict: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """Build a simulate callable from request params (ODE or model)."""
    if req_dict.get("model_name"):

        def fn(**overrides: Any) -> dict[str, Any]:
            cfg = {
                "name": req_dict["model_name"],
                "param_overrides": overrides.get("params", req_dict.get("params")),
                "dt": overrides.get("dt", req_dict.get("dt")),
                "duration": overrides.get("duration", req_dict.get("duration", 200)),
                "current": overrides.get("current", req_dict.get("current", 10)),
                "protocol": overrides.get("protocol", req_dict.get("protocol", "constant")),
                "frequency_hz": overrides.get("frequency_hz", req_dict.get("frequency_hz", 10.0)),
            }
            return simulate_model(**cfg)

        return fn
    else:

        def fn(**overrides: Any) -> dict[str, Any]:
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
                frequency_hz=overrides.get("frequency_hz", req_dict.get("frequency_hz", 10.0)),
            )

        return fn


def _parse_layer_weight_arrays(
    layer_weights: list[list[list[float]] | list[float]],
) -> list[np.ndarray[Any, Any]]:
    arrays: list[np.ndarray[Any, Any]] = []
    for idx, layer in enumerate(layer_weights):
        array = np.asarray(layer, dtype=float)
        if array.ndim not in {1, 2}:
            raise ValueError(f"layer {idx} must be 1D or 2D")
        if array.size == 0:
            raise ValueError(f"layer {idx} must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"layer {idx} contains non-finite values")
        arrays.append(array)
    return arrays


def _resolve_action_payload(
    preset_id: str,
    action_id: str,
    action: dict[str, Any],
    payload_template: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be an object")
    unknown = set(overrides) - set(payload_template)
    if unknown:
        bad = ", ".join(sorted(unknown))
        raise ValueError(f"unknown override keys: {bad}")

    resolved = dict(payload_template)
    for key, value in overrides.items():
        template_value = payload_template[key]
        if template_value is not None and not isinstance(value, type(template_value)):
            raise ValueError(f"override type mismatch for key '{key}'")
        resolved[key] = value

    return {
        "preset_id": preset_id,
        "action_id": action_id,
        "method": action.get("method"),
        "endpoint": action.get("endpoint"),
        "payload": resolved,
    }


def _run_adaptive_precision_auto_tune_payload(payload: dict[str, Any]) -> dict[str, Any]:
    layer_arrays = _parse_layer_weight_arrays(payload["layer_weights"])
    return auto_tune_synapse_precisions(
        layer_arrays,
        layer_names=payload.get("layer_names"),
        target_error_percent=float(payload.get("target_error_percent", 0.1)),
        min_bits=int(payload.get("min_bits", 4)),
        max_bits=int(payload.get("max_bits", 16)),
        min_length=int(payload.get("min_length", 32)),
        max_length=int(payload.get("max_length", 4096)),
        confidence=float(payload.get("confidence", 0.95)),
    )


def _run_adaptive_precision_formal_bundle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    layer_arrays = _parse_layer_weight_arrays(payload["layer_weights"])
    assignments = assign_synapse_precisions(
        layer_arrays,
        layer_names=payload.get("layer_names"),
        target_error=float(payload.get("target_error_percent", 0.1)) / 100.0,
        min_bits=int(payload.get("min_bits", 4)),
        max_bits=int(payload.get("max_bits", 16)),
        min_length=int(payload.get("min_length", 32)),
        max_length=int(payload.get("max_length", 4096)),
        confidence=float(payload.get("confidence", 0.95)),
    )
    module_name = str(payload.get("module_name", "adaptive_precision_plan"))
    with tempfile.TemporaryDirectory(prefix="scnc_precision_bundle_") as tmp_dir:
        bundle_manifest = write_precision_formal_evidence_bundle(
            tmp_dir, assignments, module_name=module_name
        )
        root = Path(tmp_dir)
        artifact_texts: dict[str, str] = {}
        for key, rel_path in bundle_manifest["artifacts"].items():
            artifact_path = root / rel_path
            if artifact_path.exists():
                artifact_texts[key] = artifact_path.read_text(encoding="utf-8")
            else:
                artifact_texts[key] = ""
        formal_manifest_path = root / f"{module_name}_formal_manifest.json"
        return {
            "bundle_manifest": bundle_manifest,
            "formal_manifest_json": formal_manifest_path.read_text(encoding="utf-8"),
            "artifacts_text": artifact_texts,
        }


def _execute_resolved_preset_action(resolved: dict[str, Any]) -> dict[str, Any]:
    endpoint = resolved.get("endpoint")
    payload = resolved.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("resolved payload must be an object")
    if endpoint == "/api/adaptive-precision/auto-tune":
        return _run_adaptive_precision_auto_tune_payload(payload)
    if endpoint == "/api/adaptive-precision/formal-bundle":
        return _run_adaptive_precision_formal_bundle_payload(payload)
    raise ValueError(f"preset action endpoint is not executable: {endpoint}")


def _is_executable_preset_action_endpoint(endpoint: Any) -> bool:
    return endpoint in {
        "/api/adaptive-precision/auto-tune",
        "/api/adaptive-precision/formal-bundle",
    }


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _default_flow_actions(preset_id: str) -> list[dict[str, Any]]:
    actions = get_preset_actions(preset_id)
    return [
        action
        for action in actions
        if _is_executable_preset_action_endpoint(action.get("endpoint"))
    ]


def _build_default_flow_plan_payload(preset_id: str) -> dict[str, Any]:
    actions = _default_flow_actions(preset_id)
    plan_actions: list[dict[str, Any]] = []
    for action in actions:
        action_id = action.get("id")
        endpoint = action.get("endpoint")
        method = action.get("method")
        template = action.get("payload_template")
        if not isinstance(action_id, str) or not isinstance(endpoint, str):
            continue
        if not isinstance(template, dict):
            raise ValueError(f"action '{action_id}' does not define a payload template")
        plan_actions.append(
            {
                "action_id": action_id,
                "endpoint": endpoint,
                "method": method if isinstance(method, str) else None,
                "template_keys": sorted(template.keys()),
                "template_fingerprint_sha256": _sha256_json(template),
            }
        )
    base_payload = {
        "schema_version": "sc-neurocore.studio.default-flow-plan.v1",
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": [row["action_id"] for row in plan_actions],
        "actions": plan_actions,
        "count": len(plan_actions),
    }
    plan_contract = {
        "preset_id": base_payload["preset_id"],
        "flow_id": base_payload["flow_id"],
        "action_order": base_payload["action_order"],
        "actions": [
            {
                "action_id": row["action_id"],
                "endpoint": row["endpoint"],
                "method": row["method"],
                "template_fingerprint_sha256": row["template_fingerprint_sha256"],
            }
            for row in plan_actions
        ],
    }
    base_payload["plan_fingerprint_sha256"] = _sha256_json(plan_contract)
    return base_payload


def _execute_default_flow_with_overrides(
    preset_id: str, action_overrides: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    started = time.perf_counter()
    actions = _default_flow_actions(preset_id)
    results: list[dict[str, Any]] = []
    action_order: list[str] = []
    for action in actions:
        action_id = action.get("id")
        if not isinstance(action_id, str):
            continue
        template = action.get("payload_template")
        if not isinstance(template, dict):
            raise ValueError(f"action '{action_id}' does not define a payload template")
        overrides = action_overrides.get(action_id, {})
        if not isinstance(overrides, dict):
            raise ValueError(f"action_overrides['{action_id}'] must be an object")
        resolved = _resolve_action_payload(preset_id, action_id, action, template, overrides)
        result = _execute_resolved_preset_action(resolved)
        action_order.append(action_id)
        results.append(
            {
                "action_id": action_id,
                "resolved_action": resolved,
                "result": result,
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    deterministic_results = [
        {
            "action_id": row["action_id"],
            "resolved_action": row["resolved_action"],
            "result": row["result"],
        }
        for row in results
    ]
    reproducibility_inputs = {
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": action_order,
        "resolved_actions": [row["resolved_action"] for row in deterministic_results],
    }
    reproducibility_run = {
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": action_order,
        "results": deterministic_results,
    }
    return {
        "schema_version": "sc-neurocore.studio.default-flow-run.v1",
        "preset_id": preset_id,
        "flow_id": "studio_default_adaptive_precision_v1",
        "action_order": action_order,
        "executed_count": len(results),
        "execution_time_ms": elapsed_ms,
        "results": results,
        "reproducibility_manifest": {
            "hash_algorithm": "sha256",
            "inputs_fingerprint_sha256": _sha256_json(reproducibility_inputs),
            "run_fingerprint_sha256": _sha256_json(reproducibility_run),
        },
    }


def _studio_request_id(candidate: str | None) -> str:
    if candidate is not None:
        cleaned = candidate.strip()
        if 0 < len(cleaned) <= 128 and all(
            char.isascii() and (char.isalnum() or char in "._:-") for char in cleaned
        ):
            return cleaned
    return str(uuid4())


def _studio_principal_from_headers(headers: Mapping[str, str]) -> Principal | None:
    principal_id = headers.get("x-studio-principal")
    if principal_id is None or not principal_id.strip():
        return None
    raw_roles = headers.get("x-studio-roles", "")
    roles = frozenset(role.strip() for role in raw_roles.split(",") if role.strip())
    return Principal(principal_id=principal_id.strip(), roles=roles)


def _studio_identity_from_headers(
    headers: Mapping[str, str],
    *,
    authenticator: StudioIdentityAuthenticator | None,
    allow_header_principal: bool,
) -> StudioIdentityResult:
    authorization = headers.get("authorization")
    if authorization is not None and authorization.strip():
        if authenticator is None:
            return StudioIdentityResult(
                principal=None,
                failure_reason="invalid_identity_token",
            )
        return authenticator.authenticate_authorization_header(authorization)
    if allow_header_principal:
        return StudioIdentityResult(principal=_studio_principal_from_headers(headers))
    return StudioIdentityResult(principal=None)


def _studio_route_signature(app: FastAPI, request: Request) -> tuple[str, str] | None:
    for route in app.routes:
        if not isinstance(route, Route):
            continue
        match, _ = route.matches(request.scope)
        if match is Match.FULL:
            return request.method, route.path
    return None


def create_app(runtime_settings: StudioRuntimeSettings | None = None) -> FastAPI:
    app = FastAPI(title="SC-NeuroCore Studio", version="1.0.0")
    settings = runtime_settings or build_default_studio_runtime_settings()
    studio_capabilities = build_default_studio_capability_registry()
    studio_route_policies = build_default_studio_route_policy_registry()
    studio_audit_sink = (
        JsonlAuditSink(
            Path(settings.audit_log_path),
            rotation_bytes=settings.audit_rotation_bytes,
            retained_files=settings.audit_retained_files,
        )
        if settings.audit_log_path is not None
        else InMemoryAuditSink()
    )
    studio_identity_authenticator = (
        StudioIdentityAuthenticator(load_studio_identity_store(Path(settings.identity_file_path)))
        if settings.identity_file_path is not None
        else None
    )
    studio_job_root = (
        Path(settings.job_root_path)
        if settings.job_root_path is not None
        else Path(tempfile.gettempdir()) / "sc-neurocore-studio-jobs"
    )
    studio_job_manager = StudioJobManager(
        root=studio_job_root,
        allowed_kinds=DEFAULT_STUDIO_JOB_KINDS,
        default_timeout_seconds=settings.job_default_timeout_seconds,
        configured=settings.job_root_path is not None,
    )
    studio_policy_gateway = PolicyGateway(audit_sink=studio_audit_sink)
    app.state.studio_runtime_settings = settings
    app.state.studio_capabilities = studio_capabilities
    app.state.studio_route_policies = studio_route_policies
    app.state.studio_audit_sink = studio_audit_sink
    app.state.studio_identity_authenticator = studio_identity_authenticator
    app.state.studio_job_manager = studio_job_manager
    app.state.studio_policy_gateway = studio_policy_gateway
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_studio_security_headers(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        request_id = _studio_request_id(request.headers.get(settings.request_id_header))
        content_length = request.headers.get("content-length")
        if (
            content_length is not None
            and content_length.isdecimal()
            and int(content_length) > settings.max_request_body_bytes
        ):
            response = JSONResponse(
                {"detail": "Studio request body exceeds configured limit."},
                status_code=413,
            )
        elif settings.enforce_route_policies:
            route_signature = _studio_route_signature(app, request)
            if route_signature is None:
                response = JSONResponse({"detail": "unclassified_route"}, status_code=403)
            else:
                method, path_template = route_signature
                policy = studio_route_policies.policy_for(method, path_template)
                identity_result = _studio_identity_from_headers(
                    request.headers,
                    authenticator=studio_identity_authenticator,
                    allow_header_principal=settings.allow_header_principal,
                )
                try:
                    decision = studio_policy_gateway.authorize(
                        policy,
                        principal=identity_result.principal,
                        route=path_template,
                        request_id=request_id,
                        identity_failure_reason=identity_result.failure_reason,
                    )
                    if decision.allowed:
                        response = await call_next(request)
                    else:
                        response = JSONResponse(
                            {"detail": decision.reason},
                            status_code=decision.status_code,
                        )
                except AuditSinkError:
                    response = JSONResponse(
                        {"detail": "audit_append_failed"},
                        status_code=503,
                    )
        else:
            response = await call_next(request)
        for name, value in settings.http_security_headers.items():
            response.headers.setdefault(name, value)
        response.headers.setdefault(settings.request_id_header, request_id)
        return response

    # --- Health ---
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/studio/capabilities")
    def api_studio_capabilities() -> dict[str, list[dict[str, object]]]:
        return {
            "capabilities": [
                capability.to_public_dict() for capability in studio_capabilities.health_all()
            ]
        }

    @app.get("/api/studio/capabilities/{capability_id}")
    def api_studio_capability(capability_id: str) -> dict[str, object]:
        try:
            return studio_capabilities.health(capability_id).to_public_dict()
        except KeyError as exc:
            raise HTTPException(404, f"Capability '{capability_id}' not found") from exc

    @app.get("/api/studio/audit/status")
    def api_studio_audit_status() -> dict[str, bool | str | None]:
        """Return path-free health for the configured Studio audit sink."""

        return studio_audit_sink.status().to_public_dict()

    @app.get("/api/studio/jobs/status")
    def api_studio_jobs_status() -> dict[str, bool | int | list[str] | str]:
        """Return path-free local worker health for operator dashboards."""

        return studio_job_manager.status().to_public_dict()

    @app.get("/api/studio/jobs")
    def api_studio_jobs() -> dict[str, object]:
        """Return path-free local job records for administrators."""

        return studio_job_manager.list_snapshot().to_public_dict()

    @app.get("/api/studio/jobs/{job_id}")
    def api_studio_job(job_id: str) -> dict[str, object]:
        """Return one path-free local job record for administrators."""

        try:
            return studio_job_manager.record(job_id).to_public_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    @app.get("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}")
    def api_studio_job_artifact(job_id: str, artifact_path: str) -> Response:
        """Download one declared Studio job artifact after integrity validation."""

        try:
            artifact_payload = studio_job_manager.read_artifact(job_id, artifact_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_artifact_not_found") from exc
        except (StudioJobArtifactUnavailable, ValueError) as exc:
            raise HTTPException(status_code=409, detail="job_artifact_unavailable") from exc
        filename = Path(artifact_payload.artifact.relative_path).name or "artifact.bin"
        return Response(
            content=artifact_payload.payload,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Studio-Artifact-SHA256": artifact_payload.artifact.sha256,
                "X-Studio-Artifact-Size": str(artifact_payload.artifact.size_bytes),
            },
        )

    @app.get("/api/studio/operator/status")
    def api_studio_operator_status() -> dict[str, object]:
        """Return aggregate, path-free Studio operator control-plane health."""

        return build_studio_operator_status(
            settings=settings,
            capabilities=tuple(studio_capabilities.health_all()),
            audit_status=studio_audit_sink.status(),
            job_status=studio_job_manager.status(),
        ).to_public_dict()

    @app.get("/api/studio/audit/export")
    def api_studio_audit_export(limit: int = 100) -> dict[str, AuditExportValue]:
        """Return a bounded, path-free audit export for Studio administrators."""

        if not isinstance(studio_audit_sink, JsonlAuditSink):
            raise HTTPException(status_code=409, detail="audit_export_unavailable")
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=422, detail="Audit export limit must be 1..1000")
        try:
            return studio_audit_sink.export_recent(limit=limit).to_public_dict()
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_export_failed") from exc

    # --- Templates & Models ---
    @app.get("/api/templates")
    def api_templates() -> list[dict[str, Any]]:
        return list_templates()

    @app.get("/api/templates/{name}")
    def api_template(name: str) -> Any:
        t = get_template(name)
        if not t:
            raise HTTPException(404, f"Template '{name}' not found")
        return t

    @app.get("/api/models")
    def api_models() -> Any:
        return _safe(list_models)

    # --- Model scan (behavior classification) — must precede /api/models/{name} ---
    @app.get("/api/models/scan")
    def api_model_scan() -> Any:
        return _safe(lambda: scan_all_models(current=10.0, duration=100.0))

    @app.get("/api/models/{name}")
    def api_model(name: str) -> Any:
        return _safe(
            lambda: (
                get_model_detail(name)
                or (_ for _ in ()).throw(HTTPException(404, f"Model '{name}' not found"))
            )
        )

    # --- Presets (#3) ---
    @app.get("/api/presets")
    def api_presets() -> Any:
        return list_presets()

    @app.get("/api/presets/actions/catalog")
    def api_preset_actions_catalog() -> Any:
        catalog = list_preset_action_catalog()
        executable = [
            row for row in catalog if _is_executable_preset_action_endpoint(row.get("endpoint"))
        ]
        return {"actions": executable, "count": len(executable)}

    @app.get("/api/presets/{preset_id}")
    def api_preset(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        return p

    @app.get("/api/presets/{preset_id}/actions")
    def api_preset_actions(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        return {
            "preset_id": preset_id,
            "actions": get_preset_actions(preset_id),
        }

    @app.post("/api/presets/{preset_id}/actions/{action_id}/resolve")
    def api_preset_action_resolve(
        preset_id: str, action_id: str, req: PresetActionResolveRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        action = get_preset_action(preset_id, action_id)
        if not action:
            raise HTTPException(
                status_code=404,
                detail=f"Action '{action_id}' not found for preset '{preset_id}'",
            )
        template = action.get("payload_template")
        if not isinstance(template, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Action '{action_id}' does not define a payload template",
            )
        return _safe(
            lambda: _resolve_action_payload(preset_id, action_id, action, template, req.overrides)
        )

    @app.post("/api/presets/{preset_id}/actions/{action_id}/execute")
    def api_preset_action_execute(
        preset_id: str, action_id: str, req: PresetActionResolveRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")
        action = get_preset_action(preset_id, action_id)
        if not action:
            raise HTTPException(
                status_code=404,
                detail=f"Action '{action_id}' not found for preset '{preset_id}'",
            )
        template = action.get("payload_template")
        if not isinstance(template, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Action '{action_id}' does not define a payload template",
            )

        def fn() -> dict[str, Any]:
            resolved = _resolve_action_payload(
                preset_id, action_id, action, template, req.overrides
            )
            result = _execute_resolved_preset_action(resolved)
            return {"resolved_action": resolved, "result": result}

        return _safe(fn)

    @app.post("/api/presets/{preset_id}/actions/execute-all")
    def api_preset_actions_execute_all(preset_id: str, req: PresetActionsExecuteAllRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            actions = get_preset_actions(preset_id)
            executable_actions = [
                action
                for action in actions
                if _is_executable_preset_action_endpoint(action.get("endpoint"))
            ]
            results: list[dict[str, Any]] = []
            for action in executable_actions:
                action_id = action.get("id")
                if not isinstance(action_id, str):
                    continue
                template = action.get("payload_template")
                if not isinstance(template, dict):
                    raise ValueError(f"action '{action_id}' does not define a payload template")
                overrides = req.action_overrides.get(action_id, {})
                if not isinstance(overrides, dict):
                    raise ValueError(f"action_overrides['{action_id}'] must be an object")
                resolved = _resolve_action_payload(
                    preset_id, action_id, action, template, overrides
                )
                result = _execute_resolved_preset_action(resolved)
                results.append(
                    {
                        "action_id": action_id,
                        "resolved_action": resolved,
                        "result": result,
                    }
                )
            return {
                "preset_id": preset_id,
                "executed_count": len(results),
                "results": results,
            }

        return _safe(fn)

    @app.post("/api/presets/{preset_id}/default-flow/run")
    def api_preset_default_flow_run(preset_id: str, req: PresetDefaultFlowRunRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            return _execute_default_flow_with_overrides(preset_id, req.action_overrides)

        return _safe(fn)

    @app.get("/api/presets/{preset_id}/default-flow/plan")
    def api_preset_default_flow_plan(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        return _safe(lambda: _build_default_flow_plan_payload(preset_id))

    @app.get("/api/presets/{preset_id}/default-flow/contract")
    def api_preset_default_flow_contract(preset_id: str) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            plan = _build_default_flow_plan_payload(preset_id)
            template_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            return {
                "schema_version": "sc-neurocore.studio.default-flow-contract.v1",
                "preset_id": preset_id,
                "flow_id": plan["flow_id"],
                "plan": plan,
                "guarded_run_request_template": {
                    "action_order": plan["action_order"],
                    "template_fingerprints": template_fingerprints,
                    "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                    "action_overrides": {},
                },
            }

        return _safe(fn)

    @app.post("/api/presets/{preset_id}/default-flow/verify")
    def api_preset_default_flow_verify(preset_id: str, req: PresetDefaultFlowVerifyRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            plan = _build_default_flow_plan_payload(preset_id)
            expected_order = plan["action_order"]
            expected_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            expected_plan_fingerprint = plan["plan_fingerprint_sha256"]
            order_match = req.action_order == expected_order
            fingerprints_match = req.template_fingerprints == expected_fingerprints
            plan_fingerprint_match = req.plan_fingerprint_sha256 == expected_plan_fingerprint
            return {
                "schema_version": "sc-neurocore.studio.default-flow-verify.v1",
                "preset_id": preset_id,
                "flow_id": plan["flow_id"],
                "order_match": order_match,
                "fingerprints_match": fingerprints_match,
                "plan_fingerprint_match": plan_fingerprint_match,
                "verified": order_match and fingerprints_match and plan_fingerprint_match,
                "expected_action_order": expected_order,
                "expected_template_fingerprints": expected_fingerprints,
                "expected_plan_fingerprint_sha256": expected_plan_fingerprint,
            }

        return _safe(fn)

    @app.post("/api/presets/{preset_id}/default-flow/run-guarded")
    def api_preset_default_flow_run_guarded(
        preset_id: str, req: PresetDefaultFlowGuardedRunRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            plan = _build_default_flow_plan_payload(preset_id)
            expected_order = plan["action_order"]
            expected_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            expected_plan_fingerprint = plan["plan_fingerprint_sha256"]
            order_match = req.action_order == expected_order
            fingerprints_match = req.template_fingerprints == expected_fingerprints
            plan_fingerprint_match = req.plan_fingerprint_sha256 == expected_plan_fingerprint
            if not (order_match and fingerprints_match and plan_fingerprint_match):
                raise ValueError("default-flow plan verification failed; refresh plan before run")
            run_payload = _execute_default_flow_with_overrides(preset_id, req.action_overrides)
            run_payload["verification_gate"] = {
                "order_match": order_match,
                "fingerprints_match": fingerprints_match,
                "plan_fingerprint_match": plan_fingerprint_match,
                "verified": True,
            }
            return run_payload

        return _safe(fn)

    @app.post("/api/presets/{preset_id}/default-flow/run-from-contract")
    def api_preset_default_flow_run_from_contract(
        preset_id: str, req: PresetDefaultFlowRunFromContractRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            contract = req.contract
            if not isinstance(contract, dict):
                raise ValueError("contract must be an object")
            if contract.get("schema_version") != "sc-neurocore.studio.default-flow-contract.v1":
                raise ValueError("unsupported contract schema version")
            if contract.get("preset_id") != preset_id:
                raise ValueError("contract preset_id mismatch")

            guarded = contract.get("guarded_run_request_template")
            if not isinstance(guarded, dict):
                raise ValueError("contract missing guarded_run_request_template")
            action_order = guarded.get("action_order")
            template_fingerprints = guarded.get("template_fingerprints")
            plan_fingerprint = guarded.get("plan_fingerprint_sha256")
            if not isinstance(action_order, list) or not isinstance(template_fingerprints, dict):
                raise ValueError("invalid guarded run template contract")
            if not isinstance(plan_fingerprint, str):
                raise ValueError("invalid plan fingerprint in contract")

            plan = _build_default_flow_plan_payload(preset_id)
            expected_order = plan["action_order"]
            expected_fingerprints = {
                row["action_id"]: row["template_fingerprint_sha256"] for row in plan["actions"]
            }
            expected_plan_fingerprint = plan["plan_fingerprint_sha256"]

            order_match = action_order == expected_order
            fingerprints_match = template_fingerprints == expected_fingerprints
            plan_fingerprint_match = plan_fingerprint == expected_plan_fingerprint
            if not (order_match and fingerprints_match and plan_fingerprint_match):
                raise ValueError("contract drift detected; refresh contract before run")

            run_payload = _execute_default_flow_with_overrides(preset_id, req.action_overrides)
            run_payload["verification_gate"] = {
                "order_match": order_match,
                "fingerprints_match": fingerprints_match,
                "plan_fingerprint_match": plan_fingerprint_match,
                "verified": True,
            }
            run_payload["contract_verification"] = {
                "schema_version": "sc-neurocore.studio.default-flow-contract-verify.v1",
                "contract_schema_version": contract["schema_version"],
                "verified": True,
            }
            return run_payload

        return _safe(fn)

    @app.post("/api/presets/{preset_id}/default-flow/attest")
    def api_preset_default_flow_attest(preset_id: str, req: PresetDefaultFlowAttestRequest) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            run_result = req.run_result
            if run_result.get("preset_id") != preset_id:
                raise ValueError("run_result preset_id mismatch")
            if run_result.get("flow_id") != "studio_default_adaptive_precision_v1":
                raise ValueError("unsupported flow_id for attestation")
            repro = run_result.get("reproducibility_manifest")
            if not isinstance(repro, dict):
                raise ValueError("run_result missing reproducibility_manifest")
            run_fingerprint = repro.get("run_fingerprint_sha256")
            inputs_fingerprint = repro.get("inputs_fingerprint_sha256")
            if not isinstance(run_fingerprint, str) or not isinstance(inputs_fingerprint, str):
                raise ValueError("run_result reproducibility fingerprint missing")
            if len(run_fingerprint) != 64 or len(inputs_fingerprint) != 64:
                raise ValueError("run_result reproducibility fingerprint must be sha256 hex")

            plan = _build_default_flow_plan_payload(preset_id)
            attestation_payload = {
                "schema_version": "sc-neurocore.studio.default-flow-attestation.v1",
                "preset_id": preset_id,
                "flow_id": "studio_default_adaptive_precision_v1",
                "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                "inputs_fingerprint_sha256": inputs_fingerprint,
                "run_fingerprint_sha256": run_fingerprint,
            }
            attestation_payload["attestation_fingerprint_sha256"] = _sha256_json(
                attestation_payload
            )
            return attestation_payload

        return _safe(fn)

    @app.post("/api/presets/{preset_id}/default-flow/attest/verify")
    def api_preset_default_flow_attest_verify(
        preset_id: str, req: PresetDefaultFlowAttestationVerifyRequest
    ) -> Any:
        p = get_preset(preset_id)
        if not p:
            raise HTTPException(404, f"Preset '{preset_id}' not found")

        def fn() -> dict[str, Any]:
            run_result = req.run_result
            attestation = req.attestation
            if run_result.get("preset_id") != preset_id:
                raise ValueError("run_result preset_id mismatch")
            if run_result.get("flow_id") != "studio_default_adaptive_precision_v1":
                raise ValueError("unsupported flow_id for attestation verification")
            repro = run_result.get("reproducibility_manifest")
            if not isinstance(repro, dict):
                raise ValueError("run_result missing reproducibility_manifest")
            run_fingerprint = repro.get("run_fingerprint_sha256")
            inputs_fingerprint = repro.get("inputs_fingerprint_sha256")
            if not isinstance(run_fingerprint, str) or not isinstance(inputs_fingerprint, str):
                raise ValueError("run_result reproducibility fingerprint missing")
            if len(run_fingerprint) != 64 or len(inputs_fingerprint) != 64:
                raise ValueError("run_result reproducibility fingerprint must be sha256 hex")

            plan = _build_default_flow_plan_payload(preset_id)
            expected_attestation_base = {
                "schema_version": "sc-neurocore.studio.default-flow-attestation.v1",
                "preset_id": preset_id,
                "flow_id": "studio_default_adaptive_precision_v1",
                "plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                "inputs_fingerprint_sha256": inputs_fingerprint,
                "run_fingerprint_sha256": run_fingerprint,
            }
            expected_attestation_fingerprint = _sha256_json(expected_attestation_base)

            schema_match = (
                isinstance(attestation, dict)
                and attestation.get("schema_version")
                == "sc-neurocore.studio.default-flow-attestation.v1"
            )
            plan_match = (
                schema_match
                and attestation.get("plan_fingerprint_sha256") == plan["plan_fingerprint_sha256"]
            )
            inputs_match = (
                schema_match and attestation.get("inputs_fingerprint_sha256") == inputs_fingerprint
            )
            run_match = (
                schema_match and attestation.get("run_fingerprint_sha256") == run_fingerprint
            )
            attestation_fingerprint_match = (
                schema_match
                and attestation.get("attestation_fingerprint_sha256")
                == expected_attestation_fingerprint
            )

            return {
                "schema_version": "sc-neurocore.studio.default-flow-attestation-verify.v1",
                "preset_id": preset_id,
                "verified": bool(
                    schema_match
                    and plan_match
                    and inputs_match
                    and run_match
                    and attestation_fingerprint_match
                ),
                "checks": {
                    "schema_match": bool(schema_match),
                    "plan_fingerprint_match": bool(plan_match),
                    "inputs_fingerprint_match": bool(inputs_match),
                    "run_fingerprint_match": bool(run_match),
                    "attestation_fingerprint_match": bool(attestation_fingerprint_match),
                },
                "expected_plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
                "expected_attestation_fingerprint_sha256": expected_attestation_fingerprint,
            }

        return _safe(fn)

    # --- Simulation (with auto-classification + cache) ---
    @app.post("/api/simulate")
    def api_simulate(req: SimulateRequest) -> Any:
        cache_key = {"_type": "ode", **req.model_dump()}
        cached = _cache.get(cache_key)
        if cached:
            return cached

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
            _cache.put(cache_key, result)
            return result

        return _safe(fn)

    @app.post("/api/models/simulate")
    def api_model_simulate(req: ModelSimulateRequest) -> Any:
        cache_key = {"_type": "model", **req.model_dump()}
        cached = _cache.get(cache_key)
        if cached:
            return cached

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
            _cache.put(cache_key, result)
            return result

        return _safe(fn)

    @app.get("/api/cache/stats")
    def api_cache_stats() -> dict[str, int]:
        return {"hits": _cache.hits, "misses": _cache.misses, "size": len(_cache._cache)}

    # --- Comparison (#1) ---
    @app.post("/api/compare")
    def api_compare(req: CompareRequest) -> Any:
        def fn() -> dict[str, Any]:
            sim_a = _make_simulate_fn(req.config_a)
            sim_b = _make_simulate_fn(req.config_b)
            return {"a": sim_a(), "b": sim_b()}

        return _safe(fn)

    # --- f-I Curve ---
    @app.post("/api/fi-curve")
    def api_fi_curve(req: FICurveRequest) -> Any:
        def fn() -> dict[str, Any]:
            import numpy as np

            sim_fn = _make_simulate_fn(req.model_dump())
            currents = np.linspace(req.i_min, req.i_max, req.i_steps).tolist()
            rates = [sim_fn(current=float(I))["stats"]["rate_hz"] for I in currents]
            return {"currents": currents, "rates": rates}

        return _safe(fn)

    # --- Bifurcation (#2) ---
    @app.post("/api/bifurcation")
    def api_bifurcation(req: BifurcationRequest) -> Any:
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
            return bifurcation_sweep(
                sim_fn, base_cfg, req.sweep_param, req.sweep_min, req.sweep_max, req.sweep_steps
            )

        return _safe(fn)

    # --- Sensitivity (#8) ---
    @app.post("/api/sensitivity")
    def api_sensitivity(req: SensitivityRequest) -> Any:
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
            return sensitivity_analysis(sim_fn, base_cfg, param_names)

        return _safe(fn)

    # --- Nullclines (#9) ---
    @app.post("/api/nullclines")
    def api_nullclines(req: NullclineRequest) -> Any:
        def fn() -> dict[str, Any]:
            ranges = {k: (v[0], v[1]) for k, v in req.ranges.items()}
            return nullclines_2d(req.equations, req.params, req.var_names, ranges, req.grid_size)

        return _safe(fn)

    # --- Precision Compare (#5) ---
    @app.post("/api/precision")
    def api_precision(req: PrecisionRequest) -> Any:
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

    # --- Adaptive Precision Auto-Tune ---
    @app.post("/api/adaptive-precision/auto-tune")
    def api_adaptive_precision_auto_tune(req: AdaptivePrecisionAutoTuneRequest) -> Any:
        payload = req.model_dump()
        return _safe(lambda: _run_adaptive_precision_auto_tune_payload(payload))

    @app.post("/api/adaptive-precision/formal-bundle")
    def api_adaptive_precision_formal_bundle(req: AdaptivePrecisionFormalBundleRequest) -> Any:
        payload = req.model_dump()
        return _safe(lambda: _run_adaptive_precision_formal_bundle_payload(payload))

    # --- Compile (#5 adjacent) ---
    @app.post("/api/compile")
    def api_compile(req: CompileRequest) -> Any:
        def fn() -> dict[str, Any]:
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
    def api_freq_response(req: FreqResponseRequest) -> Any:
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
            return frequency_response(
                sim_fn, base_cfg, req.freq_min, req.freq_max, req.n_freqs, req.amplitude
            )

        return _safe(fn)

    # --- 2D Heatmap ---
    @app.post("/api/heatmap")
    def api_heatmap(req: HeatmapRequest) -> Any:
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

    # --- Firing Pattern Classification ---
    @app.post("/api/classify")
    def api_classify(req: SimulateRequest) -> Any:
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

    # --- One-click Characterisation ---
    @app.post("/api/characterize")
    def api_characterize(req: ModelSimulateRequest) -> Any:
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
            return characterize_model(sim_fn, base_cfg)

        return _safe(fn)

    # --- Multi-model Overlay ---
    @app.post("/api/multi-simulate")
    def api_multi_simulate(configs: list[ModelSimulateRequest]) -> Any:
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
                results.append(r)
            return results

        return _safe(fn)

    # --- Data Import (CSV voltage trace) ---
    @app.post("/api/import-trace")
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

    # --- E-I Network Simulation ---
    @app.post("/api/network/ei")
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

    # --- Compiler Inspector (Block 2) ---
    @app.post("/api/ir/build")
    def api_ir_build(req: SimulateRequest) -> Any:
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
    def api_ir_verify(data: dict[str, Any]) -> Any:
        ir_text = data.get("ir_text", "")
        if not ir_text:
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: verify_ir(ir_text))

    @app.post("/api/ir/emit-sv")
    def api_ir_emit_sv(data: dict[str, Any]) -> Any:
        ir_text = data.get("ir_text", "")
        if not ir_text:
            raise HTTPException(422, "ir_text required")
        return _safe(lambda: emit_systemverilog(ir_text))

    @app.post("/api/ir/emit-sv-direct")
    def api_ir_emit_sv_direct(req: SimulateRequest) -> Any:
        return _safe(
            lambda: emit_sv_from_equation(
                equations=req.equations,
                params=req.params,
                threshold=req.threshold,
                reset=req.reset,
            )
        )

    @app.post("/api/ir/cosim")
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

    # --- Synthesis Dashboard (Block 3) ---
    @app.get("/api/synth/tools-status")
    def api_synth_tools() -> Any:
        return check_tools()

    @app.post("/api/synth/run")
    def api_synth_run(data: dict[str, Any]) -> Any:
        verilog = data.get("verilog", "")
        target = data.get("target", "ice40")
        if not verilog:
            raise HTTPException(422, "verilog source required")
        return _safe(lambda: run_synthesis(verilog, target))

    @app.post("/api/synth/multi-target")
    def api_synth_multi(data: dict[str, Any]) -> Any:
        verilog = data.get("verilog", "")
        if not verilog:
            raise HTTPException(422, "verilog source required")
        return _safe(lambda: multi_target_synthesis(verilog))

    @app.post("/api/synth/estimate")
    def api_synth_estimate(data: dict[str, Any]) -> Any:
        raw_ir_op_count = data.get("ir_op_count", 0)
        target = data.get("target", "ice40")
        if not isinstance(raw_ir_op_count, int):
            raise HTTPException(422, "ir_op_count must be an integer >= 1")
        ir_op_count = raw_ir_op_count
        if ir_op_count < 1:
            raise HTTPException(422, "ir_op_count must be >= 1")
        return _safe(lambda: estimate_resources(ir_op_count, target))

    @app.post("/api/synth/pnr")
    def api_synth_pnr(data: dict[str, Any]) -> Any:
        json_path = data.get("json_path", "")
        target = data.get("target", "ice40")
        if not json_path:
            raise HTTPException(422, "json_path required")
        return _safe(lambda: run_pnr(json_path, target))

    # --- Integration (Block 6) ---
    @app.post("/api/project/save")
    def api_project_save(data: dict[str, Any]) -> Any:
        name = data.get("name", "")
        state = data.get("state", {})
        if not name:
            raise HTTPException(422, "Project name required")
        return _safe(lambda: save_project(name, state))

    @app.get("/api/project/list")
    def api_project_list() -> Any:
        return list_projects()

    @app.get("/api/project/load/{name}")
    def api_project_load(name: str) -> Any:
        result = _safe(lambda: load_project(name))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @app.delete("/api/project/{name}")
    def api_project_delete(name: str) -> Any:
        result = _safe(lambda: delete_project(name))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @app.post("/api/pipeline/run")
    def api_pipeline_run(data: dict[str, Any]) -> Any:
        graph = data.get("graph", {})
        target = data.get("target", "ice40")
        return _safe(lambda: run_pipeline(graph, target))

    # --- Network Canvas (Block 5) ---
    @app.get("/api/graph/models")
    def api_graph_models() -> Any:
        return _safe(graph_available_models)

    @app.post("/api/graph/population")
    def api_create_population(data: dict[str, Any]) -> Any:
        return create_population(
            **{
                k: v
                for k, v in data.items()
                if k in ("label", "model", "count", "neuron_type", "x", "y")
            }
        )

    @app.post("/api/graph/projection")
    def api_create_projection(data: dict[str, Any]) -> Any:
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
    def api_validate_graph(data: dict[str, Any]) -> Any:
        errors = validate_graph(data)
        return {"valid": len(errors) == 0, "errors": errors}

    @app.post("/api/graph/simulate")
    def api_simulate_graph(data: dict[str, Any]) -> Any:
        return _safe(lambda: simulate_graph(data))

    @app.post("/api/graph/export-nir")
    def api_export_nir(data: dict[str, Any]) -> Any:
        return _safe(lambda: graph_to_nir(data))

    @app.post("/api/graph/import-nir")
    def api_import_nir(data: dict[str, Any]) -> Any:
        return _safe(lambda: nir_to_graph(data))

    # --- Training Monitor (Block 4) ---
    @app.get("/api/training/surrogates")
    def api_surrogates() -> Any:
        return list_surrogates()

    @app.get("/api/training/cell-types")
    def api_cell_types() -> Any:
        return list_cell_types()

    @app.post("/api/training/start")
    def api_training_start(data: dict[str, Any]) -> Any:
        return _safe(lambda: start_training(data, studio_job_manager))

    @app.post("/api/training/stop")
    def api_training_stop(data: dict[str, Any]) -> Any:
        job_id = data.get("job_id", "")
        if not job_id:
            raise HTTPException(422, "job_id required")
        return stop_training(job_id, studio_job_manager)

    @app.get("/api/training/jobs")
    def api_training_jobs() -> Any:
        return list_jobs()

    @app.get("/api/training/status/{job_id}")
    def api_training_status(job_id: str) -> Any:
        result = get_training_status(job_id)
        if result.get("error") and "job_id" not in result:
            raise HTTPException(404, result["error"])
        return result

    @app.get("/api/training/stream/{job_id}")
    def api_training_stream(job_id: str) -> Any:
        from starlette.responses import StreamingResponse

        return StreamingResponse(
            stream_metrics(job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- SVG export ---
    @app.post("/api/export/svg")
    def export_svg(req: ModelSimulateRequest) -> Any:
        from fastapi.responses import Response
        from sc_neurocore.studio.svg_export import traces_to_svg

        def fn() -> Any:
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
                dt=req.dt or 0.1,
            )
            return Response(content=svg, media_type="image/svg+xml")

        return _safe(fn)

    # --- WebSocket progress streaming ---
    @app.websocket("/ws/progress")
    async def ws_progress(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin not in settings.websocket_allowed_origins:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        from sc_neurocore.studio.progress import ws_progress_handler

        try:
            await ws_progress_handler(websocket)
        except WebSocketDisconnect:
            pass

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
        def serve_index() -> Any:
            return FileResponse(os.path.join(dist_dir, "index.html"))

        app.mount("/", StaticFiles(directory=dist_dir), name="static")

    return app


app = create_app()
