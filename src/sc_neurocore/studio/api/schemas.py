# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio API request schemas

"""Typed request bodies shared by the Studio responsibility routers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, Field, StringConstraints


class SimulateRequest(BaseModel):
    """Request body for direct ODE simulation in Studio."""

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
    """Request body for model-catalogue simulation in Studio."""

    # Accept ``model_name`` as well so the field matches the rest of the model API
    # surface (fi-curve, characterize, sensitivity, …) and the Studio frontend,
    # which sends ``model_name`` consistently.
    name: str = Field(validation_alias=AliasChoices("name", "model_name"))
    params: dict[str, float] | None = None
    dt: float | None = None
    duration: float = Field(default=100.0, gt=0)
    current: float = 10.0
    protocol: str = "constant"


class FICurveRequest(BaseModel):
    """Request body for firing-rate versus current sweeps."""

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


class DclsEvaluateRequest(BaseModel):
    """Request body for DCLS kernel parity evaluation."""

    centre_q88: int = Field(default=512, ge=0, le=65535)
    sigma_q88: int = Field(default=512, gt=0, le=65535)
    n_taps: int = Field(default=8, ge=1, le=256)
    spikes: list[int] | None = None
    weights_q88: list[int] | None = None


class BenchmarkRunRequest(BaseModel):
    """Request body for local Studio benchmark runs."""

    n_channels: int = Field(default=512, ge=16, le=8192)
    n_taps: int = Field(default=32, ge=4, le=256)
    repeats: int = Field(default=12, ge=3, le=50)


class BenchmarkContributeRequest(BaseModel):
    """Request body for benchmark-databank contribution uploads."""

    submission: dict[str, Any]
    handle: str = Field(default="", max_length=40)


class CompileRequest(BaseModel):
    """Request body for equation-to-SystemVerilog compilation."""

    equations: list[str] = Field(min_length=1)
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    module_name: str = "sc_neuron"


class ModelCompileRequest(BaseModel):
    """Request body for schema-backed catalogue-model RTL compilation."""

    model_name: str = Field(min_length=1)
    params: dict[str, float] = Field(default_factory=dict)
    dt: float | None = Field(default=None, gt=0)
    integrator: str | None = Field(default=None, min_length=1)
    q_format: str = Field(default="Q8.8", pattern=r"^Q\d+\.\d+$")
    module_name: str | None = Field(default=None, min_length=1)


class ModelCosimRequest(ModelCompileRequest):
    """Request body for real selected-model RTL co-simulation."""

    current: float = Field(default=10.0, allow_inf_nan=False)
    n_steps: int = Field(default=128, ge=1, le=2048)


class BifurcationRequest(BaseModel):
    """Request body for one-parameter bifurcation sweeps."""

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
    """Request body for Studio model sensitivity analysis."""

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
    """Request body for two-dimensional nullcline analysis."""

    equations: list[str]
    params: dict[str, float]
    var_names: list[str]
    ranges: dict[str, list[float]]
    grid_size: int = Field(default=60, ge=20, le=150)


class PrecisionRequest(BaseModel):
    """Request body for float versus fixed-point precision comparisons."""

    equations: list[str]
    threshold: str | None = None
    reset: str | None = None
    params: dict[str, float] | None = None
    init: dict[str, float] | None = None
    dt: float = 0.1
    duration: float = 200.0
    current: float = 10.0


class AdaptivePrecisionAutoTuneRequest(BaseModel):
    """Request body for adaptive synapse-precision auto-tuning."""

    layer_weights: list[list[list[float]] | list[float]]
    layer_names: list[str] | None = None
    target_error_percent: float = Field(default=0.1, gt=0.0, le=10.0)
    min_bits: int = Field(default=4, ge=1, le=32)
    max_bits: int = Field(default=16, ge=1, le=32)
    min_length: int = Field(default=32, ge=1, le=65536)
    max_length: int = Field(default=4096, ge=1, le=262144)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class AdaptivePrecisionFormalBundleRequest(BaseModel):
    """Request body for adaptive-precision formal evidence bundles."""

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
    """Request body for resolving one preset action template."""

    overrides: dict[str, Any] = Field(default_factory=dict)


class PresetActionsExecuteAllRequest(BaseModel):
    """Request body for executing all actions attached to a preset."""

    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowRunRequest(BaseModel):
    """Request body for running a preset's default action flow."""

    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowVerifyRequest(BaseModel):
    """Request body for verifying a preset default-flow plan."""

    action_order: list[str]
    template_fingerprints: dict[str, Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]]
    plan_fingerprint_sha256: Annotated[str | None, StringConstraints(pattern=r"^[0-9a-f]{64}$")] = (
        None
    )


class PresetDefaultFlowGuardedRunRequest(BaseModel):
    """Request body for running a preset flow with fingerprint guards."""

    action_order: list[str]
    template_fingerprints: dict[str, Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]]
    plan_fingerprint_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowRunFromContractRequest(BaseModel):
    """Request body for running a preset flow from a stored contract."""

    contract: dict[str, Any]
    action_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PresetDefaultFlowAttestRequest(BaseModel):
    """Request body for attesting a completed preset flow run."""

    run_result: dict[str, Any]


class StudioIdentityServiceAccountUpdateRequest(BaseModel):
    """Request body for admin service-account metadata updates."""

    roles: list[str] = Field(min_length=1)
    active: bool
    expires_at_utc: str | None = None


class StudioBrowserUserUpdateRequest(BaseModel):
    """Request body for admin browser-user metadata updates."""

    roles: list[str] = Field(min_length=1)
    active: bool
    expires_at_utc: str | None = None


class StudioBrowserUserCreateRequest(BaseModel):
    """Request body for admin browser-user creation."""

    username: str = Field(min_length=1, max_length=128)
    principal_id: str = Field(min_length=1, max_length=256)
    roles: list[str] = Field(min_length=1)
    password: str = Field(min_length=1, max_length=4096)
    active: bool = True
    expires_at_utc: str | None = None


class StudioBrowserUserPasswordRotateRequest(BaseModel):
    """Request body for admin browser-user password rotation."""

    password: str = Field(min_length=1, max_length=4096)


class StudioBrowserLoginRequest(BaseModel):
    """Request body for browser-user login."""

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=4096)


class StudioEvidenceBundleRequest(BaseModel):
    """Request body for admin evidence bundle export."""

    project_name: str | None = Field(default=None, min_length=1, max_length=128)
    simulation_results: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    analysis_results: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    model_scan_results: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    weight_restore_results: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    weight_restore_attach_results: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    default_flow_runs: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    default_flow_attestations: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    job_ids: list[str] = Field(default_factory=list, max_length=64)
    include_audit: bool = True
    audit_limit: int = Field(default=100, ge=1, le=1000)
    command_replay: dict[str, Any] | None = None


class StudioAuditQuarantineArchiveRequest(BaseModel):
    """Request body for admin audit quarantine archive creation."""

    limit: int = Field(default=100, ge=1, le=1000)


class StudioAuditQuarantineArchiveValidateRequest(BaseModel):
    """Request body for admin audit quarantine archive validation."""

    archive: dict[str, Any]
    manifest: dict[str, Any] | None = None


class StudioAuditQuarantineArchiveRestoreRequest(BaseModel):
    """Request body for admin audit quarantine archive restore materialization."""

    archive: dict[str, Any]
    manifest: dict[str, Any] | None = None


class StudioAuditQuarantineArchivePurgeRequest(BaseModel):
    """Request body for admin audit quarantine archive retention purges."""

    retain_latest: int = Field(default=10, ge=1, le=1000)


class StudioTrainingWeightRestoreRequest(BaseModel):
    """Request body for admin training weight-restore materialization."""

    source_job_id: str = Field(min_length=1, max_length=128)
    expected_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class StudioTrainingWeightAttachRequest(BaseModel):
    """Request body for admin training weight-restore warm-start attach."""

    source_job_id: str = Field(min_length=1, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)
    expected_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class StudioTrainingWeightLiveAttachRequest(BaseModel):
    """Request body for admin training weight-restore live attach."""

    target_job_id: str = Field(min_length=1, max_length=128)
    source_job_id: str = Field(min_length=1, max_length=128)
    expected_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class PresetDefaultFlowAttestationVerifyRequest(BaseModel):
    """Request body for verifying a preset flow attestation."""

    run_result: dict[str, Any]
    attestation: dict[str, Any]


class CompareRequest(BaseModel):
    """Request body for comparing two Studio simulation configurations."""

    config_a: dict[str, Any]
    config_b: dict[str, Any]


class FreqResponseRequest(BaseModel):
    """Request body for frequency-response analysis."""

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
    """Request body for two-parameter response heatmap analysis."""

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


class AnalysisJobRequest(BaseModel):
    """Request body for asynchronous heavy analysis job submission.

    The ``analysis`` field selects the synchronous analysis kind. ``payload``
    must match the corresponding synchronous request schema (for example
    :class:`BifurcationRequest` when ``analysis`` is ``bifurcation``).
    """

    analysis: Literal["fi_curve", "bifurcation", "heatmap", "sensitivity"]
    payload: dict[str, Any] = Field(default_factory=dict)


class NetworkRequest(BaseModel):
    """Request body for balanced excitatory-inhibitory network simulation."""

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
    """Request body for Studio script and one-liner generation."""

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
