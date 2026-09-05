# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio effective experiment contract

"""One effective experiment specification for every Studio simulation run.

A request (catalogue model or equation playground) is resolved into an
:class:`ExperimentSpec` before anything runs: the model revision (package
version, module and descriptor digests, canonical schema profile) or the
equation digest, the numerical profile and the effective time step, the exact
step count (an oversized synchronous run is refused, never shortened), the
typed initial state, every protocol parameter with the drive digest, the
randomness contract (kind, seed, seed source, replay or fresh trial), the
backend selection with its rejected alternatives and the runtime digest. The
spec's digest is the cache key, so two runs that differ in any of these cannot
share a cached result, and a fresh stochastic trial is never cached. Explicit
request defaults resolve to the same spec as omitted ones.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

import sc_neurocore
from sc_neurocore.neurons import equation_builder
from sc_neurocore.neurons.facet_receipts import descriptor_contract_digest_of
from sc_neurocore.neurons.model_catalogue import descriptor_path, load_descriptor
from sc_neurocore.neurons.model_identity import ModelIdentityError, schema_for_class
from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.neurons.universal_dsl import _SCHEMA_DIR, load_schema
from sc_neurocore.studio.model_run_contract import (
    STUDIO_DEFAULT_DT_MS,
    SUPPORTED_PROTOCOLS,
    ModelRunInputs,
    resolve_drive_trace,
    resolve_model_run_inputs,
)
from sc_neurocore.studio.simulation import MAX_STEPS, _make_current_trace
from sc_neurocore.studio.state_layout import declared_state

EXPERIMENT_SCHEMA_VERSION = "studio.experiment-spec.v1"
JOB_MAX_STEPS = 2_000_000
SIMULATION_JOB_ROUTE = "POST /api/analysis/jobs (analysis=simulate)"
DEFAULT_NOISE_SEED = 0
FRESH_SEED_BITS = 16

Source = Literal["model", "ode"]
Trial = Literal["replay", "fresh"]
RandomnessKind = Literal["none", "seeded-model", "diffusion-noise"]
SeedSource = Literal["none", "request", "model-default", "playground-default", "drawn"]
ExecutionMode = Literal["refused", "job_required"]

_EQUATION_PATTERN = re.compile(r"d(\w+)/dt\s*=\s*(.+)")
_NOISE_PATTERN = re.compile(r"\bxi\b")
_PROTOCOL_FRACTIONS: dict[str, dict[str, float]] = {
    "step": {"step_onset_fraction": 0.2, "step_offset_fraction": 0.8},
    "ramp": {"ramp_start": 0.0},
    "pulse": {"pulse_period_steps": 0.2, "pulse_on_fraction_of_period": 0.2},
}


class ExperimentRejected(ValueError):
    """Raised when a request cannot become one effective experiment.

    Parameters
    ----------
    field : str
        Request field that failed.
    reason : str
        Bounded, path-free reason.
    execution_mode : {"refused", "job_required"}
        ``job_required`` when the run is valid but too large for the
        synchronous route and must be submitted as a job.
    """

    def __init__(
        self, *, field: str, reason: str, execution_mode: ExecutionMode = "refused"
    ) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason
        self.execution_mode: ExecutionMode = execution_mode

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "experiment_rejected",
            "field": self.field,
            "reason": self.reason,
            "execution_mode": self.execution_mode,
            "recommended_route": SIMULATION_JOB_ROUTE
            if self.execution_mode == "job_required"
            else None,
        }


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, default=str, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_block() -> dict[str, str]:
    return {
        "package_version": sc_neurocore.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "equation_builder_sha256": _sha256_file(Path(equation_builder.__file__)),
    }


def _draw_seed() -> int:
    """Draw a fresh non-zero seed that every seeded model accepts (16-bit)."""
    return secrets.randbelow((1 << FRESH_SEED_BITS) - 1) + 1


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """The resolved, digest-bound specification of one Studio run.

    ``public`` is the path-free specification (with ``experiment_sha256``);
    ``run_kwargs`` is the executable material for the run entrypoint and is
    never returned to a client; ``cacheable`` is ``False`` for a fresh
    stochastic trial.
    """

    source: Source
    public: dict[str, Any]
    run_kwargs: dict[str, Any]
    cacheable: bool
    n_steps: int
    dt: float
    duration_ms: float

    @property
    def experiment_sha256(self) -> str:
        """Digest of the public specification: the cache key of the run."""
        return str(self.public["experiment_sha256"])

    def to_public_dict(self) -> dict[str, Any]:
        """Return the path-free experiment specification."""
        return dict(self.public)


def _finish(
    *,
    source: Source,
    body: dict[str, Any],
    run_kwargs: dict[str, Any],
    cacheable: bool,
    n_steps: int,
    dt: float,
    duration_ms: float,
) -> ExperimentSpec:
    body = {"schema_version": EXPERIMENT_SCHEMA_VERSION, "source": source, **body}
    body["experiment_sha256"] = _sha256_json(body)
    body["cache"] = {"key": body["experiment_sha256"], "cacheable": cacheable}
    return ExperimentSpec(
        source=source,
        public=body,
        run_kwargs=run_kwargs,
        cacheable=cacheable,
        n_steps=n_steps,
        dt=dt,
        duration_ms=duration_ms,
    )


def _steps(duration: float, dt: float, *, max_steps: int) -> int:
    requested = int(duration / dt)
    if requested < 1:
        raise ExperimentRejected(
            field="duration",
            reason=f"duration {duration} ms with dt {dt} ms yields no complete step",
        )
    if requested > max_steps:
        raise ExperimentRejected(
            field="duration",
            reason=(
                f"{requested} steps exceed the synchronous limit of {max_steps}; the run is not "
                "shortened, submit it as a simulation job or reduce duration or dt"
            ),
            execution_mode="job_required",
        )
    return requested


def _protocol_block(
    *, protocol: str, current: float, frequency_hz: float, samples: np.ndarray[Any, Any]
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "kind": protocol,
        "current": current,
        "frequency_hz": frequency_hz if protocol == "sine" else None,
        "drive_sha256": hashlib.sha256(
            np.ascontiguousarray(samples, dtype=np.float64).tobytes()
        ).hexdigest(),
        "drive_unit": "model-defined",
    }
    block.update(_PROTOCOL_FRACTIONS.get(protocol, {}))
    return block


def _model_revision(inputs: ModelRunInputs) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the model revision block and the numerical profile block."""
    cls = inputs.cls
    module_file = inspect.getsourcefile(cls)
    revision: dict[str, Any] = {
        "class_name": inputs.model,
        "module": cls.__module__,
        "module_sha256": _sha256_file(Path(module_file)) if module_file else "",
        "descriptor_sha256": "",
        "descriptor_contract_digest": "",
        "schema_profile": "",
        "schema_sha256": "",
    }
    path = descriptor_path(inputs.model)
    descriptor = load_descriptor(inputs.model)
    if descriptor is not None and path.is_file():
        revision["descriptor_sha256"] = _sha256_file(path)
        revision["descriptor_contract_digest"] = descriptor_contract_digest_of(path)
    numerical: dict[str, Any] = {
        "method": descriptor.integration_method if descriptor is not None else "",
        "family": "",
        "dt": inputs.dt,
        "dt_source": inputs.dt_source,
        "substeps": 1,
        "time_unit": "ms",
    }
    try:
        stem = schema_for_class(inputs.model)
        schema = load_schema(stem)
    except (ModelIdentityError, FileNotFoundError, ValueError):
        return revision, numerical
    revision["schema_profile"] = stem
    for suffix in (".toml", ".json"):
        candidate = _SCHEMA_DIR / f"{stem}{suffix}"
        if candidate.is_file():
            revision["schema_sha256"] = _sha256_file(candidate)
            break
    profile = resolve_profile(schema, stem=stem)
    numerical["family"] = profile.numerical.family
    numerical["substeps"] = profile.numerical.substeps
    numerical["time_unit"] = profile.numerical.time_unit or "ms"
    if not numerical["method"]:
        numerical["method"] = profile.numerical.method
    return revision, numerical


def _model_randomness(
    inputs: ModelRunInputs, *, seed: int | None, trial: Trial
) -> tuple[dict[str, Any], int | None]:
    """Resolve the randomness contract of a catalogue model run."""
    contract = inputs.contracts.overridable.get("seed")
    if contract is None:
        if seed is not None:
            raise ExperimentRejected(
                field="seed",
                reason=f"{inputs.model} declares no randomness contract; a seed is not accepted",
            )
        return {
            "kind": "none",
            "seed": None,
            "seed_source": "none",
            "trial": trial,
            "effective_trial": "replay",
            "generator": None,
            "note": "deterministic model: a fresh trial equals a replay",
        }, None
    if "seed" in inputs.overrides_applied and seed is not None:
        raise ExperimentRejected(
            field="seed", reason="seed given both as a parameter override and as the seed field"
        )
    if trial == "fresh":
        drawn = _draw_seed()
        return {
            "kind": "seeded-model",
            "seed": drawn,
            "seed_source": "drawn",
            "trial": trial,
            "effective_trial": "fresh",
            "generator": "model seed field",
        }, drawn
    if seed is not None:
        return {
            "kind": "seeded-model",
            "seed": seed,
            "seed_source": "request",
            "trial": trial,
            "effective_trial": "replay",
            "generator": "model seed field",
        }, seed
    if "seed" in inputs.overrides_applied:
        override = int(inputs.constructor_kwargs["seed"])
        return {
            "kind": "seeded-model",
            "seed": override,
            "seed_source": "request",
            "trial": trial,
            "effective_trial": "replay",
            "generator": "model seed field",
        }, None
    default = contract.default
    if default is None:
        raise ExperimentRejected(
            field="seed",
            reason=(
                f"{inputs.model} draws independent entropy when no seed is given; pass a "
                "seed for a replay or request trial=fresh"
            ),
        )
    return {
        "kind": "seeded-model",
        "seed": int(default),
        "seed_source": "model-default",
        "trial": trial,
        "effective_trial": "replay",
        "generator": "model seed field",
    }, None


def resolve_model_experiment(
    request: Mapping[str, Any], *, max_steps: int = MAX_STEPS
) -> ExperimentSpec:
    """Resolve a catalogue-model request into its effective experiment.

    Parameters
    ----------
    request : mapping
        Validated request fields: ``name``, ``params``, ``dt``, ``duration``,
        ``current``, ``protocol``, ``frequency_hz``, ``seed``, ``trial``.
    max_steps : int
        Largest step count this caller executes; a larger run is rejected
        with ``execution_mode = job_required``.

    Raises
    ------
    ModelInputError
        From the run contract (unknown model, parameter, unsupported step).
    ExperimentRejected
        Seed on a deterministic model, seed given twice, no complete step or
        an oversized run.
    """
    name = request["name"]
    params = dict(request.get("params") or {})
    seed = request.get("seed")
    trial: Trial = request.get("trial", "replay")
    protocol = request.get("protocol", "constant")
    current = float(request.get("current", 10.0))
    frequency_hz = float(request.get("frequency_hz", 10.0))
    duration = float(request.get("duration", 100.0))
    inputs = resolve_model_run_inputs(name, params, request.get("dt"))
    randomness, effective_seed = _model_randomness(inputs, seed=seed, trial=trial)
    run_params = dict(params)
    if effective_seed is not None:
        run_params["seed"] = effective_seed
        inputs = resolve_model_run_inputs(name, run_params, request.get("dt"))
    n_steps = _steps(duration, inputs.dt, max_steps=max_steps)
    trace = resolve_drive_trace(
        inputs,
        protocol=protocol,
        current=current,
        duration=duration,
        frequency_hz=frequency_hz,
        max_steps=n_steps,
    )
    revision, numerical = _model_revision(inputs)
    _source, _stem, declared = declared_state(name)
    body: dict[str, Any] = {
        "model": revision,
        "numerical": numerical,
        "steps": {
            "n_steps": n_steps,
            "duration_requested_ms": duration,
            "duration_effective_ms": n_steps * inputs.dt,
            "synchronous_limit": max_steps,
        },
        "parameters": inputs.effective_parameters(),
        "initial_state": {spec.name: spec.declared_init for spec in declared},
        "initial_state_source": "descriptor-declared; the run reports the constructed snapshot",
        "protocol": _protocol_block(
            protocol=protocol, current=current, frequency_hz=frequency_hz, samples=trace.samples
        ),
        "randomness": randomness,
        "backend": {
            "selected": "python",
            "rejected": [
                {
                    "name": "rust-batch",
                    "reason": "exports the membrane voltage only, no initial snapshot, default construction; not admitted for custody runs",
                }
            ],
        },
        "runtime": _runtime_block(),
    }
    run_kwargs = {
        "name": name,
        "param_overrides": run_params or None,
        "dt": request.get("dt"),
        "duration": duration,
        "current": current,
        "protocol": protocol,
        "frequency_hz": frequency_hz,
        "use_fast_path": False,
        "max_steps": max_steps,
    }
    return _finish(
        source="model",
        body=body,
        run_kwargs=run_kwargs,
        cacheable=randomness["effective_trial"] == "replay",
        n_steps=n_steps,
        dt=inputs.dt,
        duration_ms=duration,
    )


def _equation_variables(equations: list[str]) -> list[str]:
    names: list[str] = []
    for text in equations:
        match = _EQUATION_PATTERN.match(text.strip())
        if match is None:
            raise ExperimentRejected(
                field="equations",
                reason=f"cannot parse {text.strip()!r}; expected 'd<var>/dt = <expr>'",
            )
        names.append(match.group(1))
    return names


def resolve_ode_experiment(
    request: Mapping[str, Any], *, max_steps: int = MAX_STEPS
) -> ExperimentSpec:
    """Resolve an equation-playground request into its effective experiment.

    Raises
    ------
    ExperimentRejected
        Unparsable equation, initial value for an undeclared variable, seed
        on noise-free equations, no complete step or an oversized run.
    """
    equations = list(request.get("equations") or [])
    threshold = request.get("threshold") or None
    reset = request.get("reset") or None
    params = dict(request.get("params") or {})
    init = dict(request.get("init") or {})
    dt = float(request.get("dt", STUDIO_DEFAULT_DT_MS))
    duration = float(request.get("duration", 100.0))
    current = float(request.get("current", 0.0))
    protocol = request.get("protocol", "constant")
    frequency_hz = float(request.get("frequency_hz", 10.0))
    seed = request.get("seed")
    trial: Trial = request.get("trial", "replay")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ExperimentRejected(field="protocol", reason=f"unsupported protocol {protocol!r}")
    if not equations:
        raise ExperimentRejected(field="equations", reason="at least one equation is required")
    variables = _equation_variables(equations)
    unknown = sorted(set(init) - set(variables))
    if unknown:
        raise ExperimentRejected(
            field="init", reason=f"initial values for undeclared variables: {', '.join(unknown)}"
        )
    initial_state = {name: float(init.get(name, 0.0)) for name in variables}
    expressions = list(equations) + [text for text in (threshold, reset) if text]
    stochastic = any(_NOISE_PATTERN.search(text) is not None for text in expressions)
    if seed is not None and not stochastic:
        raise ExperimentRejected(
            field="seed",
            reason="the equations reference no diffusion noise (xi); a seed is not accepted",
        )
    if stochastic:
        if trial == "fresh":
            effective_seed: int | None = _draw_seed()
            seed_source: SeedSource = "drawn"
        elif seed is not None:
            effective_seed = int(seed)
            seed_source = "request"
        else:
            effective_seed = DEFAULT_NOISE_SEED
            seed_source = "playground-default"
        randomness: dict[str, Any] = {
            "kind": "diffusion-noise",
            "seed": effective_seed,
            "seed_source": seed_source,
            "trial": trial,
            "effective_trial": "fresh" if trial == "fresh" else "replay",
            "generator": "numpy.random.default_rng",
        }
    else:
        effective_seed = None
        randomness = {
            "kind": "none",
            "seed": None,
            "seed_source": "none",
            "trial": trial,
            "effective_trial": "replay",
            "generator": None,
            "note": "deterministic equations: a fresh trial equals a replay",
        }
    n_steps = _steps(duration, dt, max_steps=max_steps)
    samples = _make_current_trace(protocol, current, n_steps, dt=dt, frequency_hz=frequency_hz)
    body: dict[str, Any] = {
        "equations": {
            "equations": equations,
            "threshold": threshold,
            "reset": reset,
            "variables": variables,
            "equation_sha256": _sha256_json(
                {"equations": equations, "threshold": threshold, "reset": reset}
            ),
        },
        "numerical": {
            "method": "euler",
            "family": "ode",
            "dt": dt,
            "dt_source": "request" if "dt" in request else "studio_default",
            "substeps": 1,
            "time_unit": "ms",
        },
        "steps": {
            "n_steps": n_steps,
            "duration_requested_ms": duration,
            "duration_effective_ms": n_steps * dt,
            "synchronous_limit": max_steps,
        },
        "parameters": params,
        "initial_state": initial_state,
        "initial_state_source": "request init; undeclared variables start at 0.0",
        "protocol": _protocol_block(
            protocol=protocol, current=current, frequency_hz=frequency_hz, samples=samples
        ),
        "randomness": randomness,
        "backend": {"selected": "python", "rejected": []},
        "runtime": _runtime_block(),
    }
    run_kwargs = {
        "equations": equations,
        "threshold": threshold,
        "reset": reset,
        "params": params or None,
        "init": initial_state,
        "dt": dt,
        "duration": duration,
        "current": current,
        "protocol": protocol,
        "frequency_hz": frequency_hz,
        "seed": effective_seed,
        "max_steps": max_steps,
    }
    return _finish(
        source="ode",
        body=body,
        run_kwargs=run_kwargs,
        cacheable=randomness["effective_trial"] == "replay",
        n_steps=n_steps,
        dt=dt,
        duration_ms=duration,
    )


def resolve_experiment(request: Mapping[str, Any], *, max_steps: int = MAX_STEPS) -> ExperimentSpec:
    """Resolve a model or equation request into its effective experiment."""
    if "equations" in request:
        return resolve_ode_experiment(request, max_steps=max_steps)
    return resolve_model_experiment(request, max_steps=max_steps)


def run_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    """Execute a resolved experiment and attach its specification to the result.

    Raises
    ------
    ModelInputError, ModelSimulationFailure, ValueError
        From the run entrypoints.
    """
    from sc_neurocore.studio.model_simulate import simulate_model
    from sc_neurocore.studio.simulation import simulate

    if spec.source == "model":
        result = simulate_model(**spec.run_kwargs)
    else:
        result = simulate(**spec.run_kwargs)
    result["experiment"] = spec.to_public_dict()
    return result


__all__ = [
    "DEFAULT_NOISE_SEED",
    "EXPERIMENT_SCHEMA_VERSION",
    "JOB_MAX_STEPS",
    "SIMULATION_JOB_ROUTE",
    "ExperimentRejected",
    "ExperimentSpec",
    "resolve_experiment",
    "resolve_model_experiment",
    "resolve_ode_experiment",
    "run_experiment",
]
