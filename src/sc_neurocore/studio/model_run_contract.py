# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model-run input contract

"""Fail-closed resolution of one Studio catalogue model run into effective inputs.

The contract is derived from the model class itself: the numeric constructor
fields of a dataclass model (or the numeric keyword defaults of a plain class)
and the signature of its ``step`` method. Every rejection names the exact field
and reason; nothing is dropped, rounded, retried or replaced by a default.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import numbers
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.model_introspection import _fixed_step_attribute, _load_class
from sc_neurocore.studio.simulation import _make_current_trace

DIAGNOSTIC_LIMIT = 200
STUDIO_DEFAULT_DT_MS = 0.1
SUPPORTED_PROTOCOLS: tuple[str, ...] = ("constant", "step", "ramp", "pulse", "sine")
RECEIPT_SCHEMA_VERSION = "studio.model-run-inputs.v1"

NumericKind = Literal["float", "int"]
DtSource = Literal["override", "model_default", "model_attribute", "studio_default"]
Backend = Literal["python", "rust"]


class ModelInputError(ValueError):
    """Raised when a Studio model-run request is rejected before any simulation step.

    Parameters
    ----------
    model : str or None
        Catalogue identity the request named, or ``None`` when the name itself
        was not a string.
    field : str
        Dotted request field that failed (``name``, ``params.tau_m``, ``dt``,
        ``constructor``, ``step``, ``protocol``, ``current``, ``duration``).
    reason : str
        Bounded human-readable reason without repository paths.
    """

    def __init__(self, *, model: str | None, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.model = model
        self.field = field
        self.reason = reason

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "invalid_model_input",
            "model": self.model,
            "field": self.field,
            "reason": self.reason,
        }


class ModelSimulationFailure(RuntimeError):
    """Raised when a validated model run fails numerically at a specific step.

    Parameters
    ----------
    model : str
        Catalogue identity of the failed run.
    backend : {"python", "rust"}
        Backend that executed the failing step.
    step : int
        Zero-based step index at which the failure was detected.
    time_ms : float
        Simulated time of that step in milliseconds.
    diagnostic : str
        Bounded description of the failure (exception class and message, or
        the non-finite state variable).
    """

    def __init__(
        self,
        *,
        model: str,
        backend: Backend,
        step: int,
        time_ms: float,
        diagnostic: str,
    ) -> None:
        super().__init__(f"{model} ({backend}) failed at step {step}: {diagnostic}")
        self.model = model
        self.backend = backend
        self.step = step
        self.time_ms = time_ms
        self.diagnostic = diagnostic

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "model_simulation_failed",
            "model": self.model,
            "backend": self.backend,
            "step": self.step,
            "time_ms": self.time_ms,
            "diagnostic": self.diagnostic,
        }


def bounded_diagnostic(exc: BaseException) -> str:
    """Return ``ClassName: message`` truncated to :data:`DIAGNOSTIC_LIMIT` characters."""
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= DIAGNOSTIC_LIMIT:
        return text
    return text[: DIAGNOSTIC_LIMIT - 3] + "..."


@dataclass(frozen=True, slots=True)
class ParameterContract:
    """One numerically overridable constructor field of a model class.

    Attributes
    ----------
    name : str
        Constructor keyword.
    kind : {"float", "int"}
        Numeric kind the model declares; ``int`` fields reject fractional values.
    default : float or int or None
        Declared default, or ``None`` when the field defaults to ``None``.
    """

    name: str
    kind: NumericKind
    default: float | int | None


@dataclass(frozen=True, slots=True)
class ModelParameterContracts:
    """Overridable numeric fields and the reasons other fields are not inputs."""

    overridable: dict[str, ParameterContract]
    unsupported: dict[str, str]


@dataclass(frozen=True, slots=True)
class DriveContract:
    """How the Studio current protocol is delivered to ``step``.

    Attributes
    ----------
    parameter : str
        Name of the first ``step`` parameter after ``self``.
    kind : {"float", "int"}
        Declared type of that parameter; ``int`` models require integral samples.
    positional_only : bool
        Whether the parameter must be passed positionally.
    """

    parameter: str
    kind: NumericKind
    positional_only: bool


@dataclass(frozen=True, slots=True)
class ModelRunInputs:
    """Validated effective inputs of one model run before any step executes."""

    model: str
    cls: type
    contracts: ModelParameterContracts
    constructor_kwargs: dict[str, float | int]
    overrides_applied: tuple[str, ...]
    dt: float
    dt_source: DtSource
    drive: DriveContract

    def effective_parameters(self) -> dict[str, float | int | None]:
        """Return every overridable field with the value the run will use."""
        return {
            name: self.constructor_kwargs.get(name, contract.default)
            for name, contract in self.contracts.overridable.items()
        }

    def instantiate(self) -> Any:
        """Construct the model with the validated keywords; never retry or substitute."""
        try:
            return self.cls(**self.constructor_kwargs)
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ModelInputError(
                model=self.model, field="constructor", reason=bounded_diagnostic(exc)
            ) from exc


@dataclass(frozen=True, slots=True)
class DriveTrace:
    """Validated current-injection protocol resolved against the run inputs."""

    protocol: str
    current: float
    frequency_hz: float
    duration_ms: float
    n_steps: int
    steps_truncated: bool
    samples: np.ndarray[Any, Any]


def _type_hints(target: object) -> dict[str, Any]:
    try:
        return typing.get_type_hints(target)
    except Exception:
        return {}


def _unwrap_optional(annotation: object) -> object:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return members[0] if len(members) == 1 else object
    return annotation


def _type_label(annotation: object) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _numeric_kind(annotation: object, default: object) -> NumericKind | None:
    resolved = _unwrap_optional(annotation)
    if resolved is bool:
        return None
    if resolved is int:
        return "int"
    if resolved is float:
        return "float"
    if isinstance(default, bool):
        return None
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    return None


def _contract_default(default: object) -> float | int | None:
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return None
    return default


def model_parameter_contracts(cls: type) -> ModelParameterContracts:
    """Inventory the numerically overridable constructor fields of ``cls``.

    Parameters
    ----------
    cls : type
        Catalogue model class (dataclass or plain class with keyword defaults).

    Returns
    -------
    ModelParameterContracts
        Overridable fields keyed by name plus the reason each other constructor
        field is not an input (private state, derived ``init=False`` field,
        factory-initialised field, non-numeric type).
    """
    overridable: dict[str, ParameterContract] = {}
    unsupported: dict[str, str] = {}
    if dataclasses.is_dataclass(cls):
        hints = _type_hints(cls)
        for field in dataclasses.fields(cls):
            annotation = hints.get(field.name, field.type)
            if not field.init:
                unsupported[field.name] = "derived field (init=False) is computed by the model"
                continue
            if field.name.startswith("_"):
                unsupported[field.name] = "private model state is not an input"
                continue
            if field.default_factory is not dataclasses.MISSING:
                unsupported[field.name] = "factory-initialised field is not a scalar parameter"
                continue
            default = None if field.default is dataclasses.MISSING else field.default
            kind = _numeric_kind(annotation, default)
            if kind is None:
                unsupported[field.name] = f"non-numeric field ({_type_label(annotation)})"
                continue
            overridable[field.name] = ParameterContract(
                name=field.name, kind=kind, default=_contract_default(default)
            )
        return ModelParameterContracts(overridable=overridable, unsupported=unsupported)

    hints = _type_hints(inspect.getattr_static(cls, "__init__"))
    for name, parameter in inspect.signature(cls).parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, parameter.annotation)
        if name.startswith("_"):
            unsupported[name] = "private model state is not an input"
            continue
        default = None if parameter.default is inspect.Parameter.empty else parameter.default
        kind = _numeric_kind(annotation, default)
        if kind is None:
            unsupported[name] = f"non-numeric field ({_type_label(annotation)})"
            continue
        overridable[name] = ParameterContract(
            name=name, kind=kind, default=_contract_default(default)
        )
    return ModelParameterContracts(overridable=overridable, unsupported=unsupported)


def model_drive_contract(model: str, cls: type) -> DriveContract:
    """Derive how the current protocol enters ``cls.step``; reject unsatisfiable steps.

    Raises
    ------
    ModelInputError
        When ``step`` takes no drive input or requires further inputs without
        defaults that the Studio current protocol cannot supply.
    """
    step = getattr(cls, "step", None)
    if step is None:
        raise ModelInputError(model=model, field="step", reason="model has no step() method")
    try:
        signature = inspect.signature(step)
    except (TypeError, ValueError) as exc:
        raise ModelInputError(
            model=model,
            field="step",
            reason=f"step signature unavailable: {bounded_diagnostic(exc)}",
        ) from exc
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    if not parameters or parameters[0].kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ):
        raise ModelInputError(model=model, field="step", reason="step() takes no drive input")
    drive = parameters[0]
    missing = [
        parameter.name
        for parameter in parameters[1:]
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if missing:
        raise ModelInputError(
            model=model,
            field="step",
            reason=(
                "step() requires inputs the Studio current protocol cannot supply: "
                + ", ".join(missing)
            ),
        )
    annotation = _type_hints(step).get(drive.name, drive.annotation)
    kind: NumericKind = "int" if _unwrap_optional(annotation) is int else "float"
    return DriveContract(
        parameter=drive.name,
        kind=kind,
        positional_only=drive.kind is inspect.Parameter.POSITIONAL_ONLY,
    )


def _finite_number(model: str, field: str, raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
        raise ModelInputError(model=model, field=field, reason="must be a finite number")
    value = float(raw)
    if not math.isfinite(value):
        raise ModelInputError(model=model, field=field, reason="must be a finite number")
    return value


def _positive_finite(model: str, field: str, raw: object) -> float:
    value = _finite_number(model, field, raw)
    if value <= 0.0:
        raise ModelInputError(model=model, field=field, reason="must be a positive finite number")
    return value


def _coerce(model: str, contract: ParameterContract, raw: object) -> float | int:
    field = f"params.{contract.name}"
    value = _finite_number(model, field, raw)
    if contract.kind == "int":
        if not value.is_integer():
            raise ModelInputError(
                model=model,
                field=field,
                reason=f"integer parameter received fractional value {value!r}",
            )
        return int(value)
    return value


def _validated_overrides(
    model: str,
    param_overrides: Mapping[str, object] | None,
    contracts: ModelParameterContracts,
) -> dict[str, float | int]:
    if param_overrides is None:
        return {}
    if not isinstance(param_overrides, Mapping):
        raise ModelInputError(
            model=model,
            field="params",
            reason="parameter overrides must be a mapping of parameter name to number",
        )
    result: dict[str, float | int] = {}
    for key, raw in param_overrides.items():
        if not isinstance(key, str):
            raise ModelInputError(
                model=model, field="params", reason="parameter names must be strings"
            )
        if key == "dt":
            raise ModelInputError(
                model=model,
                field="params.dt",
                reason="the timestep is set through the dt field, not a parameter override",
            )
        contract = contracts.overridable.get(key)
        if contract is None:
            unsupported = contracts.unsupported.get(key)
            reason = f"not overridable: {unsupported}" if unsupported else "unknown parameter"
            raise ModelInputError(model=model, field=f"params.{key}", reason=reason)
        result[key] = _coerce(model, contract, raw)
    return result


def resolve_model_run_inputs(
    name: object,
    param_overrides: Mapping[str, object] | None,
    dt: object,
) -> ModelRunInputs:
    """Validate a model-run request against the model's own constructor contract.

    Parameters
    ----------
    name : object
        Catalogue identity; must be a registered class name.
    param_overrides : Mapping[str, object] or None
        Constructor overrides. Every key must be an overridable numeric field;
        every value a finite number of the declared kind.
    dt : object
        Explicit timestep in milliseconds, or ``None`` for the model default.
        A model whose step is a fixed class attribute accepts only that value;
        a model without any timestep accepts only the Studio default.

    Returns
    -------
    ModelRunInputs
        Effective inputs ready for construction. No model is constructed here.

    Raises
    ------
    ModelInputError
        On any unknown, mistyped, non-finite, fractional-integer or
        unsupported input, naming the field and reason.
    """
    if not isinstance(name, str) or name not in _CLASS_TO_MODULE:
        raise ModelInputError(
            model=name if isinstance(name, str) else None,
            field="name",
            reason="unknown model",
        )
    cls = _load_class(name)
    contracts = model_parameter_contracts(cls)
    drive = model_drive_contract(name, cls)
    kwargs = _validated_overrides(name, param_overrides, contracts)
    overrides_applied = tuple(sorted(kwargs))
    dt_contract = contracts.overridable.get("dt")
    fixed_step = _fixed_step_attribute(cls) if dt_contract is None else None
    dt_source: DtSource
    if dt is not None:
        effective_dt = _positive_finite(name, "dt", dt)
        if dt_contract is not None:
            kwargs["dt"] = _coerce(name, dt_contract, effective_dt)
            dt_source = "override"
        elif fixed_step is not None:
            if effective_dt != fixed_step:
                raise ModelInputError(
                    model=name,
                    field="dt",
                    reason=(f"model has a fixed step of {fixed_step} ms that cannot be overridden"),
                )
            dt_source = "model_attribute"
        else:
            if effective_dt != STUDIO_DEFAULT_DT_MS:
                raise ModelInputError(
                    model=name,
                    field="dt",
                    reason=(
                        "model declares no integration timestep; only the Studio "
                        f"default {STUDIO_DEFAULT_DT_MS} ms per step is accepted"
                    ),
                )
            dt_source = "studio_default"
    elif dt_contract is not None:
        if dt_contract.default is None:
            raise ModelInputError(
                model=name,
                field="dt",
                reason="model declares dt without a default; pass dt explicitly",
            )
        effective_dt = _positive_finite(name, "dt", dt_contract.default)
        dt_source = "model_default"
    elif fixed_step is not None:
        effective_dt = fixed_step
        dt_source = "model_attribute"
    else:
        effective_dt = STUDIO_DEFAULT_DT_MS
        dt_source = "studio_default"
    return ModelRunInputs(
        model=name,
        cls=cls,
        contracts=contracts,
        constructor_kwargs=kwargs,
        overrides_applied=overrides_applied,
        dt=effective_dt,
        dt_source=dt_source,
        drive=drive,
    )


def resolve_drive_trace(
    inputs: ModelRunInputs,
    *,
    protocol: object,
    current: object,
    duration: object,
    frequency_hz: object,
    max_steps: int,
) -> DriveTrace:
    """Validate the injection protocol and build the sample trace for the run.

    Raises
    ------
    ModelInputError
        On an unsupported protocol, non-finite current, non-positive duration or
        frequency, a duration shorter than one step, or fractional samples for
        a model whose ``step`` declares an integer drive.
    """
    model = inputs.model
    if not isinstance(protocol, str) or protocol not in SUPPORTED_PROTOCOLS:
        raise ModelInputError(
            model=model,
            field="protocol",
            reason=f"unsupported protocol {protocol!r}; expected one of {SUPPORTED_PROTOCOLS}",
        )
    current_value = _finite_number(model, "current", current)
    duration_value = _positive_finite(model, "duration", duration)
    frequency_value = _positive_finite(model, "frequency_hz", frequency_hz)
    requested_steps = int(duration_value / inputs.dt)
    if requested_steps < 1:
        raise ModelInputError(
            model=model,
            field="duration",
            reason=f"duration {duration_value} ms with dt {inputs.dt} ms yields no complete step",
        )
    n_steps = min(requested_steps, max_steps)
    samples = _make_current_trace(
        protocol, current_value, n_steps, dt=inputs.dt, frequency_hz=frequency_value
    )
    if inputs.drive.kind == "int" and not bool(np.all(samples == np.round(samples))):
        raise ModelInputError(
            model=model,
            field="current",
            reason=(
                f"integer-drive model requires integral current samples; protocol "
                f"{protocol!r} with current {current_value} produces fractional samples"
            ),
        )
    return DriveTrace(
        protocol=protocol,
        current=current_value,
        frequency_hz=frequency_value,
        duration_ms=duration_value,
        n_steps=n_steps,
        steps_truncated=requested_steps > n_steps,
        samples=samples,
    )


def run_receipt(
    inputs: ModelRunInputs,
    trace: DriveTrace,
    *,
    backend: Backend,
    recorded_state: tuple[str, ...],
    excluded_state: tuple[tuple[str, str], ...],
    plot_stride: int,
) -> dict[str, Any]:
    """Return the effective-input receipt attached to a successful run payload."""
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "model": inputs.model,
        "backend": backend,
        "dt": inputs.dt,
        "dt_source": inputs.dt_source,
        "parameters": inputs.effective_parameters(),
        "overrides_applied": list(inputs.overrides_applied),
        "drive": {"step_parameter": inputs.drive.parameter, "kind": inputs.drive.kind},
        "protocol": trace.protocol,
        "current": trace.current,
        "frequency_hz": trace.frequency_hz,
        "duration_requested_ms": trace.duration_ms,
        "n_steps": trace.n_steps,
        "steps_truncated": trace.steps_truncated,
        "state_recording": {
            "recorded": list(recorded_state),
            "excluded": [{"name": name, "reason": reason} for name, reason in excluded_state],
        },
        "plot_stride": plot_stride,
    }


__all__ = [
    "DIAGNOSTIC_LIMIT",
    "RECEIPT_SCHEMA_VERSION",
    "STUDIO_DEFAULT_DT_MS",
    "SUPPORTED_PROTOCOLS",
    "DriveContract",
    "DriveTrace",
    "ModelInputError",
    "ModelParameterContracts",
    "ModelRunInputs",
    "ModelSimulationFailure",
    "ParameterContract",
    "bounded_diagnostic",
    "model_drive_contract",
    "model_parameter_contracts",
    "resolve_drive_trace",
    "resolve_model_run_inputs",
    "run_receipt",
]
