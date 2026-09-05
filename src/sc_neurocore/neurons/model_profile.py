# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Versioned model profile: scientific model, numerical realisation, lowering

"""Versioned model profile contract resolved from a schema-DSL document.

A bundled schema mixes three things that must be judged separately:

* the *scientific model*: the authored equations, the biological state they
  evolve, the parameters the source defines and the event contract;
* the *numerical realisation*: the integration method and its exactness class,
  the timestep and its unit, sub-steps and what they mean, the order in which
  integration, threshold, reset and refractory logic are evaluated, the
  registers that exist only to realise the scheme, the parameters that are
  implementation choices, and the randomness contract;
* the *lowering profile*: what the RTL emitter can lower from that realisation
  and under which limits.

:func:`resolve_profile` derives all three from a schema under contract
:data:`PROFILE_CONTRACT`. A schema may author an optional ``[profile]`` section
to state what cannot be derived (which state variables are auxiliary registers,
which parameters are implementation or timebase choices, the time unit, units,
the meaning of sub-steps, an exactness claim). Everything else takes a derived
default, and every contradiction between the authored profile and the schema
is reported as a problem so an executable consumer can refuse the schema.

The contract never changes a number: it classifies what a schema already
states. An SC compatibility profile keeps its own equations and defaults; the
profile only records that its numerics are a project realisation.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from sc_neurocore.neurons._stochastic_threshold import DEFAULT_LFSR16_SEED
from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS
from sc_neurocore.neurons.schema_contracts import stateless_event_kind

PROFILE_CONTRACT = "sc-neurocore.model-profile.v1"
PROFILE_SECTION = "profile"

RealisationKind = Literal["executable", "descriptive-record"]
RealisationFamily = Literal["ode", "map", "event-only"]
StateRole = Literal["biological", "auxiliary"]
ParameterRole = Literal["source", "implementation", "timebase"]
SubstepKind = Literal["time-subdivision", "stage-iteration", "none"]
Exactness = Literal[
    "first-order",
    "first-order-sequential",
    "fourth-order",
    "linearised-exponential",
    "exact-linear-relaxation",
    "recurrence",
    "exact-flow",
    "event-only",
    "not-executable",
]
RandomnessKind = Literal[
    "none",
    "lfsr16-threshold",
    "diffusion-noise-global-rng",
    "lfsr16-threshold+diffusion-noise-global-rng",
]

EXECUTABLE_METHODS: tuple[str, ...] = tuple(SUPPORTED_METHODS)
EXECUTABLE_DETECTIONS: tuple[str, ...] = ("level", "crossing", "escape_rate", "poisson")
_ODE_METHODS: frozenset[str] = frozenset({"euler", "rk4", "exp_euler", "gauss_seidel"})
_DERIVED_EXACTNESS: dict[str, Exactness] = {
    "euler": "first-order",
    "gauss_seidel": "first-order-sequential",
    "rk4": "fourth-order",
    "exp_euler": "linearised-exponential",
    "map": "recurrence",
}
# Exactness a schema may claim beyond the derived class, per method. A claim is a
# statement about the mathematics of the update, so it is admitted only where the
# resolver can check it (exp_euler) or where the recurrence is authored as a sampled
# closed-form flow (map); an explicit-Euler or RK4 step can never be an exact flow.
_CLAIMABLE_EXACTNESS: dict[str, frozenset[str]] = {
    "exp_euler": frozenset({"exact-linear-relaxation"}),
    "map": frozenset({"exact-flow"}),
}
_TIMEBASE_NAMES: frozenset[str] = frozenset({"dt", "dt_ms", "timestep"})
_DEFAULT_TIMEBASE_UNIT: dict[str, str] = {"dt_ms": "ms"}
_PROFILE_KEYS: frozenset[str] = frozenset(
    {
        "contract",
        "time_unit",
        "substep_kind",
        "exactness",
        "admissible_methods",
        "refractory_register",
        "notes",
        "state",
        "parameters",
        "units",
    }
)

# Mapping table (method -> exactness -> admissible family -> lowering), published on
# the model-profiles page and in the generated ledger so a reader can audit why a
# realisation holds its class.
METHOD_TABLE: tuple[dict[str, str], ...] = (
    {
        "method": "euler",
        "family": "ode",
        "exactness": "first-order",
        "claimable": "",
        "lowering": "registered next-state datapath; sub-steps only for crossing, non-resetting models",
    },
    {
        "method": "gauss_seidel",
        "family": "ode",
        "exactness": "first-order-sequential",
        "claimable": "",
        "lowering": "sequential derivative wires in declaration order; same sub-step limit as euler",
    },
    {
        "method": "rk4",
        "family": "ode",
        "exactness": "fourth-order",
        "claimable": "",
        "lowering": "four staged derivative evaluations per clock; same sub-step limit as euler",
    },
    {
        "method": "exp_euler",
        "family": "ode",
        "exactness": "linearised-exponential",
        "claimable": "exact-linear-relaxation (verified: affine in its own variable, no other state)",
        "lowering": "exprel-scaled increment from the symbolic Jacobian the golden compiles",
    },
    {
        "method": "map",
        "family": "map",
        "exactness": "recurrence",
        "claimable": "exact-flow (authored: the recurrence is a sampled closed-form solution)",
        "lowering": "next state is the map value itself; stage iteration folds several clocks per macro step",
    },
)


class ModelProfileError(ValueError):
    """Raised when a schema's profile contradicts the schema or the contract."""


class ProfileAdmissionError(ModelProfileError):
    """Raised when an override or protocol is not admissible under a profile."""


@dataclass(frozen=True, slots=True)
class StateVariable:
    """One state variable and its role in the profile.

    Parameters
    ----------
    name:
        Variable name as declared in ``[state]``.
    role:
        ``biological`` (a quantity of the scientific model) or ``auxiliary`` (a
        deterministic register that exists only to realise the numerical
        scheme or the event logic).
    init:
        Initial value from the schema.
    unit:
        Declared unit, empty when not declared.
    meaning:
        Authored meaning for auxiliary registers, empty otherwise.
    """

    name: str
    role: StateRole
    init: float
    unit: str = ""
    meaning: str = ""

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "name": self.name,
            "role": self.role,
            "init": self.init,
            "unit": self.unit,
            "meaning": self.meaning,
        }


@dataclass(frozen=True, slots=True)
class Parameter:
    """One schema parameter and its role in the profile.

    Parameters
    ----------
    name:
        Parameter name as declared in ``[parameters]``.
    role:
        ``source`` (defined by the scientific source), ``implementation`` (a
        maintained choice such as an observation threshold) or ``timebase``
        (bound to ``integration.dt`` and read by the expressions).
    default:
        Default value from the schema.
    unit:
        Declared unit, empty when not declared.
    meaning:
        Authored meaning, empty when not declared.
    """

    name: str
    role: ParameterRole
    default: float
    unit: str = ""
    meaning: str = ""

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "name": self.name,
            "role": self.role,
            "default": self.default,
            "unit": self.unit,
            "meaning": self.meaning,
        }


@dataclass(frozen=True, slots=True)
class EventContract:
    """How the realisation turns state into events.

    Parameters
    ----------
    detection:
        Threshold detection mode from the schema (``level``, ``crossing``,
        ``escape_rate``, ``poisson``) or the schema's free text for a
        descriptive record.
    condition:
        Threshold condition expression, empty when none.
    stochastic_expression:
        Escape-rate or Poisson probability expression, empty when none.
    reset:
        Reset rules per state variable.
    edge_detection:
        Whether the runtime engages rising-edge logic (``crossing`` with no
        reset rule); a resetting model uses the level path under either mode.
    refractory_register:
        State variable that encodes the refractory hold inside the dynamics,
        empty when the model has none.
    """

    detection: str
    condition: str
    stochastic_expression: str
    reset: Mapping[str, str]
    edge_detection: bool
    refractory_register: str = ""

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "detection": self.detection,
            "condition": self.condition,
            "stochastic_expression": self.stochastic_expression,
            "reset": dict(self.reset),
            "edge_detection": self.edge_detection,
            "refractory_register": self.refractory_register,
        }


@dataclass(frozen=True, slots=True)
class RandomnessContract:
    """Which random draws a realisation makes and whether they replay.

    Parameters
    ----------
    kind:
        ``none``; ``lfsr16-threshold`` (a model-scoped 16-bit LFSR decides
        stochastic threshold trials from ``seed``); ``diffusion-noise-global-rng``
        (an expression names ``xi``, drawn from NumPy's process-global stream);
        or both.
    seed:
        Initial LFSR seed when the threshold is stochastic, else ``None``.
    reproducible:
        ``True`` only when every draw comes from the model-scoped seeded LFSR.
    """

    kind: RandomnessKind
    seed: int | None
    reproducible: bool

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {"kind": self.kind, "seed": self.seed, "reproducible": self.reproducible}


@dataclass(frozen=True, slots=True)
class ScientificModel:
    """The authored scientific content of a schema.

    Parameters
    ----------
    name:
        Model name from the schema metadata.
    author, year, doi:
        Source locator fields from the metadata (empty or ``None`` when absent).
    equations:
        Authored right-hand sides (or map updates) per state variable.
    biological_state:
        State variables that belong to the scientific model.
    source_parameters:
        Parameters the source defines.
    published_equations:
        ``science.equations_as_published`` when authored, else empty.
    """

    name: str
    author: str
    year: int | None
    doi: str
    equations: Mapping[str, str]
    biological_state: tuple[StateVariable, ...]
    source_parameters: tuple[Parameter, ...]
    published_equations: str = ""

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "name": self.name,
            "author": self.author,
            "year": self.year,
            "doi": self.doi,
            "equations": dict(self.equations),
            "biological_state": [variable.to_public_dict() for variable in self.biological_state],
            "source_parameters": [
                parameter.to_public_dict() for parameter in self.source_parameters
            ],
            "published_equations": self.published_equations,
        }


@dataclass(frozen=True, slots=True)
class NumericalRealisation:
    """How the scientific model is advanced in time.

    Parameters
    ----------
    method:
        Integration method (``euler``, ``map``, ``rk4``, ``exp_euler``,
        ``gauss_seidel``) or the schema's free label for a descriptive record.
    family:
        ``ode`` (the equations are derivatives), ``map`` (the equations are the
        next state) or ``event-only`` (no state at all).
    exactness:
        Exactness class of the update; see :data:`METHOD_TABLE`.
    exactness_claimed:
        ``True`` when the exactness came from an authored claim that the
        resolver admitted, ``False`` when it is the derived class.
    dt:
        Integration step from the schema.
    time_unit:
        Unit of ``dt`` (``ms`` for the conductance and IF corpus; ``iteration``
        for a recurrence without a continuous timebase; empty when unstated).
    substeps:
        Inner steps per public step.
    substep_kind:
        ``time-subdivision`` (each sub-step advances ``dt``; the public step is
        ``substeps * dt``), ``stage-iteration`` (the sub-steps are stages of one
        scheme folded into a map; the public step is one ``dt``) or ``none``.
    macro_step:
        Duration of one public ``step()`` in ``time_unit``; ``None`` for a
        descriptive record, whose sub-step convention is not the runtime's.
    evaluation_order:
        Phases of one public step in execution order.
    auxiliary_registers:
        State variables that exist only to realise the scheme or the events.
    implementation_parameters:
        Parameters that are maintained implementation choices.
    timebase_parameters:
        Parameters bound to ``dt`` and read by the expressions.
    admissible_methods:
        Methods a consumer may select for this scientific model without leaving
        its family; the declared method is always first.
    randomness:
        Randomness contract.
    event:
        Event contract.
    """

    method: str
    family: RealisationFamily
    exactness: Exactness
    exactness_claimed: bool
    dt: float
    time_unit: str
    substeps: int
    substep_kind: SubstepKind
    macro_step: float | None
    evaluation_order: tuple[str, ...]
    auxiliary_registers: tuple[StateVariable, ...]
    implementation_parameters: tuple[Parameter, ...]
    timebase_parameters: tuple[Parameter, ...]
    admissible_methods: tuple[str, ...]
    randomness: RandomnessContract
    event: EventContract

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "method": self.method,
            "family": self.family,
            "exactness": self.exactness,
            "exactness_claimed": self.exactness_claimed,
            "dt": self.dt,
            "time_unit": self.time_unit,
            "substeps": self.substeps,
            "substep_kind": self.substep_kind,
            "macro_step": self.macro_step,
            "evaluation_order": list(self.evaluation_order),
            "auxiliary_registers": [
                register.to_public_dict() for register in self.auxiliary_registers
            ],
            "implementation_parameters": [
                parameter.to_public_dict() for parameter in self.implementation_parameters
            ],
            "timebase_parameters": [
                parameter.to_public_dict() for parameter in self.timebase_parameters
            ],
            "admissible_methods": list(self.admissible_methods),
            "randomness": self.randomness.to_public_dict(),
            "event": self.event.to_public_dict(),
        }


@dataclass(frozen=True, slots=True)
class LoweringProfile:
    """What the RTL emitter can lower from the realisation.

    Parameters
    ----------
    rtl_supported:
        ``True`` when the equation compiler accepts the realisation as declared.
    limits:
        Reasons the realisation cannot be lowered, or the limits it is lowered
        under (pipelining excluded for sub-stepped or stochastic datapaths).
    cosim_methods:
        Admissible methods Studio co-simulates against the golden.
    recommended_precision:
        ``hints.recommended_precision`` when authored, else empty.
    """

    rtl_supported: bool
    limits: tuple[str, ...]
    cosim_methods: tuple[str, ...]
    recommended_precision: str = ""

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {
            "rtl_supported": self.rtl_supported,
            "limits": list(self.limits),
            "cosim_methods": list(self.cosim_methods),
            "recommended_precision": self.recommended_precision,
        }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """The resolved profile of one schema document.

    Parameters
    ----------
    contract:
        Contract identifier (:data:`PROFILE_CONTRACT`).
    stem:
        Schema stem when resolved from a bundled schema, else empty.
    realisation_kind:
        ``executable`` when UniversalNeuron can run the schema, otherwise
        ``descriptive-record`` (the schema records a hand model whose method or
        detection is outside the executable vocabulary).
    authored:
        ``True`` when the schema carries a ``[profile]`` section.
    scientific, numerical, lowering:
        The three separated layers.
    problems:
        Contradictions between the authored profile and the schema, or between
        schema fields. Non-empty means an executable consumer must refuse.
    notes:
        Authored free-text note from the profile section.
    """

    contract: str
    stem: str
    realisation_kind: RealisationKind
    authored: bool
    scientific: ScientificModel
    numerical: NumericalRealisation
    lowering: LoweringProfile
    problems: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_executable(self) -> bool:
        """Whether the schema runs through the equation runtime."""
        return self.realisation_kind == "executable"

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection with a stable field order."""
        return {
            "contract": self.contract,
            "stem": self.stem,
            "realisation_kind": self.realisation_kind,
            "authored": self.authored,
            "scientific": self.scientific.to_public_dict(),
            "numerical": self.numerical.to_public_dict(),
            "lowering": self.lowering.to_public_dict(),
            "problems": list(self.problems),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class AdmittedOverrides:
    """Overrides accepted under a profile, with the timebase propagated.

    Parameters
    ----------
    dt:
        Effective integration step.
    method:
        Effective method.
    parameters:
        Effective parameter overrides, including timebase parameters set to
        ``dt`` when the step was overridden.
    rng_seed:
        Effective LFSR seed, ``None`` when the profile draws no randomness.
    derived:
        ``True`` when the effective realisation differs from the declared one.
    """

    dt: float
    method: str
    parameters: dict[str, float]
    rng_seed: int | None
    derived: bool

    def realisation(self, profile: ModelProfile) -> dict[str, object]:
        """Return the effective numerical realisation as a public record.

        Parameters
        ----------
        profile:
            The profile the overrides were admitted under.

        Returns
        -------
        dict[str, object]
            Contract, declared and effective method, exactness of the effective
            method, effective step, sub-steps and macro step, randomness and
            whether the realisation was derived from an override.
        """
        numerical = profile.numerical
        exactness: str = numerical.exactness
        if self.method != numerical.method:
            exactness = _DERIVED_EXACTNESS.get(self.method, "not-executable")
        substeps = numerical.substeps
        macro = self.dt if numerical.substep_kind != "time-subdivision" else self.dt * substeps
        return {
            "contract": profile.contract,
            "declared_method": numerical.method,
            "method": self.method,
            "exactness": exactness,
            "dt": self.dt,
            "time_unit": numerical.time_unit,
            "substeps": substeps,
            "substep_kind": numerical.substep_kind,
            "macro_step": macro,
            "timebase_parameters": [parameter.name for parameter in numerical.timebase_parameters],
            "randomness": numerical.randomness.kind,
            "rng_seed": self.rng_seed,
            "derived": self.derived,
        }


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def expression_names(expression: str) -> frozenset[str]:
    """Return the identifiers an expression reads, with ``_prev`` aliases folded.

    Parameters
    ----------
    expression:
        A DSL expression (Python expression syntax).

    Returns
    -------
    frozenset[str]
        Identifier names; ``x_prev`` is reported as ``x``. An unparsable
        expression (a descriptive record's prose) yields the empty set.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return frozenset()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            names.add(name[:-5] if name.endswith("_prev") else name)
    return frozenset(names)


def _all_expressions(schema: Mapping[str, Any]) -> list[str]:
    expressions: list[str] = []
    for section in ("dynamics", "reset"):
        for value in _mapping(schema.get(section)).values():
            if isinstance(value, str):
                expressions.append(value)
    threshold = _mapping(schema.get("threshold"))
    for key in ("condition", "rate_expression", "probability_expression"):
        value = threshold.get(key)
        if isinstance(value, str):
            expressions.append(value)
    return expressions


def _role_declarations(section: Mapping[str, Any], key: str) -> dict[str, tuple[str, str]]:
    """Return ``name -> (role, meaning)`` from an authored role table."""
    declared: dict[str, tuple[str, str]] = {}
    for name, raw in _mapping(section.get(key)).items():
        if not isinstance(raw, str):
            continue
        role, _sep, meaning = raw.partition(":")
        declared[str(name)] = (role.strip(), meaning.strip())
    return declared


def _linear_relaxation(equations: Mapping[str, str], state_names: frozenset[str]) -> str:
    """Return why exponential Euler is not exact for ``equations``, empty when it is."""
    try:
        from sc_neurocore.neurons.expression_derivative import (
            ExpressionDifferentiationError,
            differentiate,
        )
    except ImportError:  # pragma: no cover - exercised only without SymPy
        return "the exact-linear-relaxation claim needs the symbolic extra to verify"
    for variable, expression in equations.items():
        others = (expression_names(expression) & state_names) - {variable}
        if others:
            return (
                f"{variable!r} depends on other state {sorted(others)}, so the exponential "
                "Euler step is not the exact flow"
            )
        try:
            second = differentiate(differentiate(expression, variable), variable)
        except ExpressionDifferentiationError as error:
            return f"{variable!r} is not smooth in itself: {error}"
        if second.strip() != "0":
            return f"{variable!r} is not affine in itself (second derivative {second})"
    return ""


def resolve_profile(schema: Mapping[str, Any], *, stem: str = "") -> ModelProfile:
    """Resolve the profile of one schema document.

    Parameters
    ----------
    schema:
        Parsed schema mapping (as returned by
        :func:`~sc_neurocore.neurons.universal_dsl.load_schema`).
    stem:
        Bundled schema stem, when known.

    Returns
    -------
    ModelProfile
        The separated scientific model, numerical realisation and lowering
        profile, with every contradiction listed in ``problems``.
    """
    problems: list[str] = []
    metadata = _mapping(schema.get("metadata"))
    state_section = _mapping(schema.get("state"))
    parameter_section = _mapping(schema.get("parameters"))
    integration = _mapping(schema.get("integration"))
    dynamics = _mapping(schema.get("dynamics"))
    threshold = _mapping(schema.get("threshold"))
    reset = _mapping(schema.get("reset"))
    extensions = _mapping(schema.get("extensions"))
    science = _mapping(schema.get("science"))
    hints = _mapping(schema.get("hints"))
    authored_section = schema.get(PROFILE_SECTION)
    authored = authored_section is not None
    profile_section = _mapping(authored_section)
    if authored and not isinstance(authored_section, Mapping):
        problems.append("profile section must be a table")
        profile_section = {}
    if profile_section:
        contract = profile_section.get("contract", PROFILE_CONTRACT)
        if contract != PROFILE_CONTRACT:
            problems.append(f"profile contract {contract!r} is not {PROFILE_CONTRACT}")
        unknown_keys = sorted(set(profile_section) - _PROFILE_KEYS)
        if unknown_keys:
            problems.append(f"unknown profile keys {unknown_keys}")

    method_raw = integration.get("method", "euler")
    method = str(method_raw)
    detection_raw = threshold.get("detection", "level")
    detection = str(detection_raw)
    dt_value = _number(integration.get("dt", 0.1))
    if dt_value is None or not math.isfinite(dt_value) or dt_value <= 0.0:
        problems.append(
            f"integration.dt must be a positive finite number, got {integration.get('dt')!r}"
        )
        dt_value = float("nan")
    substeps_raw = integration.get("substeps", 1)
    substeps = (
        substeps_raw if isinstance(substeps_raw, int) and not isinstance(substeps_raw, bool) else 0
    )
    if substeps < 1:
        problems.append(f"integration.substeps must be a positive integer, got {substeps_raw!r}")
        substeps = 1

    event_only = stateless_event_kind(schema)
    executable = method in EXECUTABLE_METHODS and detection in EXECUTABLE_DETECTIONS
    record_reasons: list[str] = []
    if method not in EXECUTABLE_METHODS:
        record_reasons.append(f"method {method!r} is outside the executable vocabulary")
    if detection not in EXECUTABLE_DETECTIONS:
        record_reasons.append(f"detection {detection!r} is outside the executable vocabulary")
    if executable and not dynamics and event_only is None:
        record_reasons.append("no dynamics and no supported event-only contract")
        executable = False
    realisation_kind: RealisationKind = "executable" if executable else "descriptive-record"
    if authored and not executable:
        problems.append(
            "a descriptive record cannot author a profile: " + "; ".join(record_reasons)
        )

    # --- state roles -------------------------------------------------------
    declared_state = _role_declarations(profile_section, "state")
    declared_parameters = _role_declarations(profile_section, "parameters")
    units = {
        str(name): str(unit)
        for name, unit in _mapping(profile_section.get("units")).items()
        if isinstance(unit, str)
    }
    for name in sorted(set(declared_state) - set(state_section)):
        problems.append(f"profile.state names undeclared state variable {name!r}")
    for name in sorted(set(declared_parameters) - set(parameter_section)):
        problems.append(f"profile.parameters names undeclared parameter {name!r}")
    for name in sorted(set(units) - set(state_section) - set(parameter_section)):
        problems.append(f"profile.units names unknown quantity {name!r}")
    refractory_register = str(profile_section.get("refractory_register", "") or "")
    if refractory_register and refractory_register not in state_section:
        problems.append(f"refractory_register {refractory_register!r} is not a state variable")

    biological: list[StateVariable] = []
    auxiliary: list[StateVariable] = []
    for name, raw_init in state_section.items():
        init = _number(raw_init)
        if init is None:
            problems.append(f"state variable {name!r} has a non-numeric initial value")
            init = float("nan")
        role, meaning = declared_state.get(name, ("biological", ""))
        if role not in {"biological", "auxiliary"}:
            problems.append(f"state role {role!r} for {name!r} is not biological or auxiliary")
            role = "biological"
        variable = StateVariable(
            name=name,
            role="auxiliary" if role == "auxiliary" else "biological",
            init=init,
            unit=units.get(name, ""),
            meaning=meaning,
        )
        (auxiliary if role == "auxiliary" else biological).append(variable)
    if refractory_register and refractory_register in {v.name for v in biological}:
        problems.append(
            f"refractory_register {refractory_register!r} must be declared an auxiliary register"
        )

    # --- parameter roles ---------------------------------------------------
    time_unit = str(profile_section.get("time_unit", "") or "")
    source_parameters: list[Parameter] = []
    implementation: list[Parameter] = []
    timebase: list[Parameter] = []
    read_names: set[str] = set()
    for expression in _all_expressions(schema):
        read_names |= expression_names(expression)
    for name, raw_default in parameter_section.items():
        default = _number(raw_default)
        if default is None:
            problems.append(f"parameter {name!r} has a non-numeric default")
            default = float("nan")
        if name in declared_parameters:
            role, meaning = declared_parameters[name]
        elif name in _TIMEBASE_NAMES and not math.isnan(dt_value) and default == dt_value:
            role, meaning = "timebase", ""
        else:
            role, meaning = "source", ""
        if role not in {"source", "implementation", "timebase"}:
            problems.append(
                f"parameter role {role!r} for {name!r} is not source, implementation or timebase"
            )
            role = "source"
        unit = units.get(name, "")
        if role == "timebase":
            if not math.isnan(dt_value) and default != dt_value:
                problems.append(
                    f"timebase parameter {name!r} = {default} contradicts integration.dt = {dt_value}"
                )
            unit = unit or _DEFAULT_TIMEBASE_UNIT.get(name, time_unit)
            if time_unit and unit and unit != time_unit:
                problems.append(
                    f"timebase parameter {name!r} is in {unit!r} but the profile time unit is {time_unit!r}"
                )
        parameter = Parameter(
            name=name, role=_parameter_role(role), default=default, unit=unit, meaning=meaning
        )
        if role == "timebase":
            timebase.append(parameter)
        elif role == "implementation":
            implementation.append(parameter)
        else:
            source_parameters.append(parameter)
    for name in sorted(set(parameter_section) & _TIMEBASE_NAMES):
        if name in read_names and name not in {p.name for p in timebase}:
            problems.append(
                f"parameter {name!r} is read by the expressions but is not bound to integration.dt"
            )

    # --- numerical realisation --------------------------------------------
    family: RealisationFamily
    if event_only is not None and not dynamics:
        family = "event-only"
    elif method == "map":
        family = "map"
    else:
        family = "ode"
    state_names = frozenset(state_section)
    exactness: Exactness
    exactness_claimed = False
    if not executable:
        exactness = "not-executable"
    elif family == "event-only":
        exactness = "event-only"
    else:
        exactness = _DERIVED_EXACTNESS.get(method, "not-executable")
    claim = profile_section.get("exactness")
    if claim is not None:
        claim_text = str(claim)
        if claim_text in _CLAIMABLE_EXACTNESS.get(method, frozenset()):
            if method == "exp_euler":
                reason = _linear_relaxation({k: str(v) for k, v in dynamics.items()}, state_names)
                if reason:
                    problems.append(f"exactness claim {claim_text!r} rejected: {reason}")
                else:
                    exactness, exactness_claimed = "exact-linear-relaxation", True
            else:
                exactness, exactness_claimed = "exact-flow", True
        elif claim_text == exactness:
            exactness_claimed = False
        else:
            problems.append(
                f"exactness claim {claim_text!r} is not admissible for method {method!r} "
                f"(derived class {exactness!r})"
            )

    substep_kind: SubstepKind
    declared_kind = profile_section.get("substep_kind")
    if substeps == 1:
        substep_kind = "none"
        if declared_kind not in (None, "none"):
            problems.append(f"substep_kind {declared_kind!r} declared with substeps = 1")
    elif declared_kind is None:
        if family == "map":
            problems.append(
                "a map with substeps > 1 must declare substep_kind (stage-iteration or "
                "time-subdivision); stage registers folded under map are otherwise indistinguishable "
                "from sub-sampled time"
            )
            substep_kind = "stage-iteration"
        else:
            substep_kind = "time-subdivision"
    elif declared_kind in ("time-subdivision", "stage-iteration"):
        substep_kind = (
            "stage-iteration" if declared_kind == "stage-iteration" else "time-subdivision"
        )
        if substep_kind == "stage-iteration" and family != "map":
            problems.append(
                'stage-iteration sub-steps require method = "map" (folded stage registers)'
            )
    else:
        problems.append(
            f"substep_kind {declared_kind!r} is not time-subdivision or stage-iteration"
        )
        substep_kind = "time-subdivision"
    if not time_unit:
        if family == "map" and not timebase and not read_names & _TIMEBASE_NAMES:
            time_unit = "iteration"
    macro_step: float | None
    if not executable:
        macro_step = None
    elif substep_kind == "time-subdivision":
        macro_step = dt_value * substeps
    else:
        macro_step = dt_value

    condition = str(threshold.get("condition", "") or "")
    rate_expression = str(threshold.get("rate_expression", "") or "")
    probability_expression = str(threshold.get("probability_expression", "") or "")
    stochastic_expression = rate_expression or probability_expression
    if detection in {"escape_rate", "poisson"} and condition == "stochastic":
        condition = ""
    edge_detection = detection == "crossing" and bool(condition) and not reset
    evaluation_order = _evaluation_order(
        family=family,
        detection=detection,
        substeps=substeps,
        substep_kind=substep_kind,
        has_condition=bool(condition),
        has_reset=bool(reset),
        edge_detection=edge_detection,
        refractory_register=refractory_register,
    )

    uses_xi = "xi" in read_names
    stochastic = detection in {"escape_rate", "poisson"} and bool(stochastic_expression)
    seed_raw = threshold.get("rng_seed", DEFAULT_LFSR16_SEED)
    seed = seed_raw if isinstance(seed_raw, int) and not isinstance(seed_raw, bool) else None
    if stochastic and seed is None:
        problems.append(f"threshold.rng_seed must be an integer, got {seed_raw!r}")
    randomness_kind: RandomnessKind
    if stochastic and uses_xi:
        randomness_kind = "lfsr16-threshold+diffusion-noise-global-rng"
    elif stochastic:
        randomness_kind = "lfsr16-threshold"
    elif uses_xi:
        randomness_kind = "diffusion-noise-global-rng"
    else:
        randomness_kind = "none"
    randomness = RandomnessContract(
        kind=randomness_kind,
        seed=seed if stochastic else None,
        reproducible=not uses_xi,
    )

    admissible = _admissible_methods(
        method=method,
        family=family,
        executable=executable,
        declared=profile_section.get("admissible_methods"),
        options=extensions.get("integrator_options"),
        problems=problems,
    )

    event = EventContract(
        detection=detection,
        condition=condition,
        stochastic_expression=stochastic_expression,
        reset={str(k): str(v) for k, v in reset.items()},
        edge_detection=edge_detection,
        refractory_register=refractory_register,
    )
    numerical = NumericalRealisation(
        method=method,
        family=family,
        exactness=exactness,
        exactness_claimed=exactness_claimed,
        dt=dt_value,
        time_unit=time_unit,
        substeps=substeps,
        substep_kind=substep_kind,
        macro_step=macro_step,
        evaluation_order=evaluation_order,
        auxiliary_registers=tuple(auxiliary),
        implementation_parameters=tuple(implementation),
        timebase_parameters=tuple(timebase),
        admissible_methods=admissible,
        randomness=randomness,
        event=event,
    )

    year_raw = metadata.get("year")
    scientific = ScientificModel(
        name=str(metadata.get("name", "") or ""),
        author=str(metadata.get("author", "") or ""),
        year=year_raw if isinstance(year_raw, int) and not isinstance(year_raw, bool) else None,
        doi=str(metadata.get("doi", "") or ""),
        equations={str(k): str(v) for k, v in dynamics.items()},
        biological_state=tuple(biological),
        source_parameters=tuple(source_parameters),
        published_equations=str(science.get("equations_as_published", "") or ""),
    )
    lowering = _lowering_profile(
        executable=executable,
        record_reasons=record_reasons,
        substeps=substeps,
        edge_detection=edge_detection,
        stochastic=stochastic,
        admissible=admissible,
        recommended_precision=str(hints.get("recommended_precision", "") or ""),
    )
    return ModelProfile(
        contract=PROFILE_CONTRACT,
        stem=stem,
        realisation_kind=realisation_kind,
        authored=authored,
        scientific=scientific,
        numerical=numerical,
        lowering=lowering,
        problems=tuple(problems),
        notes=str(profile_section.get("notes", "") or ""),
    )


def _parameter_role(role: str) -> ParameterRole:
    if role == "implementation":
        return "implementation"
    if role == "timebase":
        return "timebase"
    return "source"


def _evaluation_order(
    *,
    family: RealisationFamily,
    detection: str,
    substeps: int,
    substep_kind: SubstepKind,
    has_condition: bool,
    has_reset: bool,
    edge_detection: bool,
    refractory_register: str,
) -> tuple[str, ...]:
    """Return the phases of one public step in execution order."""
    phases: list[str] = []
    if family == "event-only":
        if detection == "poisson":
            return ("probability", "lfsr-trial")
        return ("threshold",)
    advance = "iterate" if family == "map" else "integrate"
    if substeps > 1:
        label = "stage" if substep_kind == "stage-iteration" else advance
        phases.append(f"{label} x{substeps}")
    else:
        phases.append(advance)
    if refractory_register:
        phases.append(f"refractory hold inside dynamics ({refractory_register})")
    if detection == "escape_rate":
        phases.extend(("hazard = rate * dt", "lfsr-trial"))
        phases.append("reset" if has_reset else "no reset rule")
    elif detection == "poisson":
        phases.extend(("probability", "lfsr-trial"))
        phases.append("reset" if has_reset else "no reset rule")
    elif has_condition:
        phases.append(
            "rising-edge threshold on the macro boundary" if edge_detection else "level threshold"
        )
        phases.append("reset" if has_reset else "no reset rule")
    else:
        phases.append("no event")
    return tuple(phases)


def _admissible_methods(
    *,
    method: str,
    family: RealisationFamily,
    executable: bool,
    declared: object,
    options: object,
    problems: list[str],
) -> tuple[str, ...]:
    """Return the methods a consumer may select without leaving the family."""
    if not executable:
        return ()
    if family == "map":
        family_methods: tuple[str, ...] = ("map",)
    elif family == "event-only":
        family_methods = (method,)
    else:
        family_methods = tuple(m for m in EXECUTABLE_METHODS if m in _ODE_METHODS)
    if declared is not None:
        if not isinstance(declared, Sequence) or isinstance(declared, str):
            problems.append("profile.admissible_methods must be a list of methods")
            declared_methods: list[str] = []
        else:
            declared_methods = [str(item) for item in declared]
        foreign = [m for m in declared_methods if m not in family_methods]
        if foreign:
            problems.append(
                f"profile.admissible_methods {foreign} leave the {family} family of method {method!r}"
            )
        if declared_methods and method not in declared_methods:
            problems.append(
                f"profile.admissible_methods must include the declared method {method!r}"
            )
        admissible = [method, *[m for m in declared_methods if m in family_methods and m != method]]
        return tuple(dict.fromkeys(admissible))
    if isinstance(options, Sequence) and not isinstance(options, str):
        foreign = [
            str(item)
            for item in options
            if isinstance(item, str) and item in EXECUTABLE_METHODS and item not in family_methods
        ]
        if foreign:
            problems.append(
                f"extensions.integrator_options {foreign} leave the {family} family of method {method!r}"
            )
    return (method, *[m for m in family_methods if m != method])


def _lowering_profile(
    *,
    executable: bool,
    record_reasons: Sequence[str],
    substeps: int,
    edge_detection: bool,
    stochastic: bool,
    admissible: tuple[str, ...],
    recommended_precision: str,
) -> LoweringProfile:
    """Mirror the equation compiler's acceptance rules for the realisation."""
    limits: list[str] = []
    if not executable:
        return LoweringProfile(False, tuple(record_reasons), (), recommended_precision)
    supported = True
    if substeps > 1 and not edge_detection:
        supported = False
        limits.append("substeps > 1 lower only for crossing, non-resetting models")
    if substeps > 1:
        limits.append("no multiply pipelining with substeps > 1")
    if stochastic:
        limits.append("no multiply pipelining with a stochastic threshold")
    cosim = tuple(m for m in admissible if m in {"euler", "map"})
    return LoweringProfile(supported, tuple(limits), cosim, recommended_precision)


def admit_overrides(
    profile: ModelProfile,
    *,
    dt: float | None = None,
    method: str | None = None,
    parameters: Mapping[str, float] | None = None,
    rng_seed: int | None = None,
) -> AdmittedOverrides:
    """Admit consumer overrides under a profile or refuse them.

    Parameters
    ----------
    profile:
        Resolved profile of the schema being instantiated.
    dt:
        Requested integration step, ``None`` to keep the schema's.
    method:
        Requested method, ``None`` to keep the schema's.
    parameters:
        Requested parameter overrides.
    rng_seed:
        Requested LFSR seed, ``None`` to keep the schema's.

    Returns
    -------
    AdmittedOverrides
        The effective step, method, parameter overrides (timebase parameters
        follow the step) and seed.

    Raises
    ------
    ProfileAdmissionError
        If the schema itself is contradictory, is a descriptive record, or the
        override leaves the profile's family, contradicts its timebase, changes
        the step of a recurrence without a timebase, or seeds a deterministic
        model.
    """
    if profile.problems:
        raise ProfileAdmissionError(
            f"schema {profile.stem or profile.scientific.name!r} contradicts its profile: "
            + "; ".join(profile.problems)
        )
    if not profile.is_executable:
        raise ProfileAdmissionError(
            f"schema {profile.stem or profile.scientific.name!r} is a descriptive record: "
            + "; ".join(profile.lowering.limits)
        )
    numerical = profile.numerical
    effective_method = numerical.method if method is None else method
    if effective_method not in numerical.admissible_methods:
        raise ProfileAdmissionError(
            f"method {effective_method!r} is not admissible for the {numerical.family} profile "
            f"of {profile.stem or profile.scientific.name!r} (admissible: "
            f"{', '.join(numerical.admissible_methods)})"
        )
    effective_dt = numerical.dt if dt is None else float(dt)
    if not math.isfinite(effective_dt) or effective_dt <= 0.0:
        raise ProfileAdmissionError(f"dt must be finite and positive, got {dt!r}")
    if (
        dt is not None
        and effective_dt != numerical.dt
        and numerical.family == "map"
        and not numerical.timebase_parameters
    ):
        raise ProfileAdmissionError(
            f"recurrence {profile.stem or profile.scientific.name!r} has no continuous timebase: "
            f"dt {effective_dt} would not change the map (declared dt {numerical.dt})"
        )
    effective_parameters = dict(parameters or {})
    for parameter in numerical.timebase_parameters:
        requested = effective_parameters.get(parameter.name)
        if requested is not None and float(requested) != effective_dt:
            raise ProfileAdmissionError(
                f"timebase parameter {parameter.name!r} = {requested} contradicts dt = {effective_dt}"
            )
        if dt is not None:
            effective_parameters[parameter.name] = effective_dt
    if rng_seed is not None and numerical.randomness.kind == "none":
        raise ProfileAdmissionError(
            f"profile {profile.stem or profile.scientific.name!r} draws no randomness; "
            "rng_seed has no effect and is refused"
        )
    effective_seed = (
        None
        if numerical.randomness.kind == "none"
        else (rng_seed if rng_seed is not None else numerical.randomness.seed)
    )
    derived = effective_method != numerical.method or effective_dt != numerical.dt
    return AdmittedOverrides(
        dt=effective_dt,
        method=effective_method,
        parameters=effective_parameters,
        rng_seed=effective_seed,
        derived=derived,
    )


def parse_profile(payload: Mapping[str, Any]) -> ModelProfile:
    """Rebuild a :class:`ModelProfile` from its public projection.

    Parameters
    ----------
    payload:
        A mapping produced by :meth:`ModelProfile.to_public_dict`.

    Returns
    -------
    ModelProfile
        A profile equal to the one that produced the payload.

    Raises
    ------
    ModelProfileError
        If the payload does not carry the contract or a required field.
    """
    if payload.get("contract") != PROFILE_CONTRACT:
        raise ModelProfileError(
            f"profile payload contract {payload.get('contract')!r} is not {PROFILE_CONTRACT}"
        )
    try:
        scientific_raw = _mapping(payload["scientific"])
        numerical_raw = _mapping(payload["numerical"])
        lowering_raw = _mapping(payload["lowering"])
        randomness_raw = _mapping(numerical_raw["randomness"])
        event_raw = _mapping(numerical_raw["event"])
        scientific = ScientificModel(
            name=str(scientific_raw["name"]),
            author=str(scientific_raw["author"]),
            year=scientific_raw["year"],
            doi=str(scientific_raw["doi"]),
            equations=dict(scientific_raw["equations"]),
            biological_state=tuple(
                _state_variable(item) for item in scientific_raw["biological_state"]
            ),
            source_parameters=tuple(
                _parameter(item) for item in scientific_raw["source_parameters"]
            ),
            published_equations=str(scientific_raw["published_equations"]),
        )
        numerical = NumericalRealisation(
            method=str(numerical_raw["method"]),
            family=numerical_raw["family"],
            exactness=numerical_raw["exactness"],
            exactness_claimed=bool(numerical_raw["exactness_claimed"]),
            dt=float(numerical_raw["dt"]),
            time_unit=str(numerical_raw["time_unit"]),
            substeps=int(numerical_raw["substeps"]),
            substep_kind=numerical_raw["substep_kind"],
            macro_step=(
                None if numerical_raw["macro_step"] is None else float(numerical_raw["macro_step"])
            ),
            evaluation_order=tuple(str(item) for item in numerical_raw["evaluation_order"]),
            auxiliary_registers=tuple(
                _state_variable(item) for item in numerical_raw["auxiliary_registers"]
            ),
            implementation_parameters=tuple(
                _parameter(item) for item in numerical_raw["implementation_parameters"]
            ),
            timebase_parameters=tuple(
                _parameter(item) for item in numerical_raw["timebase_parameters"]
            ),
            admissible_methods=tuple(str(item) for item in numerical_raw["admissible_methods"]),
            randomness=RandomnessContract(
                kind=randomness_raw["kind"],
                seed=randomness_raw["seed"],
                reproducible=bool(randomness_raw["reproducible"]),
            ),
            event=EventContract(
                detection=str(event_raw["detection"]),
                condition=str(event_raw["condition"]),
                stochastic_expression=str(event_raw["stochastic_expression"]),
                reset=dict(event_raw["reset"]),
                edge_detection=bool(event_raw["edge_detection"]),
                refractory_register=str(event_raw["refractory_register"]),
            ),
        )
        lowering = LoweringProfile(
            rtl_supported=bool(lowering_raw["rtl_supported"]),
            limits=tuple(str(item) for item in lowering_raw["limits"]),
            cosim_methods=tuple(str(item) for item in lowering_raw["cosim_methods"]),
            recommended_precision=str(lowering_raw["recommended_precision"]),
        )
        return ModelProfile(
            contract=PROFILE_CONTRACT,
            stem=str(payload["stem"]),
            realisation_kind=payload["realisation_kind"],
            authored=bool(payload["authored"]),
            scientific=scientific,
            numerical=numerical,
            lowering=lowering,
            problems=tuple(str(item) for item in payload["problems"]),
            notes=str(payload["notes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModelProfileError(f"malformed profile payload: {error}") from error


def _state_variable(item: object) -> StateVariable:
    raw = _mapping(item)
    return StateVariable(
        name=str(raw["name"]),
        role=raw["role"],
        init=float(raw["init"]),
        unit=str(raw["unit"]),
        meaning=str(raw["meaning"]),
    )


def _parameter(item: object) -> Parameter:
    raw = _mapping(item)
    return Parameter(
        name=str(raw["name"]),
        role=raw["role"],
        default=float(raw["default"]),
        unit=str(raw["unit"]),
        meaning=str(raw["meaning"]),
    )


__all__ = [
    "EXECUTABLE_DETECTIONS",
    "EXECUTABLE_METHODS",
    "METHOD_TABLE",
    "PROFILE_CONTRACT",
    "PROFILE_SECTION",
    "AdmittedOverrides",
    "EventContract",
    "LoweringProfile",
    "ModelProfile",
    "ModelProfileError",
    "NumericalRealisation",
    "Parameter",
    "ProfileAdmissionError",
    "RandomnessContract",
    "ScientificModel",
    "StateVariable",
    "admit_overrides",
    "expression_names",
    "parse_profile",
    "resolve_profile",
]
