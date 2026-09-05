# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model-declared state layout and snapshots

"""Model-declared state layout, observed shapes and exact state snapshots.

A Studio run records the state a model *declares*: the committed descriptor's
``[state]`` table (name, initial value, unit, meaning) joined with the role the
canonical schema profile assigns to each variable (``biological`` for a quantity
of the scientific model, ``auxiliary`` for a register of the numerical scheme or
event logic). Name heuristics are not a state source. The shape of every
declared variable is observed on the constructed instance before the first
step and must stay constant; a declared variable that is not observable on the
instance is reported with its reason instead of being dropped, and attributes
that change without being declared are reported as ``undeclared_mutable``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_identity import ModelIdentityError, schema_for_class
from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.neurons.universal_dsl import load_schema

LAYOUT_SCHEMA_VERSION = "studio.state-layout.v1"
FINGERPRINT_LIMIT = 4096

LayoutSource = Literal["descriptor", "equations", "undeclared"]
StateKind = Literal["scalar", "vector"]
StateRole = Literal["biological", "auxiliary", "unassigned"]

Snapshot = dict[str, float | np.ndarray[Any, Any]]


@dataclass(frozen=True, slots=True)
class DeclaredState:
    """One state variable as the model declares it.

    Parameters
    ----------
    name : str
        Attribute name on the model instance (or variable name of an equation
        neuron).
    role : {"biological", "auxiliary", "unassigned"}
        Role from the canonical schema profile; ``unassigned`` when the class
        has no bound profile or the profile does not name the variable.
    unit : str
        Declared unit, empty when undeclared.
    meaning : str
        Declared meaning, empty when undeclared.
    declared_init : float or None
        Declared initial value, ``None`` when the declaration carries none.
    """

    name: str
    role: StateRole
    unit: str
    meaning: str
    declared_init: float | None


@dataclass(frozen=True, slots=True)
class ObservedState:
    """A declared state variable with the shape observed on the instance.

    Parameters
    ----------
    declared : DeclaredState
        The declaration.
    kind : {"scalar", "vector"} or None
        Observed kind; ``None`` when the variable is not observable.
    shape : tuple of int or None
        Observed shape (``()`` for a scalar); ``None`` when not observable.
    observable : bool
        Whether the run can record the variable.
    reason : str
        Why the variable is not observable (empty when it is).
    trace : {"per-step", "snapshots-only", "none"}
        How the run records the variable: every step, only in the initial and
        final snapshots (a vector beyond the raw budget), or not at all.
    """

    declared: DeclaredState
    kind: StateKind | None
    shape: tuple[int, ...] | None
    observable: bool
    reason: str
    trace: Literal["per-step", "snapshots-only", "none"]

    @property
    def name(self) -> str:
        """Attribute name of the variable."""
        return self.declared.name

    def to_public_dict(self) -> dict[str, object]:
        """Return the path-free public description of the variable."""
        return {
            "name": self.declared.name,
            "role": self.declared.role,
            "unit": self.declared.unit,
            "meaning": self.declared.meaning,
            "declared_init": self.declared.declared_init,
            "kind": self.kind,
            "shape": list(self.shape) if self.shape is not None else None,
            "observable": self.observable,
            "reason": self.reason,
            "trace": self.trace,
        }


@dataclass(frozen=True, slots=True)
class StateLayout:
    """The declared state layout of one run and its custody verdict.

    Parameters
    ----------
    source : {"descriptor", "equations", "undeclared"}
        Where the declaration comes from.
    schema_profile : str
        Canonical schema stem whose profile assigned the roles, else empty.
    variables : tuple of ObservedState
        Every declared variable, observable or not.
    undeclared_mutable : tuple of str
        Instance attributes that changed between the initial and the final
        snapshot without being declared (private registers included).
    """

    source: LayoutSource
    schema_profile: str
    variables: tuple[ObservedState, ...]
    undeclared_mutable: tuple[str, ...] = ()
    custody_notes: tuple[str, ...] = ()

    @property
    def observable(self) -> tuple[ObservedState, ...]:
        """Variables the run records."""
        return tuple(v for v in self.variables if v.observable)

    @property
    def per_step(self) -> tuple[ObservedState, ...]:
        """Variables recorded at every step."""
        return tuple(v for v in self.variables if v.observable and v.trace == "per-step")

    @property
    def scalars(self) -> tuple[ObservedState, ...]:
        """Observable scalar variables (the ones the display projection shows)."""
        return tuple(v for v in self.variables if v.observable and v.kind == "scalar")

    def incomplete_reasons(self) -> tuple[str, ...]:
        """Return why the recorded state is not the complete declared state."""
        reasons: list[str] = []
        if self.source == "undeclared":
            reasons.append("the model declares no state layout")
        for variable in self.variables:
            if not variable.observable:
                reasons.append(f"{variable.name}: {variable.reason}")
            elif variable.trace != "per-step":
                reasons.append(f"{variable.name}: recorded in snapshots only")
        for name in self.undeclared_mutable:
            reasons.append(f"{name}: changed during the run but is not declared state")
        reasons.extend(self.custody_notes)
        return tuple(reasons)

    @property
    def complete(self) -> bool:
        """Whether every declared variable was recorded and nothing undeclared moved."""
        return not self.incomplete_reasons()

    def with_undeclared_mutable(self, names: Iterable[str]) -> StateLayout:
        """Return a copy carrying the start-to-end mutation audit."""
        return StateLayout(
            source=self.source,
            schema_profile=self.schema_profile,
            variables=self.variables,
            undeclared_mutable=tuple(sorted(names)),
            custody_notes=self.custody_notes,
        )

    def with_custody_notes(self, notes: Iterable[str]) -> StateLayout:
        """Return a copy carrying backend custody limitations."""
        return StateLayout(
            source=self.source,
            schema_profile=self.schema_profile,
            variables=self.variables,
            undeclared_mutable=self.undeclared_mutable,
            custody_notes=tuple(notes),
        )

    def to_public_dict(self) -> dict[str, object]:
        """Return the path-free public layout with its custody verdict."""
        return {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "source": self.source,
            "schema_profile": self.schema_profile,
            "variables": [variable.to_public_dict() for variable in self.variables],
            "recorded": [variable.name for variable in self.per_step],
            "complete": self.complete,
            "incomplete_reasons": list(self.incomplete_reasons()),
            "undeclared_mutable": list(self.undeclared_mutable),
            "custody_notes": list(self.custody_notes),
        }


def _profile_roles(class_name: str) -> tuple[str, dict[str, StateRole]]:
    """Return the canonical schema stem and the roles its profile assigns."""
    try:
        stem = schema_for_class(class_name)
    except ModelIdentityError:
        return "", {}
    try:
        schema = load_schema(stem)
    except (FileNotFoundError, ValueError):
        return "", {}
    profile = resolve_profile(schema, stem=stem)
    roles: dict[str, StateRole] = {}
    for variable in profile.scientific.biological_state:
        roles[variable.name] = "biological"
    for variable in profile.numerical.auxiliary_registers:
        roles[variable.name] = "auxiliary"
    return stem, roles


def declared_state(class_name: str) -> tuple[LayoutSource, str, tuple[DeclaredState, ...]]:
    """Return the declared state of a catalogue class.

    Parameters
    ----------
    class_name : str
        Registered catalogue class name.

    Returns
    -------
    tuple
        ``(source, schema_profile, variables)``: ``descriptor`` with the
        committed ``[state]`` table joined with the canonical profile roles, or
        ``undeclared`` with no variables when the descriptor is absent or
        declares no state. The descriptor is the authority; the profile only
        assigns roles.
    """
    descriptor = load_descriptor(class_name)
    if descriptor is None or not descriptor.state:
        return "undeclared", "", ()
    stem, roles = _profile_roles(class_name)
    variables = tuple(
        DeclaredState(
            name=spec.name,
            role=roles.get(spec.name, "unassigned"),
            unit=spec.unit,
            meaning=spec.meaning,
            declared_init=float(spec.init),
        )
        for spec in descriptor.state
    )
    return "descriptor", stem, variables


def equation_state(
    names: Iterable[str], init: Mapping[str, float] | None
) -> tuple[DeclaredState, ...]:
    """Return the declared state of an equation-playground neuron.

    Parameters
    ----------
    names : iterable of str
        State variable names in the neuron's evaluation order.
    init : mapping or None
        Initial values the request declared; a variable without one carries
        ``None``.
    """
    declared = dict(init or {})
    return tuple(
        DeclaredState(
            name=name,
            role="unassigned",
            unit="",
            meaning="",
            declared_init=float(declared[name]) if name in declared else None,
        )
        for name in names
    )


def scalar_value(value: object) -> float | None:
    """Return ``value`` as a float when it is a real scalar, otherwise ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value)


def vector_value(value: object) -> np.ndarray[Any, Any] | None:
    """Return ``value`` as a float64 array when it is a numeric array of rank ≥ 1."""
    if not isinstance(value, np.ndarray) or value.ndim < 1 or value.dtype.kind not in "fiu":
        return None
    return np.asarray(value, dtype=np.float64)


def _attribute(instance: object, name: str) -> tuple[bool, object]:
    """Read a state variable from an object attribute or a state mapping."""
    if isinstance(instance, Mapping):
        return name in instance, instance.get(name)
    try:
        return True, getattr(instance, name)
    except AttributeError:
        return False, None


def observe_layout(
    instance: object,
    source: LayoutSource,
    schema_profile: str,
    declared: Iterable[DeclaredState],
    *,
    n_steps: int,
    element_budget: int,
) -> StateLayout:
    """Observe the shape of every declared variable on a constructed instance.

    Parameters
    ----------
    instance : object
        The model instance before its first step.
    source, schema_profile : str
        Provenance of the declaration (see :func:`declared_state`).
    declared : iterable of DeclaredState
        The declared variables.
    n_steps : int
        Steps the run will execute; decides whether a vector trace fits the
        raw element budget.
    element_budget : int
        Maximum raw elements the run may return per variable trace.

    Returns
    -------
    StateLayout
        Every declared variable with its observed kind and shape, or the
        reason it cannot be recorded.
    """
    observed: list[ObservedState] = []
    for spec in declared:
        present, value = _attribute(instance, spec.name)
        if not present:
            observed.append(
                ObservedState(
                    spec, None, None, False, "not an attribute of the model instance", "none"
                )
            )
            continue
        scalar = scalar_value(value)
        if scalar is not None:
            observed.append(ObservedState(spec, "scalar", (), True, "", "per-step"))
            continue
        vector = vector_value(value)
        if vector is not None:
            fits = n_steps * int(vector.size) <= element_budget
            observed.append(
                ObservedState(
                    spec,
                    "vector",
                    tuple(int(n) for n in vector.shape),
                    True,
                    "",
                    "per-step" if fits else "snapshots-only",
                )
            )
            continue
        observed.append(
            ObservedState(
                spec,
                None,
                None,
                False,
                f"unsupported state value ({type(value).__name__})",
                "none",
            )
        )
    return StateLayout(source=source, schema_profile=schema_profile, variables=tuple(observed))


class StateObservationError(ValueError):
    """Raised when an observable variable is not a finite value of its observed shape.

    Parameters
    ----------
    name : str
        Variable name.
    reason : str
        Bounded description of the violation.
    """

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"state {name!r} {reason}")
        self.name = name
        self.reason = reason


def read_variable(instance: object, variable: ObservedState) -> float | np.ndarray[Any, Any]:
    """Read one observable variable and enforce its observed kind, shape and finiteness.

    Raises
    ------
    StateObservationError
        When the value is no longer of the observed kind or shape, or is not
        finite.
    """
    present, value = _attribute(instance, variable.name)
    if not present:
        raise StateObservationError(variable.name, "disappeared from the model instance")
    if variable.kind == "scalar":
        scalar = scalar_value(value)
        if scalar is None:
            raise StateObservationError(
                variable.name, f"is no longer a scalar ({type(value).__name__})"
            )
        if not math.isfinite(scalar):
            raise StateObservationError(variable.name, f"became non-finite ({scalar!r})")
        return scalar
    vector = vector_value(value)
    if vector is None:
        raise StateObservationError(
            variable.name, f"is no longer a vector ({type(value).__name__})"
        )
    if tuple(int(n) for n in vector.shape) != variable.shape:
        raise StateObservationError(
            variable.name, f"changed shape from {variable.shape} to {tuple(vector.shape)}"
        )
    if not bool(np.all(np.isfinite(vector))):
        raise StateObservationError(variable.name, "became non-finite")
    return vector.copy()


def snapshot(instance: object, layout: StateLayout) -> Snapshot:
    """Return the exact value of every observable variable of the layout."""
    return {variable.name: read_variable(instance, variable) for variable in layout.observable}


def public_snapshot(
    values: Mapping[str, float | np.ndarray[Any, Any]],
) -> dict[str, float | list[Any]]:
    """Return a snapshot as JSON values (vectors as nested lists)."""
    public: dict[str, float | list[Any]] = {}
    for name, value in values.items():
        if isinstance(value, np.ndarray):
            public[name] = value.tolist()
        else:
            public[name] = float(value)
    return public


def _fingerprint(value: object) -> str:
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return repr(value)
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}{value.dtype}:{value.tobytes().hex()[:FINGERPRINT_LIMIT]}"
    if isinstance(value, (str, bytes)):
        return repr(value)[:FINGERPRINT_LIMIT]
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return repr(value)[:FINGERPRINT_LIMIT]
    return repr(value)[:FINGERPRINT_LIMIT]


def _attribute_names(instance: object) -> list[str]:
    names: list[str] = []
    if hasattr(instance, "__dict__"):
        names.extend(vars(instance).keys())
    for klass in type(instance).__mro__:
        for slot in getattr(klass, "__slots__", ()):
            if (
                isinstance(slot, str)
                and slot not in names
                and slot not in ("__dict__", "__weakref__")
            ):
                names.append(slot)
    return names


def attribute_fingerprints(instance: object) -> dict[str, str]:
    """Fingerprint every instance attribute (public and private) for the mutation audit."""
    fingerprints: dict[str, str] = {}
    for name in _attribute_names(instance):
        present, value = _attribute(instance, name)
        if present:
            fingerprints[name] = _fingerprint(value)
    return fingerprints


def undeclared_mutations(
    before: Mapping[str, str], after: Mapping[str, str], layout: StateLayout
) -> tuple[str, ...]:
    """Return attributes that changed between two fingerprints without being declared."""
    declared = {variable.name for variable in layout.variables}
    changed = [
        name
        for name in sorted(set(before) | set(after))
        if name not in declared and before.get(name) != after.get(name)
    ]
    return tuple(changed)


__all__ = [
    "FINGERPRINT_LIMIT",
    "LAYOUT_SCHEMA_VERSION",
    "DeclaredState",
    "LayoutSource",
    "ObservedState",
    "Snapshot",
    "StateKind",
    "StateLayout",
    "StateObservationError",
    "StateRole",
    "attribute_fingerprints",
    "declared_state",
    "equation_state",
    "observe_layout",
    "public_snapshot",
    "read_variable",
    "scalar_value",
    "snapshot",
    "undeclared_mutations",
    "vector_value",
]
