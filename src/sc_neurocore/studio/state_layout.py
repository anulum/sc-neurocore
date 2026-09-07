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

#: The declared variable through which a model publishes its generator state.
#: The same name is the generator contract in ``descriptor_generator``.
RNG_STATE_VARIABLE = "rng_state"

#: The private attributes a model holds its generator in behind that variable.
#: Their movement is the declared variable's movement, not undeclared state.
RNG_STATE_BACKING = ("_rng", "_rng_state")

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


#: Descriptor names for the refractory register where the descriptor and the
#: canonical schema chose different words for one variable. The schema side is
#: not repeated here: the profile names it itself, in
#: ``numerical.event.refractory_register``, so only the descriptor's word is
#: recorded and the two cannot drift apart.
#:
#: The divergence is real and deliberate on both sides. The descriptor's name is
#: what a run records, what ``get_state`` returns and what committed conformance
#: evidence carries, so renaming it would move operator-visible trace fields;
#: the schema's name is what the profile and the lowering path use. Joining them
#: by name alone left the variable ``unassigned`` on every run — carrying none
#: of the role its own profile states.
#:
#: This is a per-identity record, never a rule: a name that matches needs no
#: entry, and a new divergence must be added deliberately rather than guessed at
#: by matching whatever is left over.
_REFRACTORY_ALIASES: dict[str, str] = {
    "BrunelWangNeuron": "ref_remaining",
    "CompteWMNeuron": "ref_remaining",
}


def _profile_roles(class_name: str) -> tuple[str, dict[str, StateRole]]:
    """Return the canonical schema stem and the roles its profile assigns.

    Roles are keyed by the name the descriptor uses, because that is the name a
    run records. Where the two authorities chose different words for the
    refractory register, :data:`_REFRACTORY_ALIASES` carries the descriptor's
    word and the profile supplies its own.
    """
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
    alias = _REFRACTORY_ALIASES.get(class_name)
    schema_name = profile.numerical.event.refractory_register
    if alias is not None and schema_name in roles:
        roles[alias] = roles[schema_name]
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
        ``undeclared`` with no variables when the descriptor is absent or its
        state has not been declared. The descriptor is the authority; the
        profile only assigns roles.

        A descriptor that asserts ``stateless`` answers ``descriptor`` with no
        variables. Emptiness is then a declaration rather than a silence: the
        run records everything the model declares, so its custody is complete,
        where a model whose state nobody has declared stays ``undeclared`` and
        incomplete. The assertion does not exempt the model from the mutation
        audit — anything that moves is still reported as undeclared state.
    """
    descriptor = load_descriptor(class_name)
    if descriptor is None:
        return "undeclared", "", ()
    if not descriptor.state:
        if not descriptor.stateless:
            return "undeclared", "", ()
        stem, _roles = _profile_roles(class_name)
        return "descriptor", stem, ()
    stem, roles = _profile_roles(class_name)
    variables = tuple(
        DeclaredState(
            name=spec.name,
            role=roles.get(spec.name, "unassigned"),
            unit=spec.unit,
            meaning=spec.meaning,
            declared_init=None if spec.init is None else float(spec.init),
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
    """Return a declared state variable as a float, or ``None`` when it is not one.

    A flag is state. ``LapicqueNeuron`` declares ``excited``, which latches the
    first threshold attainment and which its canonical schema lowers as 0.0 or
    1.0 in the RTL, and a run that refused to record it published incomplete
    custody for a register the model tracks exactly. A boolean is therefore read
    as the number the rest of the toolchain already carries it as.

    This function exists only to read *declared* state: an attribute nobody
    declared is never offered to it, so admitting flags here says nothing about
    an undeclared one.
    """
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value)


def vector_value(value: object) -> np.ndarray[Any, Any] | None:
    """Return ``value`` as a float64 array when it is a numeric vector.

    A model may hold a compartment vector, a population activity profile or a
    filter buffer as a NumPy array or as a plain list — the choice is the
    model's, and it is not a statement about whether the quantity is state. A
    numeric sequence is therefore read as the vector it is; a string, a
    dictionary, a ragged sequence or a sequence carrying a non-number is not a
    vector and is refused, so an unrecordable value is still reported with its
    reason rather than coerced into a shape.
    """
    if isinstance(value, np.ndarray):
        if value.ndim < 1 or value.dtype.kind not in "fiu":
            return None
        return np.asarray(value, dtype=np.float64)
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        return None
    if not all(
        not isinstance(item, bool) and isinstance(item, (int, float, np.integer, np.floating))
        for item in value
    ):
        return None
    return np.asarray(value, dtype=np.float64).reshape(len(value))


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


def _generator_state(value: object) -> str | None:
    """Return the state of a random generator, or ``None`` when it is not one.

    A generator advances on every draw, and that advance is the difference
    between a run that can be replayed from its snapshot and one that cannot.
    A NumPy ``Generator`` reports it through its bit generator; the library's
    own generators expose a ``state`` attribute. Neither shows in ``repr``.
    """
    bit_generator = getattr(value, "bit_generator", None)
    if bit_generator is None:
        inner = getattr(value, "_rng", None)
        bit_generator = getattr(inner, "bit_generator", None)
    if bit_generator is not None:
        return f"bitgenerator:{bit_generator.state!r}"[:FINGERPRINT_LIMIT]
    state = getattr(value, "state", None)
    if isinstance(state, (bool, int, float, np.integer, np.floating)):
        return f"state:{state!r}"
    return None


def _is_opaque(value: object) -> bool:
    """Return whether an object's ``repr`` is the address-based default.

    ``object.__repr__`` prints an identity, not a value, so it is constant
    across every mutation of the object it names. Fingerprinting such a value
    cannot detect a change inside it; the audit records that it could not look
    rather than reporting the object as unchanged.
    """
    return type(value).__repr__ is object.__repr__


def _fingerprint(value: object) -> str:
    """Return a value's fingerprint for the start-to-end mutation audit.

    The fingerprint must change when the value changes, or the audit reports a
    register that moved as one that did not. Numbers, arrays, text and
    containers fingerprint by their contents. A random generator fingerprints
    by its generator state, which no ``repr`` exposes. Anything else falls back
    to ``repr``, and an object whose ``repr`` is the address-based default is
    marked opaque rather than treated as a value, so a blind spot is
    enumerable instead of silent.
    """
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return repr(value)
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}{value.dtype}:{value.tobytes().hex()[:FINGERPRINT_LIMIT]}"
    if isinstance(value, (str, bytes)):
        return repr(value)[:FINGERPRINT_LIMIT]
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return repr(value)[:FINGERPRINT_LIMIT]
    generator = _generator_state(value)
    if generator is not None:
        return generator
    if _is_opaque(value):
        return f"opaque:{type(value).__module__}.{type(value).__qualname__}"
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
    """Return attributes that changed between two fingerprints without being declared.

    A declared variable may be a public view of a private one. The library's
    generator contract is exactly that: a model exposes ``rng_state`` as a
    read-only property over a private generator, and the descriptor declares
    the property. The generator object then moves under a name the layout never
    names, but its movement *is* the declared variable's movement, so it is
    accounted for rather than reported. A model that carries a generator
    without declaring ``rng_state`` is not accounted for, and is reported.
    """
    declared = {variable.name for variable in layout.variables}
    accounted = set(RNG_STATE_BACKING) if RNG_STATE_VARIABLE in declared else set()
    changed = [
        name
        for name in sorted(set(before) | set(after))
        if name not in declared and name not in accounted and before.get(name) != after.get(name)
    ]
    return tuple(changed)


__all__ = [
    "FINGERPRINT_LIMIT",
    "LAYOUT_SCHEMA_VERSION",
    "RNG_STATE_BACKING",
    "RNG_STATE_VARIABLE",
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
