# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A parameter-fitting problem: model, domains and a split cohort

"""State a fitting problem so that nothing about it is implicit.

A problem names a Universal DSL model (a catalogue model's canonical schema or
a candidate's model), the state variable that was observed, the domain of
every parameter to fit, and a cohort of recordings split into a training set
and a hold-out set. The split is part of the problem: the fitting objective is
given the training recordings only, and a recording whose data appears in both
sets is refused, so hold-out data cannot reach the fit by accident.

The model runs under its own declared profile through the Universal DSL. A run
whose state stops being finite is reported as a failed trial, never replaced by
a default or cut short silently.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.fitting.refusals import LaboratoryRefusal
from sc_neurocore.fitting.constraints import ParameterConstraint, constraints_from_dict

FIT_SCHEMA_VERSION = "sc-neurocore.fit.v1"
"""Legacy unconstrained document version; grouped/constrained problems use v2."""

Scale = Literal["linear", "log"]


def canonical_sha256(payload: object) -> str:
    """Return the digest of a JSON value in canonical form."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParameterDomain:
    """The range one parameter is searched over.

    Attributes
    ----------
    name:
        A parameter of the model's schema.
    low, high:
        Finite bounds, ``low < high``.
    scale:
        ``log`` searches the logarithm (bounds must be positive), for a
        parameter whose plausible values span decades.
    """

    name: str
    low: float
    high: float
    scale: Scale = "linear"

    def __post_init__(self) -> None:
        if self.scale not in ("linear", "log"):
            raise LaboratoryRefusal(f"scale of {self.name} must be linear or log")
        if not (math.isfinite(self.low) and math.isfinite(self.high)) or self.low >= self.high:
            raise LaboratoryRefusal(f"domain of {self.name} must be finite with low < high")
        if self.scale == "log" and self.low <= 0:
            raise LaboratoryRefusal(f"a log domain of {self.name} must be positive")

    def internal_bounds(self) -> tuple[float, float]:
        """Return the bounds in the space the optimiser searches."""
        if self.scale == "log":
            return math.log(self.low), math.log(self.high)
        return self.low, self.high

    def to_value(self, internal: float) -> float:
        """Map a searched coordinate back to the parameter's own value."""
        return math.exp(internal) if self.scale == "log" else internal

    def to_public_dict(self) -> dict[str, Any]:
        """Return the domain as it is exported."""
        return {"name": self.name, "low": self.low, "high": self.high, "scale": self.scale}


@dataclass(frozen=True)
class Recording:
    """One stimulus and the observed variable's response, sample by sample."""

    name: str
    current: tuple[float, ...]
    observed: tuple[float, ...]
    group: str | None = None

    def __post_init__(self) -> None:
        """Validate sample lengths, finiteness and optional acquisition custody."""
        if self.group is not None and (not isinstance(self.group, str) or not self.group.strip()):
            raise LaboratoryRefusal("acquisition groups must be nonempty")
        if len(self.current) != len(self.observed) or len(self.current) < 2:
            raise LaboratoryRefusal(
                f"recording {self.name} needs equal current and observed samples, >= 2"
            )
        if not all(math.isfinite(value) for value in (*self.current, *self.observed)):
            raise LaboratoryRefusal(f"recording {self.name} must hold finite samples only")

    @property
    def data_sha256(self) -> str:
        """Digest of the samples, independent of the recording's name."""
        return canonical_sha256({"current": self.current, "observed": self.observed})

    def to_public_dict(self) -> dict[str, Any]:
        """Return the recording as it is exported."""
        return {
            "name": self.name,
            "current": list(self.current),
            "observed": list(self.observed),
            **({"group": self.group} if self.group is not None else {}),
        }


@dataclass(frozen=True)
class FitProblem:
    """A model, what to fit in it, and a cohort split into training and hold-out."""

    schema: Mapping[str, Any]
    observable: str
    domains: tuple[ParameterDomain, ...]
    train: tuple[Recording, ...]
    holdout: tuple[Recording, ...]
    seed: int
    fixed: Mapping[str, float] = field(default_factory=dict)
    constraints: tuple[ParameterConstraint, ...] = ()

    def __post_init__(self) -> None:
        """Admit the model, parameter references, seed and leakage-safe split."""
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise LaboratoryRefusal("fit seed must be an integer in 0..2**32-1")
        state = self.schema.get("state")
        parameters = self.schema.get("parameters")
        if not isinstance(state, Mapping) or self.observable not in state:
            raise LaboratoryRefusal(f"{self.observable!r} is not a state variable of the model")
        if not isinstance(parameters, Mapping):
            raise LaboratoryRefusal("the model declares no parameters")
        names = [domain.name for domain in self.domains]
        if not names or len(set(names)) != len(names):
            raise LaboratoryRefusal("fit at least one parameter, each once")
        for name in (*names, *self.fixed):
            if name not in parameters:
                raise LaboratoryRefusal(f"{name} is not a parameter of the model")
        if set(names) & set(self.fixed):
            raise LaboratoryRefusal("a parameter is either fitted or fixed, not both")
        if not all(math.isfinite(value) for value in self.fixed.values()):
            raise LaboratoryRefusal("fixed parameters must be finite")
        labels = [constraint.name for constraint in self.constraints]
        if len(set(labels)) != len(labels):
            raise LaboratoryRefusal("constraint names must be unique")
        for constraint in self.constraints:
            if set(constraint.coefficients) - set(parameters):
                raise LaboratoryRefusal("a constraint refers to an unknown parameter")
        if not self.train or not self.holdout:
            raise LaboratoryRefusal("a fit needs at least one training and one hold-out recording")
        labels = [recording.name for recording in (*self.train, *self.holdout)]
        if len(set(labels)) != len(labels):
            raise LaboratoryRefusal("recording names must be unique across the cohort")
        if {r.group for r in self.train if r.group is not None} & {
            r.group for r in self.holdout if r.group is not None
        }:
            raise LaboratoryRefusal("acquisition groups cannot cross training and hold-out splits")
        shared = {r.data_sha256 for r in self.train} & {r.data_sha256 for r in self.holdout}
        if shared:
            raise LaboratoryRefusal(
                "a recording's data appears in both the training and the hold-out set"
            )
        from sc_neurocore.neurons.universal_dsl import UniversalNeuron

        UniversalNeuron.from_dict(dict(self.schema), parameter_overrides=dict(self.fixed))

    def to_public_dict(self) -> dict[str, Any]:
        """Return the whole problem as it is exported and replayed."""
        return {
            "schema_version": "sc-neurocore.fit.v2"
            if self.constraints or any(r.group is not None for r in (*self.train, *self.holdout))
            else FIT_SCHEMA_VERSION,
            "schema": dict(self.schema),
            "observable": self.observable,
            "domains": [domain.to_public_dict() for domain in self.domains],
            "fixed": dict(self.fixed),
            "train": [recording.to_public_dict() for recording in self.train],
            "holdout": [recording.to_public_dict() for recording in self.holdout],
            "seed": self.seed,
            **(
                {"constraints": [c.to_public_dict() for c in self.constraints]}
                if self.constraints
                else {}
            ),
        }


def problem_from_dict(document: Mapping[str, Any]) -> FitProblem:
    """Rebuild a problem from its exported form.

    Raises
    ------
    ValueError
        When the document is another version or its fields do not form a
        valid problem.
    """
    if document.get("schema_version") not in (FIT_SCHEMA_VERSION, "sc-neurocore.fit.v2"):
        raise LaboratoryRefusal(f"unsupported fit document {document.get('schema_version')!r}")

    if document.get("schema_version") == FIT_SCHEMA_VERSION and document.get("constraints"):
        raise LaboratoryRefusal("parameter constraints require a v2 fit document")

    def recordings(entries: Sequence[Mapping[str, Any]]) -> tuple[Recording, ...]:
        return tuple(
            Recording(
                name=str(entry["name"]),
                current=tuple(float(value) for value in entry["current"]),
                observed=tuple(float(value) for value in entry["observed"]),
                group=entry.get("group"),
            )
            for entry in entries
        )

    return FitProblem(
        schema=dict(document["schema"]),
        observable=str(document["observable"]),
        domains=tuple(
            ParameterDomain(
                name=str(entry["name"]),
                low=float(entry["low"]),
                high=float(entry["high"]),
                scale=entry.get("scale", "linear"),
            )
            for entry in document["domains"]
        ),
        fixed={str(name): float(value) for name, value in dict(document.get("fixed", {})).items()},
        train=recordings(document["train"]),
        holdout=recordings(document["holdout"]),
        seed=document["seed"],
        constraints=constraints_from_dict(document.get("constraints", [])),
    )


def simulate(
    schema: Mapping[str, Any],
    observable: str,
    parameters: Mapping[str, float],
    current: Sequence[float],
) -> NDArray[np.float64] | None:
    """Run the model with ``parameters`` under ``current`` and return the observable.

    Returns
    -------
    numpy.ndarray or None
        The observable after each step, or ``None`` when the state stopped
        being finite: a failed trial.
    """
    from sc_neurocore.neurons.universal_dsl import UniversalNeuron

    neuron = UniversalNeuron.from_dict(dict(schema), parameter_overrides=dict(parameters))
    trace = np.empty(len(current), dtype=np.float64)
    for index, value in enumerate(current):
        try:
            neuron.step(I=float(value))
        except FloatingPointError:
            return None
        trace[index] = float(neuron.state[observable])
    return trace


__all__ = [
    "FIT_SCHEMA_VERSION",
    "FitProblem",
    "ParameterDomain",
    "Recording",
    "canonical_sha256",
    "problem_from_dict",
    "simulate",
]
