# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Versioned shared-sample experiment cohorts

"""Describe bounded sweeps without hidden stimuli, noise, splits or units."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from sc_neurocore.fitting.constraints import ParameterConstraint, constraints_from_dict
from sc_neurocore.fitting.problem import canonical_sha256

COHORT_VERSION = "sc-neurocore.cohort.v1"
MAX_COHORT_TRIALS = 4096
MAX_COHORT_STEPS = 5_000_000


def cohort_sha256(payload: object) -> str:
    """Hash JSON with integral float values normalised for browser round trips.

    JSON readers may emit ``1`` where Python emitted ``1.0``. Those are the
    same numerical cohort sample and must retain their digest after transport.
    """

    def numeric(value: Any) -> Any:
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, Mapping):
            return {key: numeric(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [numeric(item) for item in value]
        return value

    return canonical_sha256(numeric(payload))


def _finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


@dataclass(frozen=True)
class CohortSample:
    """An independently split recording with explicit additive input noise.

    ``group`` identifies the acquisition, subject or simulation replicate;
    related recordings must stay in one split. ``observations`` names physical
    state variables; ``spikes`` contains one binary event per simulation step.
    Noise is sampled before submission and replayed exactly for every model.
    """

    name: str
    group: str
    split: Literal["train", "holdout"]
    current: tuple[float, ...]
    noise: tuple[float, ...]
    observations: Mapping[str, tuple[float, ...]]
    spikes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Refuse ambiguous splits, missing names and nonfinite samples."""
        size = len(self.current)
        if (
            not self.name.strip()
            or not self.group.strip()
            or self.split not in ("train", "holdout")
        ):
            raise ValueError("samples need names, acquisition groups and train/holdout splits")
        if size < 2 or len(self.noise) != size or (self.spikes and len(self.spikes) != size):
            raise ValueError("current, noise and spikes need equal lengths >= 2")
        if not _finite(self.current + self.noise) or any(
            value not in (0, 1) for value in self.spikes
        ):
            raise ValueError("input/noise must be finite and spikes binary")
        if any(len(values) != size or not _finite(values) for values in self.observations.values()):
            raise ValueError("observations need finite values at every step")
        if not _finite(self.effective_current):
            raise ValueError("effective current must be finite")

    @property
    def effective_current(self) -> tuple[float, ...]:
        """Return the identical additive input supplied to every trial."""
        return tuple(a + b for a, b in zip(self.current, self.noise, strict=True))

    def to_public_dict(self) -> dict[str, Any]:
        """Export all samples and their split custody."""
        return {
            "name": self.name,
            "group": self.group,
            "split": self.split,
            "current": list(self.current),
            "noise": list(self.noise),
            "observations": {key: list(value) for key, value in self.observations.items()},
            "spikes": list(self.spikes),
        }


@dataclass(frozen=True)
class SweepDomain:
    """Explicit finite parameter values, typed as integer or real."""

    name: str
    values: tuple[float, ...]
    kind: Literal["real", "integer"] = "real"

    def __post_init__(self) -> None:
        """Refuse nonfinite, duplicate or incorrectly typed values."""
        if not self.name.strip() or not self.values or not _finite(self.values):
            raise ValueError("sweep domains need a name and finite values")
        if len(set(self.values)) != len(self.values) or self.kind not in ("real", "integer"):
            raise ValueError("sweep values must be unique with a real/integer kind")
        if self.kind == "integer" and any(value != int(value) for value in self.values):
            raise ValueError("integer domains cannot contain fractional values")

    def to_public_dict(self) -> dict[str, Any]:
        """Export every trial value without resampling a range."""
        return {"name": self.name, "values": list(self.values), "kind": self.kind}


@dataclass(frozen=True)
class CohortMetric:
    """A model-specific observable and a declared lower-is-better metric.

    Trace RMSE requires the observed state and its physical unit. Binary event
    disagreement and absolute spike-count error operate on the recorded events.
    """

    kind: Literal["trace_rmse", "event_disagreement", "spike_count_error"]
    observable: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        """Require metric-specific units instead of mixing unlike errors."""
        if self.kind not in ("trace_rmse", "event_disagreement", "spike_count_error"):
            raise ValueError("unsupported cohort metric")
        expected = {"event_disagreement": "fraction", "spike_count_error": "events"}
        if self.kind == "trace_rmse":
            if not self.observable.strip() or not self.unit.strip():
                raise ValueError("trace RMSE needs an observable and physical unit")
        elif self.observable or self.unit != expected[self.kind]:
            raise ValueError("event metrics require their declared unit and no observable")

    def to_public_dict(self) -> dict[str, str]:
        """Export the precise metric contract."""
        return {"kind": self.kind, "observable": self.observable, "unit": self.unit}


@dataclass(frozen=True)
class CohortModel:
    """One complete DSL model, its parameter sweep and its chosen metric."""

    name: str
    schema: Mapping[str, Any]
    domains: tuple[SweepDomain, ...]
    metric: CohortMetric
    fixed: Mapping[str, float] = field(default_factory=dict)
    constraints: tuple[ParameterConstraint, ...] = ()

    def __post_init__(self) -> None:
        """Validate model references and unique parameter/constraint names."""
        from sc_neurocore.neurons.universal_dsl import UniversalNeuron

        UniversalNeuron.from_dict(dict(self.schema))
        parameters = self.schema["parameters"]
        names = [domain.name for domain in self.domains]
        if not self.name.strip() or len(names) != len(set(names)) or set(names) & set(self.fixed):
            raise ValueError("model names and disjoint unique swept/fixed parameters are required")
        referenced = {
            *names,
            *self.fixed,
            *(name for c in self.constraints for name in c.coefficients),
        }
        if referenced - set(parameters) or not _finite(tuple(self.fixed.values())):
            raise ValueError("unknown parameter or nonfinite fixed value")
        constraint_names = [c.name for c in self.constraints]
        if len(set(constraint_names)) != len(constraint_names):
            raise ValueError("constraint names must be unique")
        if self.metric.kind == "trace_rmse":
            units = self.schema.get("profile", {}).get("units", {})
            if (
                self.metric.observable not in self.schema["state"]
                or units.get(self.metric.observable) != self.metric.unit
            ):
                raise ValueError("trace metric must match the model's declared state unit")

    @property
    def trial_count(self) -> int:
        """Return the Cartesian sweep size without materialising trials."""
        return math.prod(len(domain.values) for domain in self.domains)

    def parameter_sets(self) -> itertools.product[tuple[float, ...]]:
        """Iterate the complete declared grid, including infeasible trials."""
        return itertools.product(*(domain.values for domain in self.domains))

    def to_public_dict(self) -> dict[str, Any]:
        """Export the schema, sweep, metric and constraints together."""
        return {
            "name": self.name,
            "schema": dict(self.schema),
            "domains": [d.to_public_dict() for d in self.domains],
            "metric": self.metric.to_public_dict(),
            "fixed": dict(self.fixed),
            "constraints": [c.to_public_dict() for c in self.constraints],
        }


@dataclass(frozen=True)
class ExperimentCohort:
    """Versioned experiment with one time/input contract and leakage-safe splits."""

    name: str
    samples: tuple[CohortSample, ...]
    models: tuple[CohortModel, ...]
    dt: float
    time_unit: str
    input_unit: str
    seed: int
    noise_provenance: str

    def __post_init__(self) -> None:
        """Check sample custody, common timebase and complete pre-run admission."""
        if (
            not self.name.strip()
            or not self.models
            or not self.samples
            or not self.input_unit.strip()
            or not self.noise_provenance.strip()
        ):
            raise ValueError(
                "a cohort needs named models/samples, input units and noise provenance"
            )
        if (
            not math.isfinite(self.dt)
            or self.dt <= 0
            or not self.time_unit.strip()
            or not 0 <= self.seed <= 2**53 - 1
        ):
            raise ValueError("a cohort needs a positive dt, time unit and a safe nonnegative seed")
        if len({s.name for s in self.samples}) != len(self.samples) or len(
            {m.name for m in self.models}
        ) != len(self.models):
            raise ValueError("model and sample names must be unique")
        train, holdout = (
            tuple(s for s in self.samples if s.split == "train"),
            tuple(s for s in self.samples if s.split == "holdout"),
        )
        if not train or not holdout:
            raise ValueError("a cohort needs both train and holdout samples")
        if {s.group for s in train} & {s.group for s in holdout}:
            raise ValueError("acquisition groups cannot cross splits")

        def data_hash(sample: CohortSample) -> str:
            return canonical_sha256(
                {
                    "input": sample.effective_current,
                    "observations": dict(sample.observations),
                    "spikes": sample.spikes,
                }
            )

        if {data_hash(s) for s in train} & {data_hash(s) for s in holdout}:
            raise ValueError("recording data cannot cross splits")
        for model in self.models:
            if (
                model.schema["integration"]["dt"] != self.dt
                or model.schema.get("profile", {}).get("time_unit") != self.time_unit
            ):
                raise ValueError("all models must use the cohort's declared timebase")
            if model.metric.kind != "trace_rmse" and any(
                len(s.spikes) != len(s.current) for s in self.samples
            ):
                raise ValueError("event metrics need recorded binary events at every step")
            if model.metric.kind == "trace_rmse" and any(
                model.metric.observable not in s.observations for s in self.samples
            ):
                raise ValueError("a trace metric needs its observations in every sample")
        if self.trial_count > MAX_COHORT_TRIALS or self.estimated_steps > MAX_COHORT_STEPS:
            raise ValueError(
                "cohort exceeds the declared trial/model-step budget; it is not shortened"
            )

    @property
    def trial_count(self) -> int:
        """Return the complete grid size across all models."""
        return sum(m.trial_count for m in self.models)

    @property
    def estimated_steps(self) -> int:
        """Return the complete simulation-step budget before execution."""
        return self.trial_count * sum(len(s.current) for s in self.samples)

    def to_public_dict(self) -> dict[str, Any]:
        """Export the full effective cohort, sufficient for replay."""
        return {
            "schema_version": COHORT_VERSION,
            "name": self.name,
            "samples": [s.to_public_dict() for s in self.samples],
            "models": [m.to_public_dict() for m in self.models],
            "dt": self.dt,
            "time_unit": self.time_unit,
            "input_unit": self.input_unit,
            "seed": self.seed,
            "noise_provenance": self.noise_provenance,
        }


def _fields(entry: Mapping[str, Any], required: set[str], optional: set[str] | None = None) -> None:
    """Refuse ignored protocol fields and missing scientific definitions."""
    if (
        not isinstance(entry, Mapping)
        or required - set(entry)
        or set(entry) - required - (optional or set())
    ):
        raise ValueError("cohort document has missing or unknown fields")


def _number(value: Any) -> float:
    """Accept JSON numbers without rounding booleans or parsing text."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("cohort samples and sweep values must be JSON numbers")
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        raise ValueError("integer samples must round-trip exactly through JSON number readers")
    return float(value)


def _text(value: Any) -> str:
    """Preserve textual identities without stringifying other document values."""
    if not isinstance(value, str):
        raise ValueError("cohort names, groups and provenance must be strings")
    return value


def cohort_from_dict(document: Mapping[str, Any]) -> ExperimentCohort:
    """Read a full cohort with strict version, field and numeric custody."""
    if document.get("schema_version") != COHORT_VERSION:
        raise ValueError("unsupported cohort document version")
    _fields(
        document,
        {
            "schema_version",
            "name",
            "samples",
            "models",
            "dt",
            "time_unit",
            "input_unit",
            "seed",
            "noise_provenance",
        },
    )
    samples: list[CohortSample] = []
    for entry in document["samples"]:
        _fields(entry, {"name", "group", "split", "current", "noise", "observations"}, {"spikes"})
        samples.append(
            CohortSample(
                _text(entry["name"]),
                _text(entry["group"]),
                entry["split"],
                tuple(_number(v) for v in entry["current"]),
                tuple(_number(v) for v in entry["noise"]),
                {
                    _text(k): tuple(_number(v) for v in values)
                    for k, values in entry["observations"].items()
                },
                tuple(entry.get("spikes", [])),
            )
        )
    models: list[CohortModel] = []
    for entry in document["models"]:
        _fields(entry, {"name", "schema", "domains", "metric"}, {"fixed", "constraints"})
        domains: list[SweepDomain] = []
        for domain in entry["domains"]:
            _fields(domain, {"name", "values"}, {"kind"})
            domains.append(
                SweepDomain(
                    _text(domain["name"]),
                    tuple(_number(v) for v in domain["values"]),
                    domain.get("kind", "real"),
                )
            )
        _fields(entry["metric"], {"kind", "unit"}, {"observable"})
        models.append(
            CohortModel(
                _text(entry["name"]),
                dict(entry["schema"]),
                tuple(domains),
                CohortMetric(**entry["metric"]),
                {_text(k): _number(v) for k, v in entry.get("fixed", {}).items()},
                constraints_from_dict(entry.get("constraints", [])),
            )
        )
    seed = document["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("cohort seed must be an integer")
    return ExperimentCohort(
        _text(document["name"]),
        tuple(samples),
        tuple(models),
        _number(document["dt"]),
        _text(document["time_unit"]),
        _text(document["input_unit"]),
        seed,
        _text(document["noise_provenance"]),
    )
