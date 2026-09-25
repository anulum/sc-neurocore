# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio network graph schema and fail-closed resolution

"""Versioned Studio network-graph schema and its resolution into a typed spec.

A canvas graph (populations, projections, timestep, duration, seed) is
validated field by field and resolved into a :class:`GraphSpec` before any
network object exists. Every population names a catalogue model whose
constructor contract (:mod:`sc_neurocore.studio.model_run_contract`) accepts
the requested parameters and the graph timestep; every projection carries an
explicit connection rule, a signed weight that agrees with its source
population's declared type, a delay that is a whole number of timesteps and a
seed that is either given or derived deterministically from the graph seed.
Nothing is clamped, rounded, defaulted from a template or silently dropped:
each rejection names the field and the reason, and :func:`validate_graph`
reports every rejection of a graph at once.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.model_run_contract import (
    ModelInputError,
    ModelRunInputs,
    model_drive_contract,
    model_parameter_contracts,
    resolve_model_run_inputs,
)
from sc_neurocore.studio.simulation import MAX_STEPS

GRAPH_SCHEMA_VERSION = "studio.network-graph.v1"
GRAPH_SPEC_SCHEMA_VERSION = "studio.network-graph-spec.v1"

DEFAULT_MODEL = "SCLapicqueLIFNeuron"
DEFAULT_DT_MS = 0.1
DEFAULT_DURATION_MS = 200.0
DEFAULT_SEED = 42
MAX_NEURONS = 2000
MAX_NEURON_STEPS = 1_000_000
MAX_GRAPH_STEPS = MAX_STEPS

NEURON_TYPES: tuple[str, ...] = ("excitatory", "inhibitory")
CONNECTION_RULES: tuple[str, ...] = ("random", "all_to_all")
DRIVE_KINDS: tuple[str, ...] = ("none", "constant", "poisson")

NeuronType = Literal["excitatory", "inhibitory"]
ConnectionRule = Literal["random", "all_to_all"]
DriveKind = Literal["none", "constant", "poisson"]
SeedSource = Literal["request", "derived", "none"]

_GRAPH_FIELDS: frozenset[str] = frozenset({"populations", "projections", "duration", "dt", "seed"})
_POPULATION_FIELDS: frozenset[str] = frozenset(
    {"id", "type", "label", "model", "count", "neuron_type", "position", "params", "drive"}
)
_PROJECTION_FIELDS: frozenset[str] = frozenset(
    {"id", "source", "target", "weight", "delay", "probability", "rule", "seed", "autapses"}
)
_DRIVE_FIELDS: frozenset[str] = frozenset({"kind", "current", "rate_hz", "weight", "seed"})
_DELAY_STEP_TOLERANCE = 1e-9
_SEED_SPAWN_PROJECTION = 1
_SEED_SPAWN_DRIVE = 2


class GraphRejected(ValueError):
    """Raised when a graph cannot be resolved into one executable specification.

    Parameters
    ----------
    field : str
        Dotted request field that failed (``populations[1].count``,
        ``projections[0].delay``, ``dt``, …).
    reason : str
        Bounded, path-free reason.
    """

    def __init__(self, *, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {"error": "graph_rejected", "field": self.field, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class GraphIssue:
    """One validation failure: the request field and the human-readable message."""

    field: str
    message: str


@dataclass(frozen=True, slots=True)
class DriveSpec:
    """External input of one population, lowered to a public stimulus object.

    ``constant`` injects ``current`` into every neuron at every step
    (``StepCurrent`` over the whole run); ``poisson`` injects ``weight`` into a
    neuron whenever its independent Poisson process at ``rate_hz`` fires
    (``PoissonInput``); ``none`` injects nothing.
    """

    kind: DriveKind
    current: float | None
    rate_hz: float | None
    weight: float | None
    seed: int | None
    seed_source: SeedSource

    def to_public_dict(self) -> dict[str, Any]:
        """Return the path-free drive block."""
        return {
            "kind": self.kind,
            "current": self.current,
            "rate_hz": self.rate_hz,
            "weight": self.weight,
            "seed": self.seed,
            "seed_source": self.seed_source,
        }


@dataclass(frozen=True, slots=True)
class PopulationSpec:
    """One resolved population: catalogue model, validated constructor inputs, drive."""

    index: int
    id: str
    label: str
    model: str
    count: int
    neuron_type: NeuronType
    inputs: ModelRunInputs
    drive: DriveSpec
    offset: int

    def to_public_dict(self) -> dict[str, Any]:
        """Return the path-free population block (effective parameters included)."""
        return {
            "id": self.id,
            "label": self.label,
            "model": self.model,
            "count": self.count,
            "neuron_type": self.neuron_type,
            "offset": self.offset,
            "parameters": self.inputs.effective_parameters(),
            "overrides_applied": list(self.inputs.overrides_applied),
            "dt": self.inputs.dt,
            "dt_source": self.inputs.dt_source,
            "drive": self.drive.to_public_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """One resolved projection: rule, signed weight, exact delay steps, seed."""

    index: int
    id: str
    source: str
    target: str
    weight: float
    sign: NeuronType
    rule: ConnectionRule
    probability: float
    delay_ms: float
    delay_steps: int
    seed: int
    seed_source: SeedSource
    autapses: bool

    def to_public_dict(self) -> dict[str, Any]:
        """Return the path-free projection block."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "sign": self.sign,
            "rule": self.rule,
            "probability": self.probability,
            "delay_ms": self.delay_ms,
            "delay_steps": self.delay_steps,
            "seed": self.seed,
            "seed_source": self.seed_source,
            "autapses": self.autapses,
        }


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """The resolved, digest-bound specification of one graph run."""

    populations: tuple[PopulationSpec, ...]
    projections: tuple[ProjectionSpec, ...]
    dt: float
    duration_ms: float
    n_steps: int
    seed: int
    n_neurons: int

    @property
    def neuron_steps(self) -> int:
        """Total neuron updates the run performs (``n_neurons * n_steps``)."""
        return self.n_neurons * self.n_steps

    def population(self, population_id: str) -> PopulationSpec:
        """Return the population with ``population_id``."""
        for population in self.populations:
            if population.id == population_id:
                return population
        raise KeyError(population_id)

    def to_public_dict(self) -> dict[str, Any]:
        """Return the path-free specification with its ``graph_sha256`` digest."""
        body: dict[str, Any] = {
            "schema_version": GRAPH_SPEC_SCHEMA_VERSION,
            "request_schema_version": GRAPH_SCHEMA_VERSION,
            "dt": self.dt,
            "time_unit": "ms",
            "duration_requested_ms": self.duration_ms,
            "duration_effective_ms": self.n_steps * self.dt,
            "n_steps": self.n_steps,
            "seed": self.seed,
            "n_neurons": self.n_neurons,
            "neuron_steps": self.neuron_steps,
            "limits": {
                "max_neurons": MAX_NEURONS,
                "max_steps": MAX_GRAPH_STEPS,
                "max_neuron_steps": MAX_NEURON_STEPS,
            },
            "populations": [population.to_public_dict() for population in self.populations],
            "projections": [projection.to_public_dict() for projection in self.projections],
        }
        body["graph_sha256"] = _sha256_json(body)
        return body


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, default=str, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derived_seed(graph_seed: int, kind: int, index: int) -> int:
    """Return the deterministic 32-bit seed of element ``index`` of ``kind``.

    The seed is spawned from ``numpy.random.SeedSequence(graph_seed,
    spawn_key=(kind, index))`` so projections and drives draw from independent
    streams that a direct public-runtime script can reproduce from the graph
    seed alone.
    """
    sequence = np.random.SeedSequence(graph_seed, spawn_key=(kind, index))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _finite(value: object) -> bool:
    return _as_float(value) is not None


def _as_float(value: object) -> float | None:
    """Return ``value`` as a float when it is a finite real number, else ``None``."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _as_int(value: object) -> int | None:
    """Return ``value`` as an int when it is an integer (not a bool), else ``None``."""
    if not isinstance(value, Integral) or isinstance(value, bool):
        return None
    return int(value)


class _Collector:
    """Accumulate validation issues in request order."""

    def __init__(self) -> None:
        self.issues: list[GraphIssue] = []

    def add(self, field: str, message: str) -> None:
        self.issues.append(GraphIssue(field=field, message=message))

    def __bool__(self) -> bool:
        return bool(self.issues)


def _unknown_fields(
    collector: _Collector,
    mapping: Mapping[str, Any],
    allowed: frozenset[str],
    field: str,
    label: str,
) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        collector.add(field, f"{label} has unknown fields: {', '.join(unknown)}")


def _validate_timing(
    collector: _Collector, graph: Mapping[str, Any]
) -> tuple[float | None, float | None, int | None]:
    duration_raw = graph.get("duration", DEFAULT_DURATION_MS)
    dt_raw = graph.get("dt", DEFAULT_DT_MS)
    seed_raw = graph.get("seed", DEFAULT_SEED)
    duration: float | None = None
    dt: float | None = None
    seed: int | None = None
    duration = _as_float(duration_raw)
    if duration is None or duration <= 0.0:
        collector.add("duration", "Graph duration must be a positive finite number of milliseconds")
        duration = None
    dt = _as_float(dt_raw)
    if dt is None or dt <= 0.0:
        collector.add("dt", "Graph dt must be a positive finite number of milliseconds")
        dt = None
    seed = _as_int(seed_raw)
    if seed is None or seed < 0 or seed >= 2**32:
        collector.add("seed", "Graph seed must be an integer in [0, 2^32)")
        seed = None
    return duration, dt, seed


def _validate_drive(
    collector: _Collector,
    raw: object,
    *,
    field: str,
    label: str,
    graph_seed: int | None,
    index: int,
) -> DriveSpec | None:
    if raw is None:
        return DriveSpec("none", None, None, None, None, "none")
    if not isinstance(raw, Mapping):
        collector.add(field, f"{label} drive must be an object")
        return None
    _unknown_fields(collector, raw, _DRIVE_FIELDS, field, f"{label} drive")
    kind = raw.get("kind", "none")
    if not isinstance(kind, str) or kind not in DRIVE_KINDS:
        collector.add(f"{field}.kind", f"{label} drive kind must be one of {DRIVE_KINDS}")
        return None
    if kind == "none":
        extra = sorted(key for key in raw if key != "kind")
        if extra:
            collector.add(field, f"{label} drive of kind none carries fields: {', '.join(extra)}")
            return None
        return DriveSpec("none", None, None, None, None, "none")
    if kind == "constant":
        current = _as_float(raw.get("current"))
        if current is None:
            collector.add(f"{field}.current", f"{label} constant drive needs a finite current")
            return None
        extra = sorted(key for key in raw if key not in ("kind", "current"))
        if extra:
            collector.add(field, f"{label} constant drive carries fields: {', '.join(extra)}")
            return None
        return DriveSpec("constant", current, None, None, None, "none")
    rate = _as_float(raw.get("rate_hz"))
    weight = _as_float(raw.get("weight"))
    if rate is None or rate <= 0.0:
        collector.add(f"{field}.rate_hz", f"{label} poisson drive needs a positive finite rate_hz")
        rate = None
    if weight is None or weight == 0.0:
        collector.add(f"{field}.weight", f"{label} poisson drive needs a finite non-zero weight")
        weight = None
    seed_raw = raw.get("seed")
    seed: int | None = None
    seed_source: SeedSource = "derived"
    if seed_raw is not None:
        seed = _as_int(seed_raw)
        if seed is None or seed < 0 or seed >= 2**32:
            collector.add(
                f"{field}.seed", f"{label} poisson drive seed must be an integer in [0, 2^32)"
            )
            return None
        seed_source = "request"
    if rate is None or weight is None:
        return None
    if seed is None and graph_seed is not None:
        seed = derived_seed(graph_seed, _SEED_SPAWN_DRIVE, index)
    return DriveSpec("poisson", None, rate, weight, seed, seed_source)


def _model_admissible(model: str, cls: type) -> str | None:
    """Return why ``model`` cannot form a population, or ``None`` when it can."""
    drive = model_drive_contract(model, cls)
    if drive.kind == "int":
        return (
            "integer-drive model: the public Network injects float currents and would truncate them"
        )
    contracts = model_parameter_contracts(cls)
    if "seed" in contracts.overridable:
        return (
            "the model draws randomness from a seed field and every neuron of a "
            "population would share it (perfectly correlated noise); not admitted"
        )
    return None


def _validate_populations(
    collector: _Collector,
    populations: Sequence[object],
    *,
    dt: float | None,
    graph_seed: int | None,
) -> tuple[list[PopulationSpec], set[str], dict[str, NeuronType]]:
    specs: list[PopulationSpec] = []
    ids: set[str] = set()
    types: dict[str, NeuronType] = {}
    offset = 0
    for index, raw in enumerate(populations):
        field = f"populations[{index}]"
        if not isinstance(raw, Mapping):
            collector.add(field, f"Population {index} must be an object")
            continue
        pop_id = raw.get("id")
        if not isinstance(pop_id, str) or not pop_id:
            collector.add(f"{field}.id", f"Population {index} id must be a non-empty string")
            continue
        label_text = pop_id
        if pop_id in ids:
            collector.add(f"{field}.id", f"Population {label_text} id is duplicated")
            continue
        ids.add(pop_id)
        _unknown_fields(collector, raw, _POPULATION_FIELDS, field, f"Population {label_text}")
        kind = raw.get("type", "population")
        if kind != "population":
            collector.add(f"{field}.type", f"Population {label_text} type must be 'population'")
        label = raw.get("label", pop_id)
        if not isinstance(label, str):
            collector.add(f"{field}.label", f"Population {label_text} label must be a string")
            label = pop_id
        neuron_type = raw.get("neuron_type")
        if not isinstance(neuron_type, str) or neuron_type not in NEURON_TYPES:
            collector.add(
                f"{field}.neuron_type",
                f"Population {label_text} neuron_type must be one of {NEURON_TYPES}",
            )
            neuron_type = None
        else:
            types[pop_id] = neuron_type  # type: ignore[assignment]
        count = _as_int(raw.get("count"))
        if count is None:
            collector.add(
                f"{field}.count", f"Population {label_text} count must be a positive integer"
            )
        elif count < 1:
            collector.add(f"{field}.count", f"Population {label_text} count must be at least 1")
            count = None
        model = raw.get("model", DEFAULT_MODEL)
        inputs: ModelRunInputs | None = None
        if not isinstance(model, str) or model not in _CLASS_TO_MODULE:
            collector.add(
                f"{field}.model",
                f"Population {label_text} model {model!r} is not a catalogue model",
            )
        else:
            params = raw.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, Mapping):
                collector.add(
                    f"{field}.params", f"Population {label_text} params must be an object"
                )
            elif dt is not None:
                try:
                    inputs = resolve_model_run_inputs(model, params, dt)
                except ModelInputError as exc:
                    collector.add(
                        f"{field}.{exc.field}",
                        f"Population {label_text} {exc.field}: {exc.reason}",
                    )
                else:
                    reason = _model_admissible(model, inputs.cls)
                    if reason is not None:
                        collector.add(
                            f"{field}.model", f"Population {label_text} model {model}: {reason}"
                        )
                        inputs = None
                    else:
                        # Each field can be in range while the combination is not a
                        # model: a threshold below the resting potential passes every
                        # per-field contract. Building one neuron asks the model itself.
                        try:
                            inputs.instantiate()
                        except ModelInputError as exc:
                            collector.add(
                                f"{field}.params",
                                f"Population {label_text} parameters: {exc.reason}",
                            )
                            inputs = None
        drive = _validate_drive(
            collector,
            raw.get("drive"),
            field=f"{field}.drive",
            label=f"Population {label_text}",
            graph_seed=graph_seed,
            index=index,
        )
        if (
            count is None
            or inputs is None
            or neuron_type is None
            or drive is None
            or not isinstance(model, str)
        ):
            continue
        specs.append(
            PopulationSpec(
                index=index,
                id=pop_id,
                label=label,
                model=model,
                count=count,
                neuron_type=neuron_type,  # type: ignore[arg-type]
                inputs=inputs,
                drive=drive,
                offset=offset,
            )
        )
        offset += count
    return specs, ids, types


def _validate_projections(
    collector: _Collector,
    projections: Sequence[object],
    *,
    ids: set[str],
    types: Mapping[str, NeuronType],
    dt: float | None,
    graph_seed: int | None,
) -> list[ProjectionSpec]:
    specs: list[ProjectionSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(projections):
        field = f"projections[{index}]"
        if not isinstance(raw, Mapping):
            collector.add(field, f"Projection {index} must be an object")
            continue
        proj_id = raw.get("id")
        label = proj_id if isinstance(proj_id, str) and proj_id else str(index)
        ok = True
        if not isinstance(proj_id, str) or not proj_id:
            collector.add(f"{field}.id", f"Projection {label} id must be a non-empty string")
            ok = False
        elif proj_id in seen:
            collector.add(f"{field}.id", f"Projection {label} id is duplicated")
            ok = False
        else:
            seen.add(proj_id)
        _unknown_fields(collector, raw, _PROJECTION_FIELDS, field, f"Projection {label}")
        source = raw.get("source")
        target = raw.get("target")
        if not isinstance(source, str) or not source:
            collector.add(
                f"{field}.source", f"Projection {label} source must be a non-empty string"
            )
            ok = False
        elif source not in ids:
            collector.add(f"{field}.source", f"Projection {label} source {source} not found")
            ok = False
        if not isinstance(target, str) or not target:
            collector.add(
                f"{field}.target", f"Projection {label} target must be a non-empty string"
            )
            ok = False
        elif target not in ids:
            collector.add(f"{field}.target", f"Projection {label} target {target} not found")
            ok = False
        weight_raw = raw.get("weight")
        weight = _as_float(weight_raw)
        if not _is_number(weight_raw):
            collector.add(f"{field}.weight", f"Projection {label} weight must be numeric")
        elif weight is None:
            collector.add(f"{field}.weight", f"Projection {label} weight must be finite")
        elif weight == 0.0:
            collector.add(f"{field}.weight", f"Projection {label} has zero weight")
            weight = None
        sign: NeuronType | None = None
        if weight is not None:
            sign = "excitatory" if weight > 0.0 else "inhibitory"
            source_type = types.get(source) if isinstance(source, str) else None
            if source_type is not None and source_type != sign:
                needed = "negative" if source_type == "inhibitory" else "positive"
                collector.add(
                    f"{field}.weight",
                    f"Projection {label} weight {weight:g} conflicts with the {source_type} "
                    f"source population {source}; {source_type} sources need a {needed} weight",
                )
                ok = False
        rule = raw.get("rule", "random")
        if not isinstance(rule, str) or rule not in CONNECTION_RULES:
            collector.add(
                f"{field}.rule", f"Projection {label} rule must be one of {CONNECTION_RULES}"
            )
            ok = False
            rule = None
        probability_raw = raw.get("probability")
        probability = _as_float(probability_raw)
        if rule == "all_to_all":
            if probability_raw is None or probability == 1.0:
                probability = 1.0
            else:
                collector.add(
                    f"{field}.probability",
                    f"Projection {label} rule all_to_all conflicts with probability "
                    f"{probability_raw!r}; omit it or set 1",
                )
                ok = False
        elif rule == "random":
            if probability_raw is None:
                collector.add(
                    f"{field}.probability",
                    f"Projection {label} rule random needs an explicit probability",
                )
                ok = False
            elif not _is_number(probability_raw):
                collector.add(
                    f"{field}.probability", f"Projection {label} probability must be numeric"
                )
                ok = False
            elif probability is None:
                collector.add(
                    f"{field}.probability", f"Projection {label} probability must be finite"
                )
                ok = False
            elif probability <= 0.0 or probability > 1.0:
                collector.add(
                    f"{field}.probability",
                    f"Projection {label} probability out of range (0, 1]",
                )
                ok = False
                probability = None
        else:
            probability = None
        delay = _as_float(raw.get("delay", 0.0))
        delay_ms: float | None = None
        delay_steps: int | None = None
        if delay is None or delay < 0.0:
            collector.add(
                f"{field}.delay",
                f"Projection {label} delay must be a finite non-negative number of milliseconds",
            )
            ok = False
        elif dt is not None:
            ratio = delay / dt
            steps = int(round(ratio))
            if abs(ratio - steps) > _DELAY_STEP_TOLERANCE * max(1.0, abs(ratio)):
                collector.add(
                    f"{field}.delay",
                    f"Projection {label} delay {delay:g} ms is not a whole number "
                    f"of {dt:g} ms steps; the public Network rounds delays, this graph does not",
                )
                ok = False
            else:
                delay_ms = delay
                delay_steps = steps
        seed_raw = raw.get("seed")
        seed: int | None = None
        seed_source: SeedSource = "derived"
        if seed_raw is not None:
            seed = _as_int(seed_raw)
            if seed is None or seed < 0 or seed >= 2**32:
                collector.add(
                    f"{field}.seed", f"Projection {label} seed must be an integer in [0, 2^32)"
                )
                ok = False
                seed = None
            else:
                seed_source = "request"
        autapses_raw = raw.get("autapses", False)
        if not isinstance(autapses_raw, bool):
            collector.add(f"{field}.autapses", f"Projection {label} autapses must be a boolean")
            ok = False
        elif autapses_raw and source != target:
            collector.add(
                f"{field}.autapses",
                f"Projection {label} autapses only apply to a self-projection",
            )
            ok = False
        if (
            not ok
            or weight is None
            or sign is None
            or rule is None
            or probability is None
            or delay_ms is None
            or delay_steps is None
            or not isinstance(proj_id, str)
            or not isinstance(source, str)
            or not isinstance(target, str)
        ):
            continue
        if seed is None:
            if graph_seed is None:
                continue
            seed = derived_seed(graph_seed, _SEED_SPAWN_PROJECTION, index)
        specs.append(
            ProjectionSpec(
                index=index,
                id=proj_id,
                source=source,
                target=target,
                weight=weight,
                sign=sign,
                rule=rule,
                probability=probability,
                delay_ms=delay_ms,
                delay_steps=delay_steps,
                seed=seed,
                seed_source=seed_source,
                autapses=bool(autapses_raw),
            )
        )
    return specs


def _analyse(graph: object) -> tuple[_Collector, GraphSpec | None]:
    collector = _Collector()
    if not isinstance(graph, Mapping):
        collector.add("graph", "Network graph must be an object")
        return collector, None
    _unknown_fields(collector, graph, _GRAPH_FIELDS, "graph", "Network graph")
    duration, dt, seed = _validate_timing(collector, graph)

    populations = graph.get("populations", [])
    projections = graph.get("projections", [])
    if not isinstance(populations, list):
        collector.add("populations", "Network populations must be a list")
        populations = []
    if not isinstance(projections, list):
        collector.add("projections", "Network projections must be a list")
        projections = []
    if not populations:
        collector.add("populations", "Network has no populations")
        return collector, None

    population_specs, ids, types = _validate_populations(
        collector, populations, dt=dt, graph_seed=seed
    )
    if not population_specs and not ids:
        collector.add("populations", "Network has no valid populations")
    projection_specs = _validate_projections(
        collector, projections, ids=ids, types=types, dt=dt, graph_seed=seed
    )

    n_neurons = sum(population.count for population in population_specs)
    if n_neurons > MAX_NEURONS:
        collector.add(
            "populations",
            f"Total neuron count {n_neurons} exceeds {MAX_NEURONS} limit for browser simulation",
        )
    n_steps: int | None = None
    if duration is not None and dt is not None:
        requested = int(duration / dt)
        if requested < 1:
            collector.add(
                "duration",
                f"duration {duration:g} ms with dt {dt:g} ms yields no complete step",
            )
        elif requested > MAX_GRAPH_STEPS:
            collector.add(
                "duration",
                f"{requested} steps exceed the synchronous limit of {MAX_GRAPH_STEPS}; "
                "the run is not shortened, reduce duration or dt",
            )
        else:
            n_steps = requested
    if n_steps is not None and n_neurons * n_steps > MAX_NEURON_STEPS:
        collector.add(
            "duration",
            f"{n_neurons} neurons over {n_steps} steps are {n_neurons * n_steps} neuron-steps, "
            f"above the synchronous limit of {MAX_NEURON_STEPS}; the run is not shortened",
        )

    if collector or dt is None or duration is None or seed is None or n_steps is None:
        return collector, None
    if len(population_specs) != len(populations) or len(projection_specs) != len(projections):
        return collector, None
    spec = GraphSpec(
        populations=tuple(population_specs),
        projections=tuple(projection_specs),
        dt=dt,
        duration_ms=duration,
        n_steps=n_steps,
        seed=seed,
        n_neurons=n_neurons,
    )
    return collector, spec


def graph_issues(graph: object) -> list[GraphIssue]:
    """Return every validation issue of ``graph`` in request order (empty = valid)."""
    collector, _spec = _analyse(graph)
    return list(collector.issues)


def validate_graph(graph: object) -> list[str]:
    """Validate a network graph and return its error messages (empty = valid).

    Structural errors (shapes, ids, endpoints), model-contract errors (unknown
    model or parameter, timestep the model cannot take, inadmissible drive or
    randomness), projection errors (sign against the source type, rule and
    probability conflicts, delays that are not whole timesteps, seeds,
    autapses) and budget errors (neuron cap, step cap, neuron-step cap) are all
    reported together.
    """
    return [issue.message for issue in graph_issues(graph)]


def resolve_graph(graph: object) -> GraphSpec:
    """Resolve ``graph`` into its executable specification.

    Raises
    ------
    GraphRejected
        With the field and message of the first validation issue.
    """
    collector, spec = _analyse(graph)
    if spec is None:
        first = collector.issues[0] if collector.issues else GraphIssue("graph", "invalid graph")
        raise GraphRejected(field=first.field, reason=first.message)
    return spec


__all__ = [
    "CONNECTION_RULES",
    "DEFAULT_DT_MS",
    "DEFAULT_DURATION_MS",
    "DEFAULT_MODEL",
    "DEFAULT_SEED",
    "DRIVE_KINDS",
    "GRAPH_SCHEMA_VERSION",
    "GRAPH_SPEC_SCHEMA_VERSION",
    "MAX_GRAPH_STEPS",
    "MAX_NEURONS",
    "MAX_NEURON_STEPS",
    "NEURON_TYPES",
    "DriveSpec",
    "GraphIssue",
    "GraphRejected",
    "GraphSpec",
    "PopulationSpec",
    "ProjectionSpec",
    "derived_seed",
    "graph_issues",
    "resolve_graph",
    "validate_graph",
]
