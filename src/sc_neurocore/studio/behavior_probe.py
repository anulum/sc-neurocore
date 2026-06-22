# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Measured behaviour probe for the model catalogue

"""Measured behaviour facet: probe a model across a current sweep.

Unlike :mod:`sc_neurocore.studio.model_scan` (one current, fail-closed), this
probe sweeps each model across a scale-robust ladder of constant-current drives
so a single classification at one operating point cannot stand in for the
model's behavioural envelope. The ladder spans several decades because the
catalogue mixes unit conventions — physical millivolt/nanoamp models, normalised
dimensionless maps, and integer fixed-point hardware models — and no single
current is meaningful across all of them.

Every observation is taken twice from an identical seed; a tag is asserted only
from an observation that reproduced. A model whose spike train differs between
two identical runs is flagged ``stochastic`` and carries only the sign-robust
excitability verdict, so the derived tags are reproducible by construction. The
probe is resilient per ``(model, current)``: a model that cannot be driven by a
constant current (a non-standard ``step`` signature, or an internal stability
guard) records the failure for that point and still yields an honest profile
rather than aborting the sweep.
"""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from sc_neurocore.neurons.behavior_taxonomy import validate_behavior_tags
from sc_neurocore.studio.codegen import classify_firing_pattern
from sc_neurocore.studio.models import list_models, simulate_model

BEHAVIOR_PROBE_SCHEMA_VERSION = "studio.behavior-probe.v1"

#: Scale-robust constant-current ladder. ``0.0`` establishes the resting
#: response; the geometric rise from 1 to 1024 spans the normalised (~1-16),
#: physical (~64-256) and strong-drive (~1024) regimes in one sweep.
BEHAVIOR_SWEEP_CURRENTS: tuple[float, ...] = (0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)

#: Probe window. Long enough to resolve adaptation and burst-pause structure at
#: the rates these drives produce, short enough to keep the full sweep tractable.
BEHAVIOR_PROBE_DURATION_MS = 200.0

#: Fixed seed so the two reproducibility runs start from an identical RNG state.
BEHAVIOR_PROBE_SEED = 20260622

#: Recorded measurement the fast catalogue gate compares descriptors against.
BEHAVIOR_EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "neurons" / "behavior_evidence.json"

#: A non-decreasing f-I curve tolerates this much rate dip (Hz) between adjacent
#: drives before it is no longer considered monotone.
_RATE_MONOTONE_TOLERANCE_HZ = 1.0

#: The f-I curve must rise by at least this much (Hz) to count as rate-coded.
_RATE_RISE_EPSILON_HZ = 1.0

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class BehaviorObservation:
    """One reproducibility-checked observation at a single drive current."""

    current: float
    pattern: str
    rate_hz: float
    spike_count: int
    reproducible: bool
    error: str | None = None

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible observation."""

        return {
            "current": self.current,
            "error": self.error,
            "pattern": self.pattern,
            "rate_hz": self.rate_hz,
            "reproducible": self.reproducible,
            "spike_count": self.spike_count,
        }


@dataclass(frozen=True, slots=True)
class ModelBehaviorProfile:
    """The measured behavioural envelope of one model over the current sweep."""

    name: str
    duration_ms: float
    currents: tuple[float, ...]
    observations: tuple[BehaviorObservation, ...]
    behavior_tags: tuple[str, ...]
    stochastic: bool
    drivable: bool
    input_sha256: str
    result_sha256: str
    schema_version: str = BEHAVIOR_PROBE_SCHEMA_VERSION
    evidence_classification: str = "measured"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible behaviour profile."""

        return {
            "behavior_tags": list(self.behavior_tags),
            "currents": list(self.currents),
            "drivable": self.drivable,
            "duration_ms": self.duration_ms,
            "evidence_classification": self.evidence_classification,
            "input_sha256": self.input_sha256,
            "name": self.name,
            "observations": [obs.to_public_dict() for obs in self.observations],
            "result_sha256": self.result_sha256,
            "schema_version": self.schema_version,
            "stochastic": self.stochastic,
        }


def derive_behavior_tags(
    observations: Sequence[BehaviorObservation], *, stochastic: bool
) -> tuple[str, ...]:
    """Derive the behaviour tags from a model's sweep observations.

    Pure and side-effect free, so the tag logic is testable without running any
    simulation. Excitability is read from the sign of the response (robust even
    for stochastic models); the firing-pattern tags are read only from
    reproducible observations, and are withheld entirely for a stochastic model.

    Parameters
    ----------
    observations:
        The per-current observations, in any order.
    stochastic:
        Whether the model's spike train failed to reproduce at some drive.

    Returns
    -------
    tuple[str, ...]
        The validated, sorted behaviour tags. Empty when no current could be
        driven (an honest "no measured behaviour").
    """

    usable = [obs for obs in observations if obs.error is None]
    if not usable:
        return ()

    tags: set[str] = set()
    spiking = [obs for obs in usable if obs.spike_count >= 1]
    tags.add("excitable" if spiking else "quiescent")

    if stochastic:
        tags.add("stochastic")
        return validate_behavior_tags(tags)

    reproducible_spiking = [obs for obs in spiking if obs.reproducible]
    patterns = {obs.pattern for obs in reproducible_spiking}
    for pattern in ("tonic", "adapting", "bursting", "irregular", "chaotic"):
        if pattern in patterns:
            tags.add(pattern)
    if "single_spike" in patterns:
        tags.add("phasic")
    if _is_rate_coded([obs for obs in usable if obs.reproducible]):
        tags.add("rate-coded")
    return validate_behavior_tags(tags)


def _is_rate_coded(observations: Sequence[BehaviorObservation]) -> bool:
    """Whether the firing rate rises monotonically with the drive current."""

    ordered = sorted(observations, key=lambda obs: obs.current)
    rates = [obs.rate_hz for obs in ordered]
    if len(rates) < 2:
        return False
    non_decreasing = all(
        later >= earlier - _RATE_MONOTONE_TOLERANCE_HZ
        for earlier, later in zip(rates, rates[1:], strict=False)
    )
    rose = max(rates) - min(rates) > _RATE_RISE_EPSILON_HZ
    return non_decreasing and rose


def probe_model_behavior(
    name: str,
    *,
    currents: Sequence[float] = BEHAVIOR_SWEEP_CURRENTS,
    duration: float = BEHAVIOR_PROBE_DURATION_MS,
    seed: int = BEHAVIOR_PROBE_SEED,
) -> ModelBehaviorProfile:
    """Probe one model across the current sweep and derive its behaviour tags.

    Never raises for a model that cannot be driven: each failed drive is
    recorded as an error observation and the profile is still returned.
    """

    sweep = tuple(float(current) for current in currents)
    observations = tuple(
        _observe(name=name, current=current, duration=float(duration), seed=seed)
        for current in sweep
    )
    stochastic = any(obs.error is None and not obs.reproducible for obs in observations)
    tags = derive_behavior_tags(observations, stochastic=stochastic)
    drivable = any(obs.error is None for obs in observations)
    input_payload: dict[str, JsonValue] = {
        "currents": list(sweep),
        "duration": float(duration),
        "name": name,
        "seed": seed,
    }
    result_payload: dict[str, JsonValue] = {
        "behavior_tags": list(tags),
        "observations": [obs.to_public_dict() for obs in observations],
        "stochastic": stochastic,
    }
    return ModelBehaviorProfile(
        name=name,
        duration_ms=float(duration),
        currents=sweep,
        observations=observations,
        behavior_tags=tags,
        stochastic=stochastic,
        drivable=drivable,
        input_sha256=_sha256_json(input_payload),
        result_sha256=_sha256_json(result_payload),
    )


def _observe(*, name: str, current: float, duration: float, seed: int) -> BehaviorObservation:
    """Take one reproducibility-checked observation at a single drive."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            first = _seeded_simulation(name=name, current=current, duration=duration, seed=seed)
            second = _seeded_simulation(name=name, current=current, duration=duration, seed=seed)
        except Exception as exc:  # a model that cannot be driven by a constant current
            return BehaviorObservation(
                current=current,
                pattern="error",
                rate_hz=0.0,
                spike_count=0,
                reproducible=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        pattern = classify_firing_pattern(first["spikes"], first["n_steps"], first["dt"])
        return BehaviorObservation(
            current=current,
            pattern=str(pattern["pattern"]),
            rate_hz=float(pattern.get("rate_hz", 0.0)),
            spike_count=int(first["spike_count"]),
            reproducible=list(first["spikes"]) == list(second["spikes"]),
            error=None,
        )


def _seeded_simulation(*, name: str, current: float, duration: float, seed: int) -> dict[str, Any]:
    """Simulate a model from a fixed RNG state."""

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    # Force the Python reference model: the facet must characterise the canonical
    # model, not whichever backend is loaded (the Rust path can be deterministic
    # where the Python reference is not, which would make the verdict environment-
    # dependent and the stochastic flag unreliable).
    return simulate_model(name, duration=duration, current=current, use_fast_path=False)


def probe_all_models(
    *,
    currents: Sequence[float] = BEHAVIOR_SWEEP_CURRENTS,
    duration: float = BEHAVIOR_PROBE_DURATION_MS,
    seed: int = BEHAVIOR_PROBE_SEED,
    names: Iterable[str] | None = None,
) -> dict[str, JsonValue]:
    """Probe every catalogue model and return a recordable evidence manifest.

    The manifest is the source the fast catalogue gate compares descriptors
    against: a per-model tag set plus the sweep configuration and digests, so a
    committed ``behavior_tags`` field can be checked for equality with the
    measurement without re-running any simulation.
    """

    sweep = tuple(float(current) for current in currents)
    model_names = list(names) if names is not None else [str(m["name"]) for m in list_models()]
    profiles = [
        probe_model_behavior(name, currents=sweep, duration=float(duration), seed=seed)
        for name in model_names
    ]
    models: dict[str, JsonValue] = {
        profile.name: {
            "behavior_tags": list(profile.behavior_tags),
            "drivable": profile.drivable,
            "result_sha256": profile.result_sha256,
            "stochastic": profile.stochastic,
        }
        for profile in profiles
    }
    sweep_config: dict[str, JsonValue] = {
        "currents": list(sweep),
        "duration_ms": float(duration),
        "seed": seed,
    }
    return {
        "schema_version": BEHAVIOR_PROBE_SCHEMA_VERSION,
        "sweep": sweep_config,
        "sweep_sha256": _sha256_json(sweep_config),
        "models": dict(sorted(models.items())),
        "result_sha256": _sha256_json({"models": models}),
    }


def _sha256_json(payload: Mapping[str, JsonValue]) -> str:
    """Return a SHA-256 digest over canonical JSON."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_behavior_evidence() -> dict[str, Any]:
    """Load the recorded behaviour evidence manifest.

    Raises
    ------
    FileNotFoundError
        If the manifest has not been generated.
    """

    evidence: dict[str, Any] = json.loads(BEHAVIOR_EVIDENCE_PATH.read_text(encoding="utf-8"))
    return evidence


def behavior_tags_for(name: str, evidence: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return the recorded behaviour tags for a model (empty if unrecorded)."""

    manifest = evidence if evidence is not None else load_behavior_evidence()
    models = manifest.get("models", {})
    entry = models.get(name)
    if not isinstance(entry, Mapping):
        return ()
    return tuple(str(tag) for tag in entry.get("behavior_tags", ()))


__all__ = [
    "BEHAVIOR_EVIDENCE_PATH",
    "BEHAVIOR_PROBE_DURATION_MS",
    "BEHAVIOR_PROBE_SCHEMA_VERSION",
    "BEHAVIOR_PROBE_SEED",
    "BEHAVIOR_SWEEP_CURRENTS",
    "BehaviorObservation",
    "ModelBehaviorProfile",
    "behavior_tags_for",
    "derive_behavior_tags",
    "load_behavior_evidence",
    "probe_all_models",
    "probe_model_behavior",
]
