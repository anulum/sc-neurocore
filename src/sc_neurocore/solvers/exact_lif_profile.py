# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Versioned exact-current LIF execution contract

"""Digest-bound, stateful execution for the exact-current SC LIF profile.

The profile binds the existing :class:`ExactLIFSolver` implementation rather
than introducing another neuron model. It makes the solver, units, event
ordering, reset boundary and numerical behaviour explicit for consumers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from .exact_lif import ExactLIFSolver

PROFILE_SCHEMA: Final = "sc-neurocore.exact-current-lif-profile.v1"
STATE_SCHEMA: Final = "sc-neurocore.exact-current-lif-state.v1"
PACKET_SCHEMA: Final = "sc-neurocore.exact-current-lif-packet.v1"
PROFILE_NAME: Final = "sc_exact_current_hard_reset_lif_v1"
MODEL_SOURCE: Final = "sc_neurocore.solvers.exact_lif:ExactLIFSolver"
MODEL_SOURCE_PATH: Final = "solvers/exact_lif.py"
MODEL_SOURCE_SHA256: Final = "064be334316184e50a85fb82b1a804cdf1342bb927c39588b4d4105c7a087762"
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_MAX_EVENTS_PER_TICK: Final = 1_000_000


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_json(name: str, serialized: str | bytes) -> Any:
    try:
        return json.loads(serialized, object_pairs_hook=_object_without_duplicates)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real value")
    return result


def _positive(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _sum_currents(currents: tuple[float, ...]) -> float:
    try:
        total = math.fsum(currents)
    except OverflowError as exc:
        raise ValueError("summed current must remain finite") from exc
    if not math.isfinite(total):
        raise ValueError("summed current must remain finite")
    return total


def _exact_keys(name: str, payload: Mapping[str, Any], expected: set[str]) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ValueError(f"{name} fields mismatch; missing={missing}, unknown={unknown}")


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class ExactCurrentLIFProfile:
    """Immutable parameters and semantics for one exact-current LIF instance."""

    tau_ms: float = 20.0
    v_rest: float = -65.0
    v_threshold: float = -50.0
    v_reset: float = -65.0
    resistance: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau_ms", _positive("tau_ms", self.tau_ms))
        object.__setattr__(self, "v_rest", _finite("v_rest", self.v_rest))
        object.__setattr__(self, "v_threshold", _finite("v_threshold", self.v_threshold))
        object.__setattr__(self, "v_reset", _finite("v_reset", self.v_reset))
        object.__setattr__(self, "resistance", _positive("resistance", self.resistance))
        if self.v_threshold <= self.v_rest:
            raise ValueError("v_threshold must be greater than v_rest")
        if self.v_reset >= self.v_threshold:
            raise ValueError("v_reset must be below v_threshold")

    def to_payload(self) -> dict[str, Any]:
        """Return the complete canonical profile payload."""
        return {
            "schema": PROFILE_SCHEMA,
            "profile": PROFILE_NAME,
            "model": {
                "family": "current_based_leaky_integrate_and_fire",
                "equation": "tau_ms*dV/dt=-(V-v_rest)+resistance*I",
                "source_identity": MODEL_SOURCE,
                "source_path": MODEL_SOURCE_PATH,
                "source_sha256": MODEL_SOURCE_SHA256,
                "source_scope": "Rotter-Diesmann exact integration of the SC current-based hard-reset LIF equation",
            },
            "parameters": {
                "tau_ms": self.tau_ms,
                "v_rest": self.v_rest,
                "v_threshold": self.v_threshold,
                "v_reset": self.v_reset,
                "resistance": self.resistance,
            },
            "domains": {
                "tau_ms": "finite > 0",
                "resistance": "finite > 0",
                "voltage": "finite binary64; v_threshold > v_rest and v_reset",
                "current_contribution": "finite binary64",
                "summed_current": "finite binary64",
                "tick_duration_ms": "finite > 0",
                "shot_time_ms": "finite >= 0",
            },
            "units": {
                "time": "ms",
                "voltage": "normalized_voltage",
                "current": "normalized_current",
                "resistance": "normalized_resistance",
            },
            "input": {
                "family": "piecewise_constant_current",
                "delivery": "all simultaneous contributions summed at tick start",
                "delay_ms": 0.0,
                "timestamp_domain": "float64_ms_relative_to_shot",
            },
            "solver": {
                "identity": "closed_form_piecewise_constant_event_driven",
                "threshold_crossing": "analytical_within_tick",
                "absolute_tolerance": 0.0,
                "relative_tolerance": 0.0,
                "singular_policy": "tau_ms and resistance must be finite and positive",
            },
            "events": {
                "order": [
                    "sum_inputs",
                    "evolve",
                    "detect_threshold_ge",
                    "emit",
                    "hard_reset",
                    "continue_tick",
                ],
                "threshold_comparison": "greater_than_or_equal",
                "timestamp": "analytical_crossing_time",
                "tie_break": "single neuron; input contributions are commutatively summed before evolution",
            },
            "state": {
                "variables": ["voltage", "time_ms", "shot_id", "reset_epoch"],
                "serialization_schema": STATE_SCHEMA,
                "trace_phases": ["initial", "threshold", "reset", "tick_end"],
                "initial_voltage": self.v_rest,
                "persistence": "across execute calls",
                "reset_boundary": "explicit shot reset only",
                "refractory_ms": 0.0,
                "refractory_input_policy": "not_applicable_zero_duration",
            },
            "numeric": {
                "representation": "IEEE-754 binary64",
                "rounding": "round_to_nearest_ties_to_even",
                "overflow": "fail_closed_non_finite",
                "saturation": "none",
                "backend": "python_reference",
            },
            "backend_capabilities": {
                "python_reference": "required_exact",
                "native_and_fixed_point": "separate_digest_bound_profiles_required",
            },
            "rng": {"algorithm": "none", "seed": None, "state": None},
            "compatibility": {
                "unknown_fields": "reject",
                "unknown_schema": "reject",
                "digest_mismatch": "reject",
            },
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 of canonical profile JSON."""
        return hashlib.sha256(_canonical_json(self.to_payload()).encode()).hexdigest()

    def to_json(self) -> str:
        """Serialize the profile canonically."""
        return _canonical_json(self.to_payload())

    def verify_source_binding(self) -> None:
        """Fail closed if the shipped model source differs from this profile."""
        source_path = Path(__file__).resolve().parents[1] / MODEL_SOURCE_PATH
        observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if observed != MODEL_SOURCE_SHA256:
            raise ValueError(
                f"model source digest mismatch: expected {MODEL_SOURCE_SHA256}, observed {observed}"
            )

    @classmethod
    def from_json(cls, serialized: str | bytes) -> ExactCurrentLIFProfile:
        """Parse strict canonical semantics; reject versions, fields and units."""
        raw = _load_json("profile", serialized)
        payload = _mapping("profile", raw)
        expected = set(cls().to_payload())
        _exact_keys("profile", payload, expected)
        if payload["schema"] != PROFILE_SCHEMA or payload["profile"] != PROFILE_NAME:
            raise ValueError("unsupported profile schema or name")
        parameters = _mapping("parameters", payload["parameters"])
        _exact_keys(
            "parameters", parameters, {"tau_ms", "v_rest", "v_threshold", "v_reset", "resistance"}
        )
        parsed = cls(**parameters)
        expected_payload = parsed.to_payload()
        for field in expected - {"parameters"}:
            if payload[field] != expected_payload[field]:
                raise ValueError(f"unsupported or altered profile field: {field}")
        return parsed


@dataclass(frozen=True)
class CurrentDriveTick:
    """One duration with simultaneous piecewise-constant current inputs."""

    duration_ms: float
    currents: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_ms", _positive("duration_ms", self.duration_ms))
        currents = tuple(_finite("current", value) for value in self.currents)
        object.__setattr__(self, "currents", currents)
        _sum_currents(currents)

    @property
    def total_current(self) -> float:
        """Return the order-independent sum delivered at tick start."""
        return _sum_currents(self.currents)

    def to_payload(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "currents": list(self.currents),
            "total_current": self.total_current,
        }


@dataclass(frozen=True)
class ExactLIFState:
    """Complete persistent runtime state at a shot-relative instant."""

    voltage: float
    time_ms: float
    shot_id: str
    reset_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "voltage", _finite("voltage", self.voltage))
        time_ms = _finite("time_ms", self.time_ms)
        if time_ms < 0.0:
            raise ValueError("time_ms must be non-negative")
        object.__setattr__(self, "time_ms", time_ms)
        if not isinstance(self.shot_id, str) or not self.shot_id or len(self.shot_id) > 128:
            raise ValueError("shot_id must be a non-empty string of at most 128 characters")
        if (
            isinstance(self.reset_epoch, bool)
            or not isinstance(self.reset_epoch, int)
            or self.reset_epoch < 0
        ):
            raise ValueError("reset_epoch must be a non-negative integer")

    def to_payload(self) -> dict[str, Any]:
        return {
            "voltage": self.voltage,
            "time_ms": self.time_ms,
            "shot_id": self.shot_id,
            "reset_epoch": self.reset_epoch,
        }


@dataclass(frozen=True)
class ExactLIFStateSample:
    """One ordered point in the complete execution state trace."""

    sequence: int
    tick: int
    phase: Literal["initial", "threshold", "reset", "tick_end"]
    time_ms: float
    voltage: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "tick": self.tick,
            "phase": self.phase,
            "time_ms": self.time_ms,
            "voltage": self.voltage,
        }


@dataclass(frozen=True)
class ExactLIFEvent:
    """One exact threshold-crossing event."""

    sequence: int
    tick: int
    time_ms: float
    voltage_before_reset: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "tick": self.tick,
            "time_ms": self.time_ms,
            "voltage_before_reset": self.voltage_before_reset,
        }


@dataclass(frozen=True)
class ExactLIFExecutionPacket:
    """Immutable complete trace and provenance packet for one execute call."""

    producer_commit: str
    profile_digest: str
    initial_state: ExactLIFState
    ticks: tuple[CurrentDriveTick, ...]
    state_trace: tuple[ExactLIFStateSample, ...]
    events: tuple[ExactLIFEvent, ...]
    final_state: ExactLIFState

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": PACKET_SCHEMA,
            "producer_commit": self.producer_commit,
            "profile": {
                "name": PROFILE_NAME,
                "schema": PROFILE_SCHEMA,
                "sha256": self.profile_digest,
            },
            "solver": "closed_form_piecewise_constant_event_driven",
            "numeric": "IEEE-754 binary64/round_to_nearest_ties_to_even/fail_closed_non_finite",
            "rng": "none",
            "reset_boundary": "explicit_shot_reset_only",
            "initial_state": self.initial_state.to_payload(),
            "ticks": [tick.to_payload() for tick in self.ticks],
            "state_trace": [sample.to_payload() for sample in self.state_trace],
            "events": [event.to_payload() for event in self.events],
            "final_state": self.final_state.to_payload(),
        }

    def to_json(self) -> str:
        """Serialize the packet canonically for deterministic evidence."""
        return _canonical_json(self.to_payload())

    @classmethod
    def from_json(
        cls,
        serialized: str | bytes,
        *,
        profile: ExactCurrentLIFProfile,
        expected_producer_commit: str,
    ) -> ExactLIFExecutionPacket:
        """Validate a packet by strict parsing and deterministic replay."""
        payload = _mapping("packet", _load_json("packet", serialized))
        _exact_keys(
            "packet",
            payload,
            {
                "schema",
                "producer_commit",
                "profile",
                "solver",
                "numeric",
                "rng",
                "reset_boundary",
                "initial_state",
                "ticks",
                "state_trace",
                "events",
                "final_state",
            },
        )
        if payload["schema"] != PACKET_SCHEMA:
            raise ValueError("unsupported packet schema")
        if payload["producer_commit"] != expected_producer_commit:
            raise ValueError("packet producer commit mismatch")
        profile_binding = _mapping("packet profile", payload["profile"])
        _exact_keys("packet profile", profile_binding, {"name", "schema", "sha256"})
        expected_binding = {
            "name": PROFILE_NAME,
            "schema": PROFILE_SCHEMA,
            "sha256": profile.digest,
        }
        if profile_binding != expected_binding:
            raise ValueError("packet profile binding mismatch")
        initial_payload = _mapping("initial_state", payload["initial_state"])
        _exact_keys(
            "initial_state", initial_payload, {"voltage", "time_ms", "shot_id", "reset_epoch"}
        )
        initial_state = ExactLIFState(**initial_payload)
        raw_ticks = payload["ticks"]
        if not isinstance(raw_ticks, list):
            raise ValueError("ticks must be an array")
        ticks: list[CurrentDriveTick] = []
        for raw_tick in raw_ticks:
            tick_payload = _mapping("tick", raw_tick)
            _exact_keys("tick", tick_payload, {"duration_ms", "currents", "total_current"})
            currents = tick_payload["currents"]
            if not isinstance(currents, list):
                raise ValueError("tick currents must be an array")
            tick = CurrentDriveTick(tick_payload["duration_ms"], tuple(currents))
            if tick_payload["total_current"] != tick.total_current:
                raise ValueError("tick total_current mismatch")
            ticks.append(tick)
        verifier = ExactCurrentLIFSession(
            profile,
            producer_commit=expected_producer_commit,
            shot_id=initial_state.shot_id,
        )
        verifier.restore_state(
            _canonical_json(
                {
                    "schema": STATE_SCHEMA,
                    "profile_sha256": profile.digest,
                    "state": initial_state.to_payload(),
                }
            )
        )
        replay = verifier.execute(ticks)
        if replay.to_payload() != payload:
            raise ValueError("packet content failed deterministic replay")
        return replay


class ExactCurrentLIFSession:
    """Stateful, failure-atomic executor for :class:`ExactCurrentLIFProfile`."""

    def __init__(
        self,
        profile: ExactCurrentLIFProfile,
        *,
        producer_commit: str,
        shot_id: str = "shot-0",
    ) -> None:
        if not isinstance(profile, ExactCurrentLIFProfile):
            raise TypeError("profile must be an ExactCurrentLIFProfile")
        if not isinstance(producer_commit, str) or _COMMIT_RE.fullmatch(producer_commit) is None:
            raise ValueError("producer_commit must be a lowercase 40-character Git SHA-1")
        profile.verify_source_binding()
        self.profile = profile
        self.producer_commit = producer_commit
        self._state = ExactLIFState(profile.v_rest, 0.0, shot_id, 0)

    @property
    def state(self) -> ExactLIFState:
        """Return the current immutable persistent state."""
        return self._state

    def reset_shot(self, shot_id: str) -> ExactLIFState:
        """Reset voltage and time explicitly at a new shot boundary."""
        candidate = ExactLIFState(self.profile.v_rest, 0.0, shot_id, self._state.reset_epoch + 1)
        self._state = candidate
        return candidate

    def serialize_state(self) -> str:
        """Serialize state with the exact profile digest."""
        return _canonical_json(
            {
                "schema": STATE_SCHEMA,
                "profile_sha256": self.profile.digest,
                "state": self._state.to_payload(),
            }
        )

    def restore_state(self, serialized: str | bytes) -> ExactLIFState:
        """Restore compatible state atomically; reject drift and unknown fields."""
        raw = _load_json("state", serialized)
        payload = _mapping("state envelope", raw)
        _exact_keys("state envelope", payload, {"schema", "profile_sha256", "state"})
        if payload["schema"] != STATE_SCHEMA:
            raise ValueError("unsupported state schema")
        if payload["profile_sha256"] != self.profile.digest:
            raise ValueError("state profile digest mismatch")
        state_payload = _mapping("state", payload["state"])
        _exact_keys("state", state_payload, {"voltage", "time_ms", "shot_id", "reset_epoch"})
        candidate = ExactLIFState(**state_payload)
        if candidate.voltage >= self.profile.v_threshold:
            raise ValueError("restored voltage must be below threshold")
        self._state = candidate
        return candidate

    def execute(self, ticks: Sequence[CurrentDriveTick]) -> ExactLIFExecutionPacket:
        """Execute a complete current sequence and commit state only on success."""
        frozen_ticks = tuple(ticks)
        if any(not isinstance(tick, CurrentDriveTick) for tick in frozen_ticks):
            raise TypeError("ticks must contain only CurrentDriveTick values")
        initial = self._state
        voltage = initial.voltage
        time_ms = initial.time_ms
        state_trace = [ExactLIFStateSample(0, -1, "initial", time_ms, voltage)]
        events: list[ExactLIFEvent] = []
        solver = ExactLIFSolver(
            tau=self.profile.tau_ms,
            v_rest=self.profile.v_rest,
            v_thresh=self.profile.v_threshold,
            v_reset=self.profile.v_reset,
            r_m=self.profile.resistance,
        )

        for tick_index, tick in enumerate(frozen_ticks):
            end_ms = time_ms + tick.duration_ms
            if not math.isfinite(end_ms):
                raise ValueError("execution time overflowed binary64")
            emitted_this_tick = 0
            while time_ms < end_ms:
                remaining = end_ms - time_ms
                crossing = solver.next_spike_time(voltage, tick.total_current)
                if crossing is not None and not math.isfinite(crossing):
                    raise FloatingPointError("non-finite threshold-crossing interval")
                if crossing is None or crossing > remaining:
                    voltage = solver.evolve_to_time(voltage, remaining, tick.total_current)
                    if not math.isfinite(voltage):
                        raise FloatingPointError("membrane evolution produced a non-finite voltage")
                    time_ms = end_ms
                    break
                if crossing <= 0.0:
                    raise FloatingPointError("non-positive threshold-crossing interval")
                next_time_ms = time_ms + crossing
                if not math.isfinite(next_time_ms) or next_time_ms <= time_ms:
                    raise FloatingPointError("threshold-crossing time made no finite progress")
                time_ms = next_time_ms
                voltage = self.profile.v_threshold
                state_trace.append(
                    ExactLIFStateSample(len(state_trace), tick_index, "threshold", time_ms, voltage)
                )
                events.append(ExactLIFEvent(len(events), tick_index, time_ms, voltage))
                voltage = self.profile.v_reset
                state_trace.append(
                    ExactLIFStateSample(len(state_trace), tick_index, "reset", time_ms, voltage)
                )
                emitted_this_tick += 1
                if emitted_this_tick > _MAX_EVENTS_PER_TICK:
                    raise ValueError("event count exceeded the bounded execution contract")
            state_trace.append(
                ExactLIFStateSample(len(state_trace), tick_index, "tick_end", time_ms, voltage)
            )

        final = ExactLIFState(voltage, time_ms, initial.shot_id, initial.reset_epoch)
        packet = ExactLIFExecutionPacket(
            producer_commit=self.producer_commit,
            profile_digest=self.profile.digest,
            initial_state=initial,
            ticks=frozen_ticks,
            state_trace=tuple(state_trace),
            events=tuple(events),
            final_state=final,
        )
        packet.to_json()
        self._state = final
        return packet


__all__ = [
    "CurrentDriveTick",
    "ExactCurrentLIFProfile",
    "ExactCurrentLIFSession",
    "ExactLIFEvent",
    "ExactLIFExecutionPacket",
    "ExactLIFState",
    "ExactLIFStateSample",
    "MODEL_SOURCE_SHA256",
    "PACKET_SCHEMA",
    "PROFILE_NAME",
    "PROFILE_SCHEMA",
    "STATE_SCHEMA",
]
