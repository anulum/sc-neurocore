# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio replay pack: an experiment another process can re-run

"""A sealed, re-runnable record of one Studio experiment.

An export is only useful if running it reproduces the experiment it came from.
A replay pack therefore carries three separable things.

**The request that re-resolves.** Not the executed keyword arguments — those
are private to :mod:`sc_neurocore.studio.experiment_spec` — but the public
request fields, with any drawn randomness pinned, so that
:func:`~sc_neurocore.studio.experiment_spec.resolve_experiment` in another
process produces the same effective experiment. A fresh stochastic trial is
pinned to the seed it drew and sealed as a replay; a pack is never a lottery.

**The identity of the experiment.** ``experiment_identity_sha256`` digests the
scientific blocks of the specification — model revision, numerical profile,
steps, parameters, initial state, protocol, randomness, backend — and
deliberately excludes the runtime block and the cache key. The identity is what
must be unchanged for a replay to mean anything; the runtime is what must be
*reported* when it differs.

**The complete expectation.** Every spike event, a digest per state trace with
its endpoints and range, the initial and final state, the drive digest and the
run statistics. Comparison is exact on events and digests, and falls back to a
stated numerical tolerance that reports the largest deviation it found. A
replay never "passes" because two spike counts happen to agree.

Verification refuses before anything executes: an unsupported schema, a model
that is not in the installed catalogue, an experiment identity that no longer
resolves the same way, a drifted model revision, or a drifted runtime. Runtime
drift can be admitted explicitly, never silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

import sc_neurocore
from sc_neurocore.studio.experiment_spec import (
    ExperimentRejected,
    ExperimentSpec,
    resolve_experiment,
    run_experiment,
)
from sc_neurocore.studio.simulation import MAX_STEPS
from sc_neurocore.studio.trace_projection import full_state_traces

REPLAY_PACK_SCHEMA_VERSION = "studio.replay-pack.v1"
SUPPORTED_REPLAY_PACK_SCHEMA_VERSIONS = frozenset({REPLAY_PACK_SCHEMA_VERSION})
DEFAULT_STATE_TOLERANCE = 0.0
MAX_PACK_BYTES = 32 * 1024 * 1024

#: Specification blocks that carry the science of the experiment. The runtime
#: block and the cache key are excluded on purpose: a different interpreter is
#: reportable drift, not a different experiment.
IDENTITY_BLOCKS = (
    "schema_version",
    "source",
    "model",
    "equations",
    "numerical",
    "steps",
    "parameters",
    "initial_state",
    "initial_state_source",
    "protocol",
    "randomness",
    "backend",
)

#: Request fields a pack may carry. Anything else is refused rather than
#: silently dropped on the way out or on the way back in.
MODEL_REQUEST_FIELDS = (
    "name",
    "params",
    "dt",
    "duration",
    "current",
    "protocol",
    "frequency_hz",
    "seed",
    "trial",
)
ODE_REQUEST_FIELDS = (
    "equations",
    "threshold",
    "reset",
    "params",
    "init",
    "dt",
    "duration",
    "current",
    "protocol",
    "frequency_hz",
    "seed",
    "trial",
)

RefusalStage = Literal["schema", "request", "identity", "revision", "runtime"]
Verdict = Literal["match", "match-within-tolerance", "mismatch"]


class ReplayRejected(ValueError):
    """Raised when a pack cannot be replayed, before anything is executed.

    Parameters
    ----------
    stage : {"schema", "request", "identity", "revision", "runtime"}
        Which admission step refused.
    reason : str
        Bounded, path-free explanation.
    differences : sequence of str, optional
        Named blocks or fields that differ, for a drift refusal.
    """

    def __init__(
        self, *, stage: RefusalStage, reason: str, differences: Sequence[str] = ()
    ) -> None:
        super().__init__(f"{stage}: {reason}")
        self.stage: RefusalStage = stage
        self.reason = reason
        self.differences = tuple(differences)

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "replay_refused",
            "stage": self.stage,
            "reason": self.reason,
            "differences": list(self.differences),
        }


@dataclass(frozen=True, slots=True)
class ReplayAdmission:
    """The outcome of admitting a pack, before the experiment runs.

    Attributes
    ----------
    spec : ExperimentSpec
        The experiment re-resolved in this process.
    identity_sha256 : str
        Identity digest of the re-resolved experiment; equal to the pack's.
    runtime_differences : tuple of str
        Runtime fields that differ from the sealed environment. Non-empty only
        when the caller admitted runtime drift explicitly.
    """

    spec: ExperimentSpec
    identity_sha256: str
    runtime_differences: tuple[str, ...] = field(default=())


def _canonical_json(value: Any) -> Any:
    """Return a value in the form every JSON implementation agrees on.

    JSON has one number type. Python distinguishes ``1000`` from ``1000.0`` and
    a browser's ``JSON.stringify`` does not, so a pack that travels through the
    browser comes back with its integral floats narrowed to integers. Digesting
    the raw Python types would call that a corrupt document. Integral floats are
    therefore canonicalised to integers before hashing, and mapping keys are
    ordered, so the digest describes the JSON value rather than the Python
    objects that happened to carry it.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    return value


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _canonical_json(payload),
        allow_nan=False,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_floats(values: Sequence[float]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_ints(values: Sequence[int]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def experiment_identity(public: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scientific blocks of a public specification.

    Parameters
    ----------
    public : mapping
        A public :class:`~sc_neurocore.studio.experiment_spec.ExperimentSpec`
        projection.

    Returns
    -------
    dict
        The blocks named in :data:`IDENTITY_BLOCKS` that the specification
        actually carries, in that order.
    """
    return {name: public[name] for name in IDENTITY_BLOCKS if name in public}


def experiment_identity_sha256(public: Mapping[str, Any]) -> str:
    """Return the digest of a specification's scientific identity."""
    return _sha256_json(experiment_identity(public))


def _environment_block() -> dict[str, str]:
    return {
        "package_version": sc_neurocore.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.system(),
        "machine": platform.machine(),
    }


def pinned_request(request: Mapping[str, Any], spec: ExperimentSpec) -> dict[str, Any]:
    """Return the request fields that re-resolve to this experiment.

    A drawn or defaulted seed is written back into the request and the trial is
    sealed as ``replay``, so the pack reproduces the run it recorded instead of
    drawing new randomness. A deterministic experiment carries no seed, because
    the run contract refuses one.

    Parameters
    ----------
    request : mapping
        The original request. Unknown keys are dropped here rather than
        travelling into the pack.
    spec : ExperimentSpec
        The experiment resolved from that request.

    Returns
    -------
    dict
        A JSON-safe request with pinned randomness.
    """
    fields = ODE_REQUEST_FIELDS if spec.source == "ode" else MODEL_REQUEST_FIELDS
    pinned = {key: request[key] for key in fields if key in request and request[key] is not None}
    randomness = spec.public["randomness"]
    if randomness["kind"] == "none":
        pinned.pop("seed", None)
        pinned["trial"] = "replay"
        return pinned
    pinned["seed"] = int(randomness["seed"])
    pinned["trial"] = "replay"
    if spec.source == "model":
        params = dict(pinned.get("params") or {})
        params.pop("seed", None)
        if params:
            pinned["params"] = params
        else:
            pinned.pop("params", None)
    return pinned


def replay_expectation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise a run result into the complete expectation a replay must meet.

    Every scalar state trace contributes a digest of its float64 bytes plus its
    endpoints and range, so a mismatch can be reported as a number rather than
    as "the digest differs". Spike events are carried in full: they are the
    observable a spiking experiment exists to produce.

    Parameters
    ----------
    result : mapping
        A result from
        :func:`~sc_neurocore.studio.experiment_spec.run_experiment`.

    Returns
    -------
    dict
        The expectation block of a replay pack.
    """
    traces = full_state_traces(result)
    raw = result.get("raw")
    raw_included = bool(isinstance(raw, Mapping) and raw.get("included"))
    spikes = [int(index) for index in result.get("spikes", [])]
    drive = list(
        raw["drive"]
        if raw_included and isinstance(raw, Mapping) and "drive" in raw
        else result.get("current_trace", [])
    )
    states: dict[str, Any] = {}
    for name in sorted(traces):
        values = traces[name]
        states[name] = {
            "sha256": _sha256_floats(values),
            "n_samples": len(values),
            "first": float(values[0]) if values else None,
            "last": float(values[-1]) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    stats = result.get("stats") or {}
    return {
        "trace_source": "raw" if raw_included else "display",
        "n_steps": int(result["n_steps"]),
        "dt": float(result["dt"]),
        "spike_count": int(result.get("spike_count", len(spikes))),
        "spikes": spikes,
        "spikes_sha256": _sha256_ints(spikes),
        "states": states,
        "initial_state": dict(result.get("initial_state") or {}),
        "final_state": dict(result.get("final_state") or {}),
        "drive_sha256": _sha256_floats(drive),
        "statistics": {
            "rate_hz": stats.get("rate_hz"),
            "isi_mean_ms": stats.get("isi_mean_ms"),
            "isi_cv": stats.get("isi_cv"),
        },
    }


def build_replay_pack(request: Mapping[str, Any], *, max_steps: int = MAX_STEPS) -> dict[str, Any]:
    """Resolve, pin, execute and seal one experiment into a replay pack.

    The pack is built from the *pinned* experiment, so what it promises is
    exactly what a replay of it produces. A fresh stochastic trial is executed
    once with the seed it drew.

    Parameters
    ----------
    request : mapping
        A Studio simulate request (catalogue model or equation playground).
    max_steps : int
        Largest synchronous step count; a larger run is refused by the
        experiment contract with ``execution_mode = job_required``.

    Returns
    -------
    dict
        A ``studio.replay-pack.v1`` document.

    Raises
    ------
    ModelInputError
        From the run contract: unknown model, parameter or unsupported step.
    ExperimentRejected
        From the experiment contract: invalid randomness, no complete step or
        an oversized run.
    """
    first = resolve_experiment(request, max_steps=max_steps)
    sealed_request = pinned_request(request, first)
    spec = resolve_experiment(sealed_request, max_steps=max_steps)
    result = run_experiment(spec)
    public = spec.public
    return {
        "schema_version": REPLAY_PACK_SCHEMA_VERSION,
        "source": spec.source,
        "request": sealed_request,
        "experiment": public,
        "experiment_sha256": spec.experiment_sha256,
        "experiment_identity_sha256": experiment_identity_sha256(public),
        "expectation": replay_expectation(result),
        "environment": _environment_block(),
        "runner": {
            "module": "sc_neurocore.studio.replay_pack",
            "command": "python -m sc_neurocore.studio.replay_pack <pack.json>",
            "entrypoint": "sc_neurocore.studio.replay_pack.replay_pack",
        },
    }


def _require_mapping(pack: Any) -> Mapping[str, Any]:
    if not isinstance(pack, Mapping):
        raise ReplayRejected(stage="schema", reason="a replay pack must be a JSON object")
    return pack


def _admit_schema(pack: Mapping[str, Any]) -> None:
    version = pack.get("schema_version")
    if version not in SUPPORTED_REPLAY_PACK_SCHEMA_VERSIONS:
        raise ReplayRejected(
            stage="schema",
            reason=(
                f"unsupported pack schema {version!r}; this installation replays "
                f"{', '.join(sorted(SUPPORTED_REPLAY_PACK_SCHEMA_VERSIONS))}"
            ),
        )
    for key in ("request", "experiment", "expectation", "environment"):
        if not isinstance(pack.get(key), Mapping):
            raise ReplayRejected(
                stage="schema", reason=f"pack field {key!r} is missing or not an object"
            )
    if not isinstance(pack.get("experiment_identity_sha256"), str):
        raise ReplayRejected(
            stage="schema", reason="pack field 'experiment_identity_sha256' is missing"
        )
    sealed = experiment_identity_sha256(pack["experiment"])
    if sealed != pack["experiment_identity_sha256"]:
        raise ReplayRejected(
            stage="schema",
            reason=(
                "the pack's sealed identity digest does not describe its own specification; "
                "the document is corrupt or was edited after it was written"
            ),
        )


def _admit_request(pack: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(pack["request"])
    source = pack.get("source")
    allowed = set(ODE_REQUEST_FIELDS if source == "ode" else MODEL_REQUEST_FIELDS)
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ReplayRejected(
            stage="request",
            reason=f"the pack request carries fields this contract does not execute: {', '.join(unknown)}",
            differences=unknown,
        )
    if source == "ode":
        if not request.get("equations"):
            raise ReplayRejected(stage="request", reason="an equation pack carries no equations")
    elif not request.get("name"):
        raise ReplayRejected(stage="request", reason="a model pack carries no model name")
    return request


def _block_differences(sealed: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Name the blocks that differ, comparing JSON values rather than Python types."""
    names = sorted(set(sealed) | set(current))
    return [
        name
        for name in names
        if _canonical_json(sealed.get(name)) != _canonical_json(current.get(name))
    ]


def _admit_runtime(pack: Mapping[str, Any], *, allow_runtime_drift: bool) -> tuple[str, ...]:
    sealed = dict(pack["environment"])
    current = _environment_block()
    differences = tuple(_block_differences(sealed, current))
    if not differences:
        return ()
    if not allow_runtime_drift:
        detail = ", ".join(
            f"{name}: pack {sealed.get(name)!r} vs installed {current.get(name)!r}"
            for name in differences
        )
        raise ReplayRejected(
            stage="runtime",
            reason=(
                f"the runtime that sealed this pack is not the runtime replaying it ({detail}); "
                "replay with runtime drift admitted if the difference is acceptable"
            ),
            differences=list(differences),
        )
    return differences


def verify_replay_pack(
    pack: Mapping[str, Any],
    *,
    allow_runtime_drift: bool = False,
    max_steps: int = MAX_STEPS,
) -> ReplayAdmission:
    """Admit a pack for replay, refusing before any side effect.

    The pack's request is re-resolved against the installed package and the
    resulting scientific identity is compared with the sealed one. Nothing is
    executed until every refusal has been ruled out.

    Parameters
    ----------
    pack : mapping
        A ``studio.replay-pack.v1`` document.
    allow_runtime_drift : bool
        Admit a package, interpreter, NumPy or platform difference and report
        it, instead of refusing. Never implicit.
    max_steps : int
        Largest synchronous step count for the re-resolved run.

    Returns
    -------
    ReplayAdmission
        The re-resolved experiment and any admitted runtime differences.

    Raises
    ------
    ReplayRejected
        Unsupported schema, unexecutable request, model or experiment drift, or
        unadmitted runtime drift.
    """
    document = _require_mapping(pack)
    _admit_schema(document)
    request = _admit_request(document)
    runtime_differences = _admit_runtime(document, allow_runtime_drift=allow_runtime_drift)
    try:
        spec = resolve_experiment(request, max_steps=max_steps)
    except ExperimentRejected as exc:
        raise ReplayRejected(
            stage="identity",
            reason=f"the sealed request no longer resolves here ({exc.field}: {exc.reason})",
            differences=[exc.field],
        ) from exc
    except Exception as exc:  # noqa: BLE001 - the run contract raises its own family
        raise ReplayRejected(
            stage="identity",
            reason=f"the sealed request no longer resolves here ({exc})",
        ) from exc
    public = spec.public
    identity = experiment_identity_sha256(public)
    if identity != document["experiment_identity_sha256"]:
        sealed_identity = experiment_identity(document["experiment"])
        differences = _block_differences(sealed_identity, experiment_identity(public))
        raise ReplayRejected(
            stage="identity" if "model" not in differences else "revision",
            reason=(
                "the installed package resolves a different experiment from the sealed one; "
                f"differing blocks: {', '.join(differences) or 'none named'}"
            ),
            differences=differences,
        )
    return ReplayAdmission(
        spec=spec, identity_sha256=identity, runtime_differences=runtime_differences
    )


def _compare_states(
    sealed: Mapping[str, Any], observed: Mapping[str, Any], *, tolerance: float
) -> tuple[list[str], float, bool]:
    """Compare state traces by digest, then by deviation.

    Returns the differences, the largest deviation seen, and whether any trace
    passed only because the caller allowed a tolerance.
    """
    differences: list[str] = []
    worst = 0.0
    tolerated = False
    missing = sorted(set(sealed) - set(observed))
    extra = sorted(set(observed) - set(sealed))
    differences.extend(f"state {name} absent from the replay" for name in missing)
    differences.extend(f"state {name} not in the pack" for name in extra)
    for name in sorted(set(sealed) & set(observed)):
        want = sealed[name]
        got = observed[name]
        if want["sha256"] == got["sha256"]:
            continue
        if want["n_samples"] != got["n_samples"]:
            differences.append(
                f"state {name}: {got['n_samples']} samples replayed, {want['n_samples']} sealed"
            )
            continue
        deviation = max(
            abs(float(got[key]) - float(want[key]))
            for key in ("first", "last", "min", "max")
            if want[key] is not None and got[key] is not None
        )
        worst = max(worst, deviation)
        if deviation > tolerance:
            differences.append(
                f"state {name}: endpoints and range deviate by {deviation:.6g} "
                f"(tolerance {tolerance:g})"
            )
        else:
            tolerated = True
    return differences, worst, tolerated


def compare_to_expectation(
    expectation: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_STATE_TOLERANCE,
) -> dict[str, Any]:
    """Compare a replayed result with a sealed expectation.

    Spike events are compared exactly; a spike train is an observable, not a
    rounding matter. State traces are compared by digest first and, when the
    digests differ, by the largest deviation of their endpoints and range
    against ``tolerance``.

    Parameters
    ----------
    expectation : mapping
        The ``expectation`` block of a replay pack.
    result : mapping
        The result of replaying the pack's experiment.
    tolerance : float
        Largest absolute state deviation still called a match. Zero means the
        replay must be bit-identical.

    Returns
    -------
    dict
        ``verdict``, the list of ``differences`` and the observed expectation.
    """
    observed = replay_expectation(result)
    differences: list[str] = []
    for key in ("n_steps", "spike_count"):
        if observed[key] != expectation.get(key):
            differences.append(f"{key}: {observed[key]} replayed, {expectation.get(key)} sealed")
    if observed["spikes_sha256"] != expectation.get("spikes_sha256"):
        sealed_spikes = list(expectation.get("spikes") or [])
        first_divergence = next(
            (
                index
                for index, (want, got) in enumerate(zip(sealed_spikes, observed["spikes"]))
                if want != got
            ),
            min(len(sealed_spikes), len(observed["spikes"])),
        )
        differences.append(
            f"spike events diverge at index {first_divergence} "
            f"({len(observed['spikes'])} replayed, {len(sealed_spikes)} sealed)"
        )
    if observed["drive_sha256"] != expectation.get("drive_sha256"):
        differences.append("the drive samples differ from the sealed protocol")
    state_differences, worst, traces_tolerated = _compare_states(
        expectation.get("states") or {}, observed["states"], tolerance=tolerance
    )
    differences.extend(state_differences)
    snapshots_tolerated = False
    for key in ("initial_state", "final_state"):
        sealed_state = expectation.get(key) or {}
        observed_state = observed[key]
        for name in sorted(set(sealed_state) | set(observed_state)):
            want = sealed_state.get(name)
            got = observed_state.get(name)
            if want is None or got is None:
                differences.append(f"{key}: variable {name} present on one side only")
                continue
            deviation = abs(float(got) - float(want))
            if deviation > tolerance:
                differences.append(f"{key}.{name}: {got!r} replayed, {want!r} sealed")
            elif deviation > 0.0:
                snapshots_tolerated = True
                worst = max(worst, deviation)
    # "match" means the replay reproduced the sealed values, not that a
    # deviation happened to fall inside the tolerance the caller allowed.
    exact = not (traces_tolerated or snapshots_tolerated)
    verdict: Verdict
    if differences:
        verdict = "mismatch"
    elif exact:
        verdict = "match"
    else:
        verdict = "match-within-tolerance"
    return {
        "verdict": verdict,
        "differences": differences,
        "worst_state_deviation": worst,
        "tolerance": tolerance,
        "observed": observed,
    }


def replay_pack(
    pack: Mapping[str, Any],
    *,
    allow_runtime_drift: bool = False,
    tolerance: float = DEFAULT_STATE_TOLERANCE,
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Admit, execute and judge one replay pack.

    Parameters
    ----------
    pack : mapping
        A ``studio.replay-pack.v1`` document.
    allow_runtime_drift : bool
        Admit and report a runtime difference instead of refusing.
    tolerance : float
        Largest absolute state deviation still called a match.
    max_steps : int
        Largest synchronous step count for the replay.

    Returns
    -------
    dict
        The comparison outcome with the admitted experiment digests and any
        runtime differences.

    Raises
    ------
    ReplayRejected
        From :func:`verify_replay_pack`, before the experiment runs.
    """
    admission = verify_replay_pack(
        pack, allow_runtime_drift=allow_runtime_drift, max_steps=max_steps
    )
    result = run_experiment(admission.spec)
    outcome = compare_to_expectation(pack["expectation"], result, tolerance=tolerance)
    outcome["experiment_sha256"] = admission.spec.experiment_sha256
    outcome["experiment_identity_sha256"] = admission.identity_sha256
    outcome["runtime_differences"] = list(admission.runtime_differences)
    return outcome


def load_replay_pack(path: Path) -> dict[str, Any]:
    """Read a replay pack from a regular file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the pack. It must be a regular file (a directory, device or
        dangling symlink is refused) no larger than 32 MiB.

    Returns
    -------
    dict
        The parsed pack; its contents are validated on admission, not here.

    Raises
    ------
    ReplayRejected
        The path is not a readable regular file, is too large, or does not
        contain a JSON object.
    """
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReplayRejected(stage="schema", reason="the replay pack path is not a regular file")
    size = resolved.stat().st_size
    if size > MAX_PACK_BYTES:
        raise ReplayRejected(
            stage="schema",
            reason=f"the replay pack is {size} bytes; the limit is {MAX_PACK_BYTES}",
        )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayRejected(
            stage="schema", reason=f"the replay pack is not valid JSON ({exc})"
        ) from exc
    return dict(_require_mapping(document))


def main(argv: Sequence[str] | None = None) -> int:
    """Replay a pack from the command line.

    Exit code 0 means the experiment reproduced, 1 that it did not, and 2 that
    the pack was refused before it ran.
    """
    parser = argparse.ArgumentParser(
        prog="python -m sc_neurocore.studio.replay_pack",
        description="Replay a sealed SC-NeuroCore Studio experiment and compare it in full.",
    )
    parser.add_argument("pack", type=Path, help="path to a studio.replay-pack.v1 JSON file")
    parser.add_argument(
        "--allow-runtime-drift",
        action="store_true",
        help="admit a package, interpreter, NumPy or platform difference and report it",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_STATE_TOLERANCE,
        help="largest absolute state deviation still called a match (default: exact)",
    )
    parser.add_argument("--json", action="store_true", help="print the full outcome as JSON")
    args = parser.parse_args(argv)
    try:
        document = load_replay_pack(args.pack)
        outcome = replay_pack(
            document,
            allow_runtime_drift=args.allow_runtime_drift,
            tolerance=args.tolerance,
        )
    except ReplayRejected as exc:
        print(json.dumps(exc.to_public_detail(), indent=2), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(outcome, indent=2, default=str))
    else:
        print(f"verdict: {outcome['verdict']}")
        print(f"experiment: {outcome['experiment_identity_sha256']}")
        for difference in outcome["differences"]:
            print(f"  - {difference}")
        for difference in outcome["runtime_differences"]:
            print(f"  ~ runtime drift admitted: {difference}")
    return 0 if outcome["verdict"] != "mismatch" else 1


__all__ = [
    "DEFAULT_STATE_TOLERANCE",
    "IDENTITY_BLOCKS",
    "MAX_PACK_BYTES",
    "MODEL_REQUEST_FIELDS",
    "ODE_REQUEST_FIELDS",
    "REPLAY_PACK_SCHEMA_VERSION",
    "SUPPORTED_REPLAY_PACK_SCHEMA_VERSIONS",
    "ReplayAdmission",
    "ReplayRejected",
    "build_replay_pack",
    "compare_to_expectation",
    "experiment_identity",
    "experiment_identity_sha256",
    "load_replay_pack",
    "main",
    "pinned_request",
    "replay_expectation",
    "replay_pack",
    "verify_replay_pack",
]


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
