# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Run a candidate model, its reference tests, and its review packet

"""Simulate a validated candidate and assemble what a reviewer receives.

A candidate runs through the Universal DSL under its own declared numerical
profile — the integration method and timestep it states — so what is
simulated is what is proposed. Every run is bounded by
:data:`~sc_neurocore.studio.candidate_package.MAX_CANDIDATE_STEPS`, and a run
whose state stops being finite — which the equation engine refuses to carry
on from — reports the step where it diverged and why, instead of returning
numbers that mean nothing.

A reference test is the author's own proposal: a drive, a length and bounds on
the spike count or on the final state. Running them shows that the candidate
does what its author says it does; it is not an independent check against the
source, and the review packet says so. The packet binds the candidate, its
validation, its diff against the parent and the test results under one digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping
from typing import Any

from sc_neurocore.studio.candidate_diff import diff_candidate
from sc_neurocore.studio.candidate_package import (
    MAX_CANDIDATE_STEPS,
    candidate_sha256,
    validate_candidate,
)

REVIEW_PACKET_SCHEMA_VERSION = "sc-neurocore.studio.candidate-review.v1"
MAX_TRACE_POINTS = 5_000
"""Most samples per state variable a simulation returns; longer runs are strided."""

NOT_ESTABLISHED = (
    "catalogue promotion: a candidate is never listed as a catalogue model; promotion is "
    "separately authorised and evidence-gated",
    "independent validation: the reference tests are the author's proposals, not an "
    "independent oracle against the cited source",
    "dimensional consistency: units are declared per quantity; the equations are not checked "
    "against them",
    "hardware: no fixed-point, RTL or co-simulation evidence is produced for a candidate",
)
"""What a review packet does not show, stated in every packet."""


class CandidateRejected(ValueError):
    """The candidate is not valid, so it is not run."""

    def __init__(self, validation: dict[str, Any]) -> None:
        super().__init__("the candidate is not valid")
        self.validation = validation


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require_valid_candidate(document: Mapping[str, Any]) -> None:
    """Refuse a candidate that is not valid.

    Raises
    ------
    CandidateRejected
        Carrying the located validation, when the candidate has any problem.
    """
    validation = validate_candidate(document)
    if not validation.valid:
        raise CandidateRejected(validation.to_public_dict())


def _run(
    model: Mapping[str, Any], *, current: float, steps: int, keep_trace: bool
) -> dict[str, Any]:
    """Step the model ``steps`` times under a constant ``current``."""
    from sc_neurocore.neurons.universal_dsl import UniversalNeuron

    neuron = UniversalNeuron.from_dict(dict(model))
    stride = max(1, math.ceil(steps / MAX_TRACE_POINTS))
    names = list(neuron.state)
    trace: dict[str, list[float]] = {name: [] for name in names}
    spike_steps: list[int] = []
    diverged_at: int | None = None
    divergence: str | None = None
    for step in range(steps):
        try:
            spiked = neuron.step(I=current)
        except FloatingPointError as exc:
            # The equation engine refuses a state that stops being finite.
            diverged_at, divergence = step, str(exc)
            break
        if spiked:
            spike_steps.append(step)
        if keep_trace and step % stride == 0:
            state = neuron.state
            for name in names:
                trace[name].append(float(state[name]))
    final = {name: float(neuron.state[name]) for name in names}
    result: dict[str, Any] = {
        "steps": steps,
        # The profile the Universal DSL realised, so the reported timestep and
        # method are the ones that ran, not a default read from the document.
        "profile": neuron.realised_profile(),
        "current": current,
        "spike_count": len(spike_steps),
        "spike_steps": spike_steps,
        "final_state": final if diverged_at is None else None,
        "diverged_at_step": diverged_at,
        "divergence": divergence,
    }
    if keep_trace:
        result["sample_every"] = stride
        result["trace"] = trace
    return result


def simulate_candidate(
    document: Mapping[str, Any], *, current: float, steps: int
) -> dict[str, Any]:
    """Simulate a valid candidate under a constant current.

    Raises
    ------
    CandidateRejected
        When the candidate is not valid; the exception carries the validation.
    ValueError
        When ``steps`` is outside 1 .. ``MAX_CANDIDATE_STEPS``.
    """
    if not 1 <= steps <= MAX_CANDIDATE_STEPS:
        raise ValueError(f"steps must be from 1 to {MAX_CANDIDATE_STEPS}")
    require_valid_candidate(document)
    run = _run(document["model"], current=current, steps=steps, keep_trace=True)
    return {
        "candidate": document["name"],
        "candidate_sha256": candidate_sha256(document),
        "units": {"current": document["units"]["current"], "time": document["units"]["time"]},
        **run,
    }


def _within(bounds: Mapping[str, Any], value: float) -> bool:
    return float(bounds.get("min", -math.inf)) <= value <= float(bounds.get("max", math.inf))


def run_reference_tests(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run every reference test a valid candidate proposes.

    Each result states what was observed and, per expectation, whether it held.
    """
    require_valid_candidate(document)
    results: list[dict[str, Any]] = []
    for test in document["reference_tests"]:
        run = _run(
            document["model"],
            current=float(test["current"]),
            steps=int(test["steps"]),
            keep_trace=False,
        )
        checks: list[dict[str, Any]] = []
        expect = test["expect"]
        if "spike_count" in expect:
            checks.append(
                {
                    "quantity": "spike_count",
                    "bounds": expect["spike_count"],
                    "observed": run["spike_count"],
                    "held": _within(expect["spike_count"], run["spike_count"]),
                }
            )
        for variable, bounds in expect.get("final_state", {}).items():
            observed = None if run["final_state"] is None else run["final_state"].get(variable)
            checks.append(
                {
                    "quantity": f"final_state.{variable}",
                    "bounds": bounds,
                    "observed": observed,
                    "held": observed is not None and _within(bounds, observed),
                }
            )
        results.append(
            {
                "name": test["name"],
                "current": run["current"],
                "steps": run["steps"],
                "diverged_at_step": run["diverged_at_step"],
                "checks": checks,
                "passed": run["diverged_at_step"] is None and all(c["held"] for c in checks),
            }
        )
    return results


def review_packet(document: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the review packet of a valid candidate.

    The packet carries the candidate unchanged, its validation, its diff
    against the parent, the reference-test results, the environment that
    produced them and what the packet does not establish, all under one digest.
    """
    from sc_neurocore import __version__

    validation = validate_candidate(document)
    if not validation.valid:
        raise CandidateRejected(validation.to_public_dict())
    tests = run_reference_tests(document)
    body: dict[str, Any] = {
        "schema_version": REVIEW_PACKET_SCHEMA_VERSION,
        "candidate": dict(document),
        "candidate_sha256": validation.candidate_sha256,
        "validation": validation.to_public_dict(),
        "diff": diff_candidate(document),
        "reference_tests": tests,
        "reference_tests_passed": all(test["passed"] for test in tests),
        "environment": {
            "sc_neurocore": __version__,
            "python": platform.python_version(),
        },
        "not_established": list(NOT_ESTABLISHED),
    }
    body["packet_sha256"] = _digest(body)
    return body


__all__ = [
    "CandidateRejected",
    "MAX_TRACE_POINTS",
    "NOT_ESTABLISHED",
    "REVIEW_PACKET_SCHEMA_VERSION",
    "require_valid_candidate",
    "review_packet",
    "run_reference_tests",
    "simulate_candidate",
]
