#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Run the behaviour cases the model descriptors state and report what each showed

"""Run the behaviour cases of every catalogue model and say what each one showed.

A model descriptor may state behaviour cases: what the model does at a named
drive, in a form that can fail (an interval with both ends, or a contrast with
a least margin). This tool runs them. Every case is run through the model's
public class, the code a user runs, from a newly built model, with the case's
constant inputs passed by name to ``step``.

Each case is reported with what was observed, whether it held, and whether it
was *exercised*: an interval on the spike count is exercised when at least one
spike was observed, an interval on a state variable when the variable left its
starting value, a contrast when it held. A model is then in one state:

``held``
    every case held and at least one was exercised.
``failed``
    a case did not hold.
``held_vacuously``
    every case held and none was exercised. A model that does nothing at all
    satisfies such cases, so this counts as a failure of the check.
``invalid``
    a case could not be run: it names an input the model's step does not take
    or leaves a required one out, its state variable is not held by the model
    or is not one finite number, the step refused the drive, a spike count
    was asked of a step that returned something other than 0 or 1, or it asks
    for more than :data:`MAX_CASE_STEPS` steps.
``no_case``
    the descriptor states no case.

The cases are the author's statement of what the model does. They show that
the model does what its author says; they are not an independent reference.

What a case can state is limited, and the report says so in its ``notes``. A
drive is a set of constant finite numbers passed by name, each as a float, so
a model whose step takes a vector, or requires a truth value or a whole
number, cannot be driven. A spike count needs a step that returns 0 or 1; a model whose step
returns a rate or several values can state cases on its state variables only.

Each case is written into the report as stated (steps, inputs, bounds or the
reference drive and margin) beside what was observed. Each model that states
cases carries ``cases_sha256``, the SHA-256 of its cases as parsed, in the
descriptor's order: the contract digest of a receipt does not cover the cases,
so this digest is what ties a result to the text that was run.

Observed spike counts are exact. A final state value is written with six
significant digits, so the committed report is the same on every machine that
agrees to that precision.

Usage::

    python tools/model_behaviour_report.py --write
    python tools/model_behaviour_report.py --check
    python tools/model_behaviour_report.py --summary

``--check`` fails when the committed report differs from what the tool
computes, and when any model failed, held vacuously or is invalid.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import inspect
import json
import math
import numbers
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_neurocore.neurons import models as _models  # noqa: E402
from sc_neurocore.neurons.model_catalogue import load_descriptor  # noqa: E402
from sc_neurocore.neurons.model_descriptor import (  # noqa: E402
    BEHAVIOUR_SPIKE_OBSERVABLE,
    BehaviourCase,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE  # noqa: E402

#: Schema of the emitted report.
REPORT_SCHEMA = "sc-neurocore.model-behaviour-report.v1"

#: Where the report is committed.
DEFAULT_OUTPUT = "docs/_generated/model_behaviour_report.json"

#: The states a model can be in, in the order the summary prints them.
MODEL_STATES = ("held", "failed", "held_vacuously", "invalid", "no_case")

#: Most steps one case may ask for: the bound of a Studio candidate run, so
#: that every case can also be run on a variant of the model's schema.
MAX_CASE_STEPS = 100_000

#: The states that make ``--check`` fail.
_FAILING_STATES = ("failed", "held_vacuously", "invalid")

#: What a reader of the report must know about every case in it.
REPORT_NOTES = (
    "a case is the author's statement of what the model does, not an independent reference",
    "a drive is constant finite numbers passed by name as floats; a model whose step takes a "
    "vector, or requires a truth value or a whole number, cannot state a case in this form",
    "a spike count needs a step that returns 0 or 1; a model whose step returns a rate or "
    "several values can state cases on its state variables only",
)

_STATE_PREFIX = "state:"


class _CannotRun(Exception):
    """A case cannot be run; the message is the reason the report gives."""


def _check_inputs(model: type[Any], inputs: Mapping[str, float]) -> None:
    """Refuse a drive that names an input the step does not take or leaves a required one out."""
    parameters = list(inspect.signature(model.step).parameters.values())[1:]
    named = [
        parameter
        for parameter in parameters
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    ]
    takes_any = any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)
    unknown = sorted(set(inputs) - {parameter.name for parameter in named})
    if unknown and not takes_any:
        raise _CannotRun(f"the step takes no input named {unknown}")
    missing = [
        parameter.name
        for parameter in named
        if parameter.default is parameter.empty and parameter.name not in inputs
    ]
    if missing:
        raise _CannotRun(f"the step requires the input {missing}")


def _observe(
    model: type[Any], inputs: Mapping[str, float], steps: int, observable: str
) -> tuple[float, bool]:
    """Run a new model for ``steps`` steps and return the observable and whether it moved."""
    if steps > MAX_CASE_STEPS:
        raise _CannotRun(f"the case asks for more than {MAX_CASE_STEPS} steps")
    _check_inputs(model, inputs)
    try:
        instance = model()
    except TypeError as error:
        raise _CannotRun("the model cannot be built without arguments") from error
    state = observable[len(_STATE_PREFIX) :] if observable != BEHAVIOUR_SPIKE_OBSERVABLE else None
    start: object = None
    if state is not None:
        if not hasattr(instance, state):
            raise _CannotRun(f"the model holds no state variable named {state!r}")
        start = getattr(instance, state)
    spikes = 0
    for _ in range(steps):
        try:
            returned = instance.step(**inputs)
        except (ValueError, ArithmeticError, TypeError) as error:
            raise _CannotRun(f"the step refused the drive ({type(error).__name__})") from error
        if state is None:
            if not isinstance(returned, numbers.Real) or returned not in (0, 1):
                raise _CannotRun("the step returned a value that is not 0 or 1")
            spikes += 1 if returned == 1 else 0
    if state is None:
        return spikes, spikes > 0
    final = getattr(instance, state)
    if not isinstance(final, numbers.Real) or not math.isfinite(final):
        raise _CannotRun(f"the state variable {state!r} is not a single finite number")
    return float(final), bool(final != start)


def _written(value: float, observable: str) -> int | float:
    """Return an observed value as the report writes it: an exact count, or six significant digits."""
    if observable == BEHAVIOUR_SPIKE_OBSERVABLE:
        return int(value)
    return float(f"{value:.6g}")


def _stated(value: float, observable: str) -> int | float:
    """Return a bound or a margin as the case states it: a whole count, or the number unchanged."""
    return int(value) if observable == BEHAVIOUR_SPIKE_OBSERVABLE else value


def _evaluate_case(model: type[Any], case: BehaviourCase) -> dict[str, object]:
    """Run one case and return its entry of the report."""
    entry: dict[str, object] = {
        "name": case.name,
        "kind": case.kind,
        "observable": case.observable,
        "steps": case.steps,
        "inputs": dict(case.inputs),
    }
    if case.bounds is not None:
        entry["bounds"] = [_stated(end, case.observable) for end in case.bounds]
    else:
        assert case.margin is not None
        entry["reference_inputs"] = dict(case.reference_inputs)
        entry["margin"] = _stated(case.margin, case.observable)
    try:
        observed, moved = _observe(model, dict(case.inputs), case.steps, case.observable)
        entry["observed"] = _written(observed, case.observable)
        if case.kind == "interval":
            assert case.bounds is not None
            low, high = case.bounds
            entry["held"] = low <= observed <= high
            entry["exercised"] = moved
        else:
            assert case.margin is not None
            reference, _ = _observe(model, dict(case.reference_inputs), case.steps, case.observable)
            entry["reference_observed"] = _written(reference, case.observable)
            entry["held"] = observed - reference >= case.margin
            entry["exercised"] = entry["held"]
    except _CannotRun as refusal:
        entry.pop("observed", None)
        entry["held"] = False
        entry["exercised"] = False
        entry["reason"] = str(refusal)
    return entry


def cases_digest(cases: Sequence[BehaviourCase]) -> str:
    """Return the SHA-256 that binds the cases of one model as stated.

    Parameters
    ----------
    cases : sequence of BehaviourCase
        The cases, in the descriptor's order.

    Returns
    -------
    str
        The hexadecimal SHA-256 of the canonical JSON of the cases: every
        field of every case, keys sorted, no white space.
    """
    stated = [dataclasses.asdict(case) for case in cases]
    canonical = json.dumps(stated, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_model(model: type[Any], cases: Sequence[BehaviourCase]) -> dict[str, Any]:
    """Run the cases of one model through its public class.

    Parameters
    ----------
    model : type
        The model's class; every run builds a new instance without arguments.
    cases : sequence of BehaviourCase
        The cases its descriptor states.

    Returns
    -------
    dict
        ``state`` (one of :data:`MODEL_STATES`); ``cases_sha256``
        (:func:`cases_digest`, ``None`` without cases); and ``cases``, one
        entry per case: the case as stated (``name``, ``kind``,
        ``observable``, ``steps``, ``inputs`` and ``bounds``, or
        ``reference_inputs`` and ``margin``), then ``held``, ``exercised`` and
        either ``observed`` (and ``reference_observed`` for a contrast) or the
        ``reason`` it could not be run.
    """
    entries = [_evaluate_case(model, case) for case in cases]
    if not entries:
        state = "no_case"
    elif any("reason" in entry for entry in entries):
        state = "invalid"
    elif not all(entry["held"] for entry in entries):
        state = "failed"
    elif not any(entry["exercised"] for entry in entries):
        state = "held_vacuously"
    else:
        state = "held"
    return {
        "state": state,
        "cases_sha256": cases_digest(cases) if entries else None,
        "cases": entries,
    }


def build_report() -> dict[str, object]:
    """Return the behaviour report over the whole catalogue.

    Returns
    -------
    dict
        The report payload: one row per model with its state and its cases,
        and the count of models per state.
    """
    rows: list[dict[str, object]] = []
    for name in sorted(_CLASS_TO_MODULE):
        descriptor = load_descriptor(name)
        cases = descriptor.behaviour if descriptor is not None else ()
        if cases:
            rows.append({"model": name, **evaluate_model(getattr(_models, name), cases)})
        else:
            rows.append({"model": name, **evaluate_model(object, ())})
    return report_of(rows)


def report_of(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the report payload for the given model rows.

    Parameters
    ----------
    rows : sequence of mapping
        One row per model: ``model`` and the keys :func:`evaluate_model` returns.

    Returns
    -------
    dict
        Schema version, the notes every reader needs, the number of models,
        the count per state and the rows.
    """
    return {
        "schema_version": REPORT_SCHEMA,
        "notes": list(REPORT_NOTES),
        "catalogue_models": len(rows),
        "totals": {
            state: sum(1 for row in rows if row["state"] == state) for state in MODEL_STATES
        },
        "models": [dict(row) for row in rows],
    }


def check_problems(report: Mapping[str, object], committed: str | None) -> list[str]:
    """Say why a report does not pass its check.

    Parameters
    ----------
    report : mapping
        The report the tool computes now.
    committed : str or None
        The text of the committed report, ``None`` when the file is absent.

    Returns
    -------
    list of str
        Empty when the committed text equals the computed report and every
        model that states cases holds them with at least one exercised.
    """
    problems: list[str] = []
    if committed != encode(dict(report)):
        problems.append("the committed report is absent or has drifted; run --write")
    totals = report["totals"]
    assert isinstance(totals, Mapping)
    failing = {state: int(totals[state]) for state in _FAILING_STATES if totals[state]}
    if failing:
        problems.append(f"behaviour cases do not hold: {failing}")
    return problems


def encode(report: dict[str, object]) -> str:
    """Return the report as the exact bytes committed to disk."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_summary(report: dict[str, object]) -> str:
    """Return a short operator-readable summary of the report."""
    totals = report["totals"]
    assert isinstance(totals, dict)
    lines = [f"catalogue models: {report['catalogue_models']}"]
    for state in MODEL_STATES:
        lines.append(f"  {state:15s} {int(totals[state]):5d}")
    lines.extend(REPORT_NOTES)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Write, check or summarise the report from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        ``0`` on success; ``1`` when ``--check`` finds the file absent or
        stale, or a model that failed, held vacuously or is invalid.
    """
    parser = argparse.ArgumentParser(description="Run the behaviour cases of the catalogue.")
    parser.add_argument("--write", action="store_true", help="write the report")
    parser.add_argument("--check", action="store_true", help="fail on a stale or failing report")
    parser.add_argument("--summary", action="store_true", help="print the summary")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report()
    rendered = encode(report)
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.check:
        committed = args.output.read_text(encoding="utf-8") if args.output.is_file() else None
        problems = check_problems(report, committed)
        if problems:
            for problem in problems:
                print(f"{args.output}: {problem}")
            return 1
        print(f"{args.output} is up to date")
    if args.summary or not (args.write or args.check):
        print(render_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
