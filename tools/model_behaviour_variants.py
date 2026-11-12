#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Change the sign of one term of a model's equations and ask its behaviour cases to tell

"""Vary a model's equations one sign at a time and ask its behaviour cases to tell the difference.

A behaviour case can hold while the model does the opposite of its purpose: a
term with the wrong sign passes every case that does not depend on that sign.
This tool asks the cases of a model that question directly. For a catalogue
model that states behaviour cases and has a bound schema, it builds every
*variant* of the schema: the same schema with the sign of one term of one
equation changed. A term is an operand of a sum or difference anywhere in an
equation (also inside a factor or a function call); an equation that is not a
sum has one more variant, its whole right-hand side negated. A term that is
the number zero has no variant. Variants are built on the syntax tree of the
equation, never by a search in its text.

The unchanged schema is run first and must hold every case. Then every
variant is run on the cases in the descriptor's order, and stops at the first
case it breaks. A schema is run as the Studio runs a candidate: through the
Universal DSL, with the current of the case as the one input. A model is then
in one state:

``discriminated``
    the schema holds every case and every variant breaks one.
``not_discriminated``
    a variant holds every case: no case decides the sign of that term. The
    term is named. This makes the check fail; a case that tells the variant
    from the model is what removes it.
``schema_differs``
    the unchanged schema does not hold a case that the model's class holds
    (the behaviour report runs the class). This makes the check fail.
``no_schema``
    the model has no bound schema.
``schema_not_executable``
    the engine refuses the schema itself, with its reason (a descriptive
    record of the catalogue).
``cases_not_drivable``
    a case is not driven by the one input ``current``, observes a state
    variable the schema does not hold, or asks for more steps than a run may
    take.
``no_variant``
    the schema's equations have no term to negate.

A variant that breaks a case shows that the case depends on that sign. It
does not show that the sign is right: the cases are the author's statement of
what the model does, not an independent reference.

A variant whose run stops (the engine refuses a state that is no longer
finite) has broken the case; the report names the error and no observed
value. A variant the engine refuses to build is not handled here: none exists
among the variants of the bound schemas, and the tool then stops with the
engine's own error.

Usage::

    python tools/model_behaviour_variants.py --write
    python tools/model_behaviour_variants.py --check
    python tools/model_behaviour_variants.py --summary

``--check`` fails when the committed report differs from what the tool
computes, and when any model is ``not_discriminated`` or ``schema_differs``.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
import copy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from sc_neurocore.neurons.equation_refusals import EquationRefusal  # noqa: E402
from sc_neurocore.neurons.model_catalogue import load_descriptor  # noqa: E402
from sc_neurocore.neurons.model_descriptor import (  # noqa: E402
    BEHAVIOUR_SPIKE_OBSERVABLE,
    BehaviourCase,
)
from sc_neurocore.neurons.models import _CLASS_TO_MODULE  # noqa: E402
from sc_neurocore.neurons.universal_dsl import UniversalNeuron  # noqa: E402
from sc_neurocore.studio.candidate_catalogue import (  # noqa: E402
    CatalogueCandidateUnavailable,
    candidate_from_catalogue,
)
from tools.model_behaviour_report import MAX_CASE_STEPS, as_written, case_holds  # noqa: E402

#: Schema of the emitted report.
REPORT_SCHEMA = "sc-neurocore.model-behaviour-variants.v1"

#: Where the report is committed.
DEFAULT_OUTPUT = "docs/_generated/model_behaviour_variants.json"

#: The states a model that states cases can be in, in the order the summary prints them.
MODEL_STATES = (
    "discriminated",
    "not_discriminated",
    "schema_differs",
    "no_schema",
    "schema_not_executable",
    "cases_not_drivable",
    "no_variant",
)

#: The states that make ``--check`` fail.
_FAILING_STATES = ("not_discriminated", "schema_differs")

#: What a reader of the report must know about every variant in it.
REPORT_NOTES = (
    "a variant is a model's bound schema with the sign of one term of one equation changed, "
    "run through the Universal DSL with the current of each case as the one input",
    "a variant that breaks a case shows that the case depends on that sign; it does not show "
    "that the sign is right",
    "a model without a bound schema, or whose cases are not driven by the one input 'current', "
    "has no variants here",
)

#: The name of the variant that negates an equation that is not a sum.
WHOLE_SIDE = "the whole right-hand side"

_STATE_PREFIX = "state:"
_INPUT = "current"


def _is_sum(node: ast.AST) -> bool:
    """Say whether a node adds or subtracts two operands."""
    return isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))


def _terms(node: ast.expr, sign: int = 1) -> list[tuple[ast.expr, int]]:
    """Return the operands of a sum with their signs, through nested sums and differences."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        right = sign if isinstance(node.op, ast.Add) else -sign
        return _terms(node.left, sign) + _terms(node.right, right)
    return [(node, sign)]


def _joined(terms: Sequence[tuple[ast.expr, int]]) -> ast.expr:
    """Return the sum of signed operands as one expression."""
    first, sign = terms[0]
    joined: ast.expr = first if sign > 0 else ast.UnaryOp(ast.USub(), first)
    for node, sign in terms[1:]:
        joined = ast.BinOp(joined, ast.Add() if sign > 0 else ast.Sub(), node)
    return joined


def _sums(node: ast.AST, inside_a_sum: bool = False) -> list[ast.expr]:
    """Return every outermost sum of an expression tree, outer before inner, left before right."""
    found: list[ast.expr] = []
    here = _is_sum(node)
    if here and not inside_a_sum:
        assert isinstance(node, ast.expr)
        found.append(node)
    for child in ast.iter_child_nodes(node):
        found.extend(_sums(child, here))
    return found


def _is_zero(node: ast.expr) -> bool:
    """Say whether an operand is the number zero."""
    return (
        isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value == 0
    )


class _Swap(ast.NodeTransformer):
    """Replace one sum of a tree by another expression."""

    def __init__(self, target: ast.expr, replacement: ast.expr) -> None:
        self._target = target
        self._replacement = replacement

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Return the replacement for the target sum and visit every other operation."""
        if node is self._target:
            return self._replacement
        return self.generic_visit(node)


def equation_variants(expression: str) -> list[dict[str, str]]:
    """Return every variant of one equation: one sign of one term changed.

    Parameters
    ----------
    expression : str
        The right-hand side of a dynamics equation, in the Universal DSL's
        expression syntax.

    Returns
    -------
    list of dict
        One entry per variant, sums outer before inner and terms left to
        right: ``term`` (the operand with its sign, or
        :data:`WHOLE_SIDE`), ``within`` (the sum it is an operand of) and
        ``expression`` (the varied right-hand side).

    Raises
    ------
    SyntaxError
        If the text is not an expression.
    """
    variants: list[dict[str, str]] = []
    for index in range(len(_sums(ast.parse(expression, mode="eval").body))):
        for position in range(len(_terms(_sums(ast.parse(expression, mode="eval").body)[index]))):
            tree = ast.parse(expression, mode="eval")
            target = _sums(tree.body)[index]
            terms = _terms(target)
            node, sign = terms[position]
            if _is_zero(node):
                continue
            terms[position] = (node, -sign)
            within = ast.unparse(target)
            term = f"{'+' if sign > 0 else '-'} {ast.unparse(node)}"
            varied = ast.fix_missing_locations(_Swap(target, _joined(terms)).visit(tree))
            variants.append({"term": term, "within": within, "expression": ast.unparse(varied)})
    body = ast.parse(expression, mode="eval").body
    if not _is_sum(body) and not _is_zero(body):
        negated = ast.unparse(ast.UnaryOp(ast.USub(), body))
        variants.append({"term": WHOLE_SIDE, "within": ast.unparse(body), "expression": negated})
    return variants


class _Stopped(Exception):
    """A run of a schema stopped before its last step; the message names the error."""


def _observe(schema: Mapping[str, Any], current: float, steps: int, observable: str) -> float:
    """Run a schema for ``steps`` steps at a constant current and return the observable."""
    neuron = UniversalNeuron.from_dict(dict(schema))
    spikes = 0
    for _ in range(steps):
        try:
            spiked = neuron.step(I=current)
        except (FloatingPointError, EquationRefusal) as error:
            raise _Stopped(type(error).__name__) from error
        spikes += 1 if spiked else 0
    if observable == BEHAVIOUR_SPIKE_OBSERVABLE:
        return spikes
    return float(neuron.state[observable[len(_STATE_PREFIX) :]])


def _case_entry(schema: Mapping[str, Any], case: BehaviourCase) -> dict[str, object]:
    """Run one case on a schema and return what was observed and whether it held."""
    entry: dict[str, object] = {"name": case.name}
    try:
        observed = _observe(schema, dict(case.inputs)[_INPUT], case.steps, case.observable)
        entry["observed"] = as_written(observed, case.observable)
        reference = None
        if case.kind == "contrast":
            drive = dict(case.reference_inputs)[_INPUT]
            reference = _observe(schema, drive, case.steps, case.observable)
            entry["reference_observed"] = as_written(reference, case.observable)
        entry["held"] = case_holds(case, observed, reference)
    except _Stopped as stopped:
        entry.pop("observed", None)
        entry["stopped"] = str(stopped)
        entry["held"] = False
    return entry


def _variant_entry(
    schema: Mapping[str, Any],
    equation: str,
    variant: Mapping[str, str],
    cases: Sequence[BehaviourCase],
) -> dict[str, object]:
    """Run the cases on one variant, in order, up to the first case it breaks."""
    changed = copy.deepcopy(dict(schema))
    changed["dynamics"][equation] = variant["expression"]
    entry: dict[str, object] = {"equation": equation, **variant}
    for case in cases:
        result = _case_entry(changed, case)
        if not result["held"]:
            entry["outcome"] = "violated"
            entry["case"] = case.name
            for field in ("observed", "reference_observed", "stopped"):
                if field in result:
                    entry[field] = result[field]
            return entry
    entry["outcome"] = "not_discriminated"
    return entry


def _not_drivable(schema: Mapping[str, Any], cases: Sequence[BehaviourCase]) -> str | None:
    """Return why a schema cannot be driven by the cases, or ``None`` when it can."""
    for case in cases:
        if [name for name, _ in case.inputs] != [_INPUT]:
            return f"case {case.name!r} is not driven by the one input {_INPUT!r}"
        variable = case.observable[len(_STATE_PREFIX) :]
        if case.observable != BEHAVIOUR_SPIKE_OBSERVABLE and variable not in schema["state"]:
            return f"case {case.name!r} observes {variable!r}, which the schema does not hold"
        if case.steps > MAX_CASE_STEPS:
            return f"case {case.name!r} asks for more than {MAX_CASE_STEPS} steps"
    return None


def evaluate_schema(schema: Mapping[str, Any], cases: Sequence[BehaviourCase]) -> dict[str, Any]:
    """Run the cases of a model on its schema and on every variant of the schema.

    Parameters
    ----------
    schema : mapping
        A Universal DSL schema: the model's bound schema.
    cases : sequence of BehaviourCase
        The cases the model's descriptor states, in its order.

    Returns
    -------
    dict
        ``state`` (one of :data:`MODEL_STATES` other than ``no_schema``);
        ``unchanged``, one entry per case run on the schema as it is
        (``name``, ``observed``, ``held``, and ``reference_observed`` for a
        contrast, or ``stopped`` with the error's name); ``variants``, one
        entry per variant: ``equation``, ``term``, ``within``,
        ``expression``, ``outcome`` (``violated`` or ``not_discriminated``)
        and, when violated, the first ``case`` it broke with what was
        observed or the error that ``stopped`` the run; and ``reason`` for
        the states ``schema_not_executable`` and ``cases_not_drivable``.
    """
    result: dict[str, Any] = {"state": "no_variant", "unchanged": [], "variants": []}
    try:
        UniversalNeuron.from_dict(dict(schema))
    except EquationRefusal as refusal:
        return {**result, "state": "schema_not_executable", "reason": str(refusal)}
    reason = _not_drivable(schema, cases)
    if reason is not None:
        return {**result, "state": "cases_not_drivable", "reason": reason}
    variants = [
        (equation, variant)
        for equation, expression in schema["dynamics"].items()
        for variant in equation_variants(str(expression))
    ]
    if not variants:
        return result
    result["unchanged"] = [_case_entry(schema, case) for case in cases]
    if not all(entry["held"] for entry in result["unchanged"]):
        return {**result, "state": "schema_differs"}
    result["variants"] = [
        _variant_entry(schema, equation, variant, cases) for equation, variant in variants
    ]
    decided = all(entry["outcome"] == "violated" for entry in result["variants"])
    return {**result, "state": "discriminated" if decided else "not_discriminated"}


def build_report() -> dict[str, object]:
    """Return the variants report over the catalogue models that state behaviour cases.

    Returns
    -------
    dict
        The report payload: one row per model that states cases, with its
        state, the cases run on the unchanged schema and its variants.
    """
    rows: list[dict[str, object]] = []
    for name in sorted(_CLASS_TO_MODULE):
        descriptor = load_descriptor(name)
        cases = descriptor.behaviour if descriptor is not None else ()
        if not cases:
            continue
        try:
            schema = json.loads(candidate_from_catalogue(name)["candidate_text"])["model"]
        except CatalogueCandidateUnavailable:
            rows.append({"model": name, "state": "no_schema", "unchanged": [], "variants": []})
            continue
        rows.append({"model": name, **evaluate_schema(schema, cases)})
    return report_of(rows, catalogue_models=len(_CLASS_TO_MODULE))


def report_of(rows: Sequence[Mapping[str, Any]], *, catalogue_models: int) -> dict[str, object]:
    """Return the report payload for the given model rows.

    Parameters
    ----------
    rows : sequence of mapping
        One row per model that states cases: ``model`` and the keys
        :func:`evaluate_schema` returns.
    catalogue_models : int
        The number of models in the catalogue.

    Returns
    -------
    dict
        Schema version, the notes every reader needs, how many models state
        cases and how many do not, the count per state, the count of variants
        per outcome and the rows.
    """
    variants = [variant for row in rows for variant in row["variants"]]
    return {
        "schema_version": REPORT_SCHEMA,
        "notes": list(REPORT_NOTES),
        "catalogue_models": catalogue_models,
        "models_with_cases": len(rows),
        "models_without_cases": catalogue_models - len(rows),
        "totals": {
            state: sum(1 for row in rows if row["state"] == state) for state in MODEL_STATES
        },
        "variants": {
            "total": len(variants),
            "violated": sum(1 for variant in variants if variant["outcome"] == "violated"),
            "not_discriminated": sum(
                1 for variant in variants if variant["outcome"] == "not_discriminated"
            ),
        },
        "models": [dict(row) for row in rows],
    }


def check_problems(report: Mapping[str, Any], committed: str | None) -> list[str]:
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
        Empty when the committed text equals the computed report, every
        schema holds its model's cases and every variant breaks one.
    """
    problems: list[str] = []
    if committed != encode(report):
        problems.append("the committed report is absent or has drifted; run --write")
    totals = report["totals"]
    failing = {state: int(totals[state]) for state in _FAILING_STATES if totals[state]}
    if failing:
        problems.append(f"cases do not decide every term: {failing}")
    return problems


def encode(report: Mapping[str, Any]) -> str:
    """Return the report as the exact bytes committed to disk."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_summary(report: Mapping[str, Any]) -> str:
    """Return a short operator-readable summary of the report."""
    lines = [
        f"models that state cases: {report['models_with_cases']} of {report['catalogue_models']}"
    ]
    for state in MODEL_STATES:
        lines.append(f"  {state:22s} {int(report['totals'][state]):5d}")
    variants = report["variants"]
    lines.append(
        f"variants: {variants['total']}, of them violated {variants['violated']}, "
        f"not discriminated {variants['not_discriminated']}"
    )
    lines.extend(REPORT_NOTES)
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    build: Callable[[], Mapping[str, Any]] = build_report,
) -> int:
    """Write, check or summarise the report from the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.
    build : callable, optional
        Returns the report; :func:`build_report` when omitted. A caller that
        has built the report already passes it back to avoid a second run.

    Returns
    -------
    int
        ``0`` on success; ``1`` when ``--check`` finds the file absent or
        stale, a schema that does not hold its model's cases, or a variant
        that holds every case.
    """
    parser = argparse.ArgumentParser(description="Vary the equations of the catalogue's models.")
    parser.add_argument("--write", action="store_true", help="write the report")
    parser.add_argument("--check", action="store_true", help="fail on a stale or failing report")
    parser.add_argument("--summary", action="store_true", help="print the summary")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build()
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encode(report), encoding="utf-8")
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
