# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate model packages: schema and located validation

"""A candidate model is a proposal, never a catalogue model.

A candidate package carries one model written in the Universal DSL together
with what a reviewer needs to judge it: the unit of every state variable and
parameter, the source it follows, the assumptions it makes, who wrote it, the
catalogue model it derives from, and the reference tests its author proposes.
Studio validates, simulates and diffs it and exports a review packet; nothing
here writes a canonical file or lists the candidate in the catalogue.
Promotion into the catalogue is a separate, authorised and evidence-gated
step.

Validation is fail-closed and located. Every problem is reported with a JSON
pointer to the field that caused it, so an editor can put the message beside
the input, and all problems are reported at once rather than the first only.
A document is valid only when it has no problem and the Universal DSL admits
its model.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_refusals import EquationRefusal
from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator
from sc_neurocore.studio.candidate_document import (
    CandidateDocumentRefused,
    encode_candidate_document,
)
from sc_neurocore.studio.candidate_expressions import validate_candidate_calls

CANDIDATE_SCHEMA_VERSION = "sc-neurocore.studio.candidate.v1"
"""The one candidate schema this build reads and writes."""

MAX_REFERENCE_TESTS = 16
"""Most reference tests one package may propose."""

MAX_CANDIDATE_STEPS = 100_000
"""Most integration steps one simulation or reference test may take."""

INPUT_SYMBOLS = frozenset({"I", "xi"})
"""Inputs an expression may read besides state and parameters: the injected
current and the diffusion noise sample."""

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "parent",
        "model",
        "units",
        "source",
        "assumptions",
        "authors",
        "reference_tests",
    }
)
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_FUNCTIONS = frozenset(name for name, value in build_eval_namespace().items() if callable(value))
_CONSTANTS = frozenset(
    name for name, value in build_eval_namespace().items() if not callable(value)
)


@dataclass(frozen=True)
class CandidateDiagnostic:
    """One problem with a candidate, located by a JSON pointer."""

    location: str
    message: str

    def to_public_dict(self) -> dict[str, str]:
        """Return a located diagnostic with JSON-safe Unicode text.

        Returns
        -------
        dict[str, str]
            The location and authored message. Lone surrogates are escaped;
            valid Unicode and the stored diagnostic remain unchanged.
        """
        return {
            key: value.encode("utf-8", errors="backslashreplace").decode("utf-8")
            for key, value in (("location", self.location), ("message", self.message))
        }


@dataclass(frozen=True)
class CandidateValidation:
    """The outcome of validating one candidate document."""

    diagnostics: tuple[CandidateDiagnostic, ...]
    candidate_sha256: str | None

    @property
    def valid(self) -> bool:
        """Whether the document has no problem."""
        return not self.diagnostics

    def to_public_dict(self) -> dict[str, Any]:
        """Return the validation as the API reports it."""
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "valid": self.valid,
            "candidate_sha256": self.candidate_sha256,
            "diagnostics": [diagnostic.to_public_dict() for diagnostic in self.diagnostics],
        }


def candidate_sha256(document: Mapping[str, Any]) -> str:
    """Return the digest of the candidate's finite canonical JSON form.

    Parameters
    ----------
    document:
        Candidate fields and attribution; the mapping is not modified.

    Returns
    -------
    str
        SHA-256 of sorted, compact UTF-8 JSON, preserving valid finite digests.

    Raises
    ------
    CandidateDocumentRefused
        If values cannot form finite JSON or text contains lone surrogates.
    """
    return hashlib.sha256(encode_candidate_document(document)).hexdigest()


def _pointer(*parts: str | int) -> str:
    """Return the JSON pointer naming ``parts`` (RFC 6901 escaping)."""
    escaped = (str(part).replace("~", "~0").replace("/", "~1") for part in parts)
    return "/" + "/".join(escaped) if parts else ""


class _Checker:
    """Collect every located problem with one document."""

    def __init__(self) -> None:
        self.diagnostics: list[CandidateDiagnostic] = []
        self._safety = ExpressionSafetyValidator()

    def fail(self, location: str, message: str) -> None:
        self.diagnostics.append(CandidateDiagnostic(location, message))

    def text(self, value: object, location: str, what: str) -> str | None:
        if not isinstance(value, str) or not value.strip():
            self.fail(location, f"{what} must be non-empty text")
            return None
        return value

    def number(self, value: object, location: str, what: str) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(location, f"{what} must be a number")
            return None
        try:
            number = float(value)
        except OverflowError:
            self.fail(location, f"{what} must be finite")
            return None
        if not math.isfinite(number):
            self.fail(location, f"{what} must be finite")
            return None
        return number

    def expression(self, value: object, location: str, known: frozenset[str]) -> None:
        text = self.text(value, location, "an expression")
        if text is None:
            return
        try:
            self._safety.validate(text)
        except EquationRefusal as exc:
            self.fail(location, str(exc))
            return
        except ValueError:
            self.fail(location, "the expression cannot be read")
            return
        tree = ast.parse(text, mode="eval")
        unknown = sorted(
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in known
        )
        if unknown:
            self.fail(location, f"unknown symbol(s) {', '.join(unknown)}")
            return
        try:
            validate_candidate_calls(tree)
        except EquationRefusal as exc:
            self.fail(location, str(exc))


def _validation(check: _Checker, document: Mapping[str, Any]) -> CandidateValidation:
    """Bind diagnostics to readable JSON, refusing invalid document encoding."""
    try:
        digest = candidate_sha256(document)
    except CandidateDocumentRefused as exc:
        check.fail("", str(exc))
        digest = None
    return CandidateValidation(tuple(check.diagnostics), digest)


def validate_candidate(
    document: object,
    *,
    catalogue: frozenset[str] | None = None,
) -> CandidateValidation:
    """Validate a candidate document and locate every problem.

    Parameters
    ----------
    document:
        The parsed candidate package.
    catalogue:
        Registered catalogue model names; taken from the registry when
        omitted. A candidate may not take a catalogue model's name, and its
        parent must be one.

    Returns
    -------
    CandidateValidation
        The problems, each with a JSON pointer, and the document digest when
        it is an object.
    """
    check = _Checker()
    if not isinstance(document, Mapping):
        check.fail("", "a candidate package must be a JSON object")
        return CandidateValidation(tuple(check.diagnostics), None)
    if catalogue is None:
        from sc_neurocore.neurons.models import _CLASS_TO_MODULE

        catalogue = frozenset(_CLASS_TO_MODULE)
    version = document.get("schema_version")
    if version != CANDIDATE_SCHEMA_VERSION:
        check.fail(
            _pointer("schema_version"),
            f"unsupported candidate schema {version!r}; this build reads {CANDIDATE_SCHEMA_VERSION}",
        )
        return _validation(check, document)
    for field in sorted(set(document) - _TOP_LEVEL_FIELDS):
        check.fail(
            _pointer(field), "unknown field; a candidate carries no field this build ignores"
        )
    _check_identity(check, document, catalogue)
    names = _check_model(check, document.get("model"))
    _check_units(check, document.get("units"), names)
    _check_attribution(check, document)
    _check_reference_tests(check, document.get("reference_tests"))
    if not check.diagnostics:
        _check_admission(check, document["model"])
    return _validation(check, document)


def _check_identity(
    check: _Checker, document: Mapping[str, Any], catalogue: frozenset[str]
) -> None:
    name = document.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        check.fail(_pointer("name"), "name must be an identifier of at most 64 characters")
    elif name in catalogue:
        check.fail(
            _pointer("name"),
            f"{name} is a catalogue model; a candidate is never listed as one",
        )
    parent = document.get("parent")
    if parent is not None and (not isinstance(parent, str) or parent not in catalogue):
        check.fail(_pointer("parent"), f"parent {parent!r} is not a catalogue model")


def _check_model(check: _Checker, model: object) -> tuple[set[str], set[str]]:
    """Check the DSL model and return its state and parameter names."""
    state: set[str] = set()
    parameters: set[str] = set()
    if not isinstance(model, Mapping):
        check.fail(_pointer("model"), "model must be a Universal DSL schema object")
        return state, parameters
    metadata = model.get("metadata")
    if not isinstance(metadata, Mapping):
        check.fail(_pointer("model", "metadata"), "metadata must be an object")
    else:
        check.text(metadata.get("name"), _pointer("model", "metadata", "name"), "metadata.name")
    for section, into in (("state", state), ("parameters", parameters)):
        values = model.get(section)
        if not isinstance(values, Mapping) or (section == "state" and not values):
            check.fail(_pointer("model", section), f"{section} must be an object of numbers")
            continue
        for key, value in values.items():
            location = _pointer("model", section, key)
            if not _NAME.fullmatch(str(key)) or key in INPUT_SYMBOLS or key in _FUNCTIONS:
                check.fail(location, f"{key!r} cannot name a {section} entry")
            check.number(value, location, f"{section}.{key}")
            into.add(str(key))
    for shared in sorted(state & parameters):
        check.fail(
            _pointer("model", "parameters", shared),
            f"{shared} is both a state variable and a parameter",
        )
    known = frozenset(state | parameters | INPUT_SYMBOLS | _FUNCTIONS | _CONSTANTS)
    dynamics = model.get("dynamics")
    if not isinstance(dynamics, Mapping) or not dynamics:
        check.fail(_pointer("model", "dynamics"), "dynamics must give each state its equation")
    else:
        for key, expression in dynamics.items():
            location = _pointer("model", "dynamics", key)
            if key not in state:
                check.fail(location, f"{key} is not a state variable")
            check.expression(expression, location, known)
        for missing in sorted(state - set(dynamics)):
            check.fail(_pointer("model", "dynamics"), f"state {missing} has no equation")
    _check_events(check, model, state, known)
    return state, parameters


def _check_events(
    check: _Checker, model: Mapping[str, Any], state: set[str], known: frozenset[str]
) -> None:
    threshold = model.get("threshold")
    if threshold is not None:
        if not isinstance(threshold, Mapping):
            check.fail(_pointer("model", "threshold"), "threshold must be an object")
        else:
            check.expression(
                threshold.get("condition"), _pointer("model", "threshold", "condition"), known
            )
    reset = model.get("reset")
    if reset is not None:
        if not isinstance(reset, Mapping):
            check.fail(_pointer("model", "reset"), "reset must map state variables to expressions")
        else:
            for key, expression in reset.items():
                location = _pointer("model", "reset", key)
                if key not in state:
                    check.fail(location, f"{key} is not a state variable")
                check.expression(expression, location, known)


def _check_units(check: _Checker, units: object, names: tuple[set[str], set[str]]) -> None:
    if not isinstance(units, Mapping):
        check.fail(_pointer("units"), "units must name the unit of every state and parameter")
        return
    for section, expected in zip(("state", "parameters"), names, strict=True):
        declared = units.get(section)
        if not isinstance(declared, Mapping):
            check.fail(_pointer("units", section), f"units.{section} must be an object")
            continue
        for missing in sorted(expected - set(declared)):
            check.fail(_pointer("units", section), f"{missing} has no unit")
        for extra in sorted(set(declared) - expected):
            check.fail(_pointer("units", section, extra), f"{extra} is not a {section} entry")
        for key in sorted(expected & set(declared)):
            _check_unit(check, declared[key], _pointer("units", section, key))
    for field in ("current", "time"):
        _check_unit(check, units.get(field), _pointer("units", field))


def _check_unit(check: _Checker, unit: object, location: str) -> None:
    text = check.text(unit, location, "a unit")
    if text is None:
        return
    import tokenize

    import pint

    try:
        _unit_registry().parse_units(text)
    except (
        pint.errors.UndefinedUnitError,
        pint.errors.DefinitionSyntaxError,
        tokenize.TokenError,
        TypeError,
        ValueError,
        ArithmeticError,
    ):
        check.fail(location, f"{text!r} is not a unit")


_REGISTRY: list[Any] = []


def _unit_registry() -> Any:
    """Return the one unit registry candidate units are parsed with.

    pint parses unit expressions with its own tokenizer, never ``eval``, so an
    uploaded unit string cannot execute anything.
    """
    if not _REGISTRY:
        import pint

        _REGISTRY.append(pint.UnitRegistry())
    return _REGISTRY[0]


def _check_attribution(check: _Checker, document: Mapping[str, Any]) -> None:
    source = document.get("source")
    if not isinstance(source, Mapping):
        check.fail(_pointer("source"), "source must cite what the model follows")
    else:
        check.text(source.get("citation"), _pointer("source", "citation"), "source.citation")
        for field in ("doi", "url"):
            if source.get(field) is not None:
                check.text(source[field], _pointer("source", field), f"source.{field}")
        for extra in sorted(set(source) - {"citation", "doi", "url"}):
            check.fail(_pointer("source", extra), "unknown field")
    for field, what in (("assumptions", "an assumption"), ("authors", "an author")):
        values = document.get(field)
        if not isinstance(values, Sequence) or isinstance(values, str) or not values:
            check.fail(_pointer(field), f"{field} must list at least one entry")
            continue
        for index, value in enumerate(values):
            check.text(value, _pointer(field, index), what)


def _check_reference_tests(check: _Checker, tests: object) -> None:
    if not isinstance(tests, Sequence) or isinstance(tests, str):
        check.fail(_pointer("reference_tests"), "reference_tests must be a list")
        return
    if len(tests) > MAX_REFERENCE_TESTS:
        check.fail(_pointer("reference_tests"), f"at most {MAX_REFERENCE_TESTS} reference tests")
    names: set[str] = set()
    for index, test in enumerate(tests):
        base = ("reference_tests", index)
        if not isinstance(test, Mapping):
            check.fail(_pointer(*base), "a reference test must be an object")
            continue
        name = check.text(test.get("name"), _pointer(*base, "name"), "a test name")
        if name is not None and name in names:
            check.fail(_pointer(*base, "name"), f"a second test is named {name}")
        names.add(name or "")
        check.number(test.get("current"), _pointer(*base, "current"), "current")
        steps = check.number(test.get("steps"), _pointer(*base, "steps"), "steps")
        if steps is not None and not (1 <= steps <= MAX_CANDIDATE_STEPS and steps.is_integer()):
            check.fail(
                _pointer(*base, "steps"),
                f"steps must be a whole number from 1 to {MAX_CANDIDATE_STEPS}",
            )
        _check_expectation(check, test.get("expect"), base)


def _check_expectation(check: _Checker, expect: object, base: tuple[str, int]) -> None:
    location = _pointer(*base, "expect")
    if not isinstance(expect, Mapping) or not expect:
        check.fail(location, "expect must state spike_count and/or final_state bounds")
        return
    for key in sorted(set(expect) - {"spike_count", "final_state"}):
        check.fail(_pointer(*base, "expect", key), "unknown expectation")
    if "spike_count" in expect:
        _check_bounds(check, expect["spike_count"], _pointer(*base, "expect", "spike_count"))
    final_state = expect.get("final_state")
    if final_state is not None:
        if not isinstance(final_state, Mapping) or not final_state:
            check.fail(_pointer(*base, "expect", "final_state"), "final_state must bound variables")
        else:
            for variable, bounds in final_state.items():
                _check_bounds(check, bounds, _pointer(*base, "expect", "final_state", variable))


def _check_bounds(check: _Checker, bounds: object, location: str) -> None:
    if not isinstance(bounds, Mapping) or not {"min", "max"} >= set(bounds) or not bounds:
        check.fail(location, "bounds must give min and/or max and nothing else")
        return
    low = check.number(bounds["min"], location + "/min", "min") if "min" in bounds else None
    high = check.number(bounds["max"], location + "/max", "max") if "max" in bounds else None
    if low is not None and high is not None and low > high:
        check.fail(location, "min exceeds max")


def _check_admission(check: _Checker, model: Mapping[str, Any]) -> None:
    """Ask the Universal DSL itself whether it admits the model."""
    from sc_neurocore.neurons.universal_dsl import UniversalNeuron

    try:
        UniversalNeuron.from_dict(dict(model))
    except EquationRefusal as exc:
        check.fail(_pointer("model"), f"the Universal DSL refuses the model: {exc}")
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
        check.fail(
            _pointer("model"),
            "the Universal DSL refuses the model: a model field is missing or invalid",
        )


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateDiagnostic",
    "CandidateValidation",
    "INPUT_SYMBOLS",
    "MAX_CANDIDATE_STEPS",
    "MAX_REFERENCE_TESTS",
    "candidate_sha256",
    "validate_candidate",
]
