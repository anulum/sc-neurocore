# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate package validation: every refusal, located

"""A candidate is valid only when nothing is wrong, and each fault says where.

Every case starts from a real, valid candidate and breaks one thing; the
validation must name that field with a JSON pointer and a reason an author can
act on.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import pytest

from sc_neurocore.studio.candidate_package import (
    CANDIDATE_SCHEMA_VERSION,
    MAX_CANDIDATE_STEPS,
    MAX_REFERENCE_TESTS,
    candidate_sha256,
    validate_candidate,
)
from tests.studio_candidate_support import adex_candidate

Mutation = Callable[[dict[str, Any]], None]


def _set(path: tuple[str | int, ...], value: object) -> Mutation:
    def mutate(document: dict[str, Any]) -> None:
        target: Any = document
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value

    return mutate


def _delete(path: tuple[str | int, ...]) -> Mutation:
    def mutate(document: dict[str, Any]) -> None:
        target: Any = document
        for part in path[:-1]:
            target = target[part]
        del target[path[-1]]

    return mutate


def _diagnostics(document: object) -> list[tuple[str, str]]:
    return [
        (diagnostic.location, diagnostic.message)
        for diagnostic in validate_candidate(document).diagnostics
    ]


def test_a_complete_candidate_is_valid_and_digest_bound() -> None:
    document = adex_candidate()
    validation = validate_candidate(document)

    assert validation.valid
    assert validation.candidate_sha256 == candidate_sha256(document)
    assert validation.to_public_dict() == {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "valid": True,
        "candidate_sha256": candidate_sha256(document),
        "diagnostics": [],
    }
    # The digest is of the canonical form: key order does not change it.
    reordered = json.loads(json.dumps(document, sort_keys=True))
    assert candidate_sha256(reordered) == candidate_sha256(document)


def test_a_document_that_is_not_an_object_has_no_digest() -> None:
    validation = validate_candidate(["not", "a", "package"])
    assert not validation.valid
    assert validation.candidate_sha256 is None
    assert _diagnostics("text") == [("", "a candidate package must be a JSON object")]


def test_an_unsupported_schema_version_is_refused_before_anything_else() -> None:
    document = adex_candidate()
    document["schema_version"] = "sc-neurocore.studio.candidate.v0"
    document["name"] = "not an identifier"
    diagnostics = _diagnostics(document)
    assert len(diagnostics) == 1
    assert diagnostics[0][0] == "/schema_version"
    assert "this build reads sc-neurocore.studio.candidate.v1" in diagnostics[0][1]


CASES: list[tuple[str, Mutation, str, str]] = [
    ("unknown field", _set(("notes",), "x"), "/notes", "unknown field"),
    ("name pattern", _set(("name",), "9lives"), "/name", "must be an identifier"),
    ("catalogue name", _set(("name",), "AdExNeuron"), "/name", "is a catalogue model"),
    ("parent", _set(("parent",), "NoSuchNeuron"), "/parent", "is not a catalogue model"),
    ("model object", _set(("model",), []), "/model", "Universal DSL schema object"),
    ("metadata", _set(("model", "metadata"), "x"), "/model/metadata", "must be an object"),
    (
        "metadata name",
        _set(("model", "metadata", "name"), " "),
        "/model/metadata/name",
        "non-empty text",
    ),
    ("empty state", _set(("model", "state"), {}), "/model/state", "object of numbers"),
    ("parameters", _set(("model", "parameters"), [1]), "/model/parameters", "object of numbers"),
    ("input as state", _set(("model", "state", "I"), 0.0), "/model/state/I", "cannot name"),
    ("boolean value", _set(("model", "parameters", "a"), True), "/model/parameters/a", "a number"),
    (
        "non-finite value",
        _set(("model", "parameters", "a"), math.inf),
        "/model/parameters/a",
        "must be finite",
    ),
    (
        "state and parameter",
        _set(("model", "parameters", "w"), 1.0),
        "/model/parameters/w",
        "both a state variable and a parameter",
    ),
    ("no dynamics", _set(("model", "dynamics"), {}), "/model/dynamics", "each state"),
    (
        "dynamics of a non-state",
        _set(("model", "dynamics", "z"), "0.0"),
        "/model/dynamics/z",
        "z is not a state variable",
    ),
    (
        "unsafe expression",
        _set(("model", "dynamics", "v"), "__import__('os')"),
        "/model/dynamics/v",
        "Blocked function",
    ),
    (
        "unknown symbol",
        _set(("model", "dynamics", "v"), "v + mystery"),
        "/model/dynamics/v",
        "unknown symbol(s) mystery",
    ),
    ("state without equation", _delete(("model", "dynamics", "w")), "/model/dynamics", "w has"),
    ("threshold", _set(("model", "threshold"), "v > 0"), "/model/threshold", "must be an object"),
    (
        "threshold condition",
        _set(("model", "threshold", "condition"), ""),
        "/model/threshold/condition",
        "non-empty text",
    ),
    ("reset", _set(("model", "reset"), 1), "/model/reset", "map state variables"),
    ("reset target", _set(("model", "reset", "z"), "0"), "/model/reset/z", "not a state variable"),
    ("units", _set(("units",), "mV"), "/units", "unit of every state"),
    ("units section", _set(("units", "state"), []), "/units/state", "must be an object"),
    ("missing unit", _delete(("units", "state", "w")), "/units/state", "w has no unit"),
    ("extra unit", _set(("units", "state", "q"), "mV"), "/units/state/q", "not a state entry"),
    ("unknown unit", _set(("units", "state", "v"), "banana"), "/units/state/v", "is not a unit"),
    ("broken unit", _set(("units", "state", "v"), "mV)"), "/units/state/v", "is not a unit"),
    ("scaled unit", _set(("units", "state", "v"), "2*mV"), "/units/state/v", "is not a unit"),
    ("open unit", _set(("units", "state", "v"), "("), "/units/state/v", "is not a unit"),
    ("current unit", _delete(("units", "current")), "/units/current", "non-empty text"),
    ("source", _set(("source",), "a paper"), "/source", "must cite"),
    ("citation", _set(("source", "citation"), ""), "/source/citation", "non-empty text"),
    ("doi", _set(("source", "doi"), 7), "/source/doi", "non-empty text"),
    ("source field", _set(("source", "isbn"), "x"), "/source/isbn", "unknown field"),
    ("assumptions", _set(("assumptions",), []), "/assumptions", "at least one entry"),
    ("assumption text", _set(("assumptions",), "one"), "/assumptions", "at least one entry"),
    ("author", _set(("authors",), [""]), "/authors/0", "an author must be non-empty text"),
    ("tests list", _set(("reference_tests",), {}), "/reference_tests", "must be a list"),
    ("test object", _set(("reference_tests", 0), "x"), "/reference_tests/0", "must be an object"),
    (
        "test name",
        _set(("reference_tests", 1, "name"), "rests without drive"),
        "/reference_tests/1/name",
        "a second test is named rests without drive",
    ),
    (
        "fractional steps",
        _set(("reference_tests", 0, "steps"), 10.5),
        "/reference_tests/0/steps",
        "whole number",
    ),
    (
        "too many steps",
        _set(("reference_tests", 0, "steps"), MAX_CANDIDATE_STEPS + 1),
        "/reference_tests/0/steps",
        "whole number",
    ),
    ("expect", _set(("reference_tests", 0, "expect"), {}), "/reference_tests/0/expect", "state"),
    (
        "unknown expectation",
        _set(("reference_tests", 1, "expect", "rate"), 3),
        "/reference_tests/1/expect/rate",
        "unknown expectation",
    ),
    (
        "final state",
        _set(("reference_tests", 0, "expect", "final_state"), []),
        "/reference_tests/0/expect/final_state",
        "must bound variables",
    ),
    (
        "bounds shape",
        _set(("reference_tests", 1, "expect", "spike_count"), {"at_least": 1}),
        "/reference_tests/1/expect/spike_count",
        "min and/or max",
    ),
    (
        "bounds order",
        _set(("reference_tests", 1, "expect", "spike_count"), {"min": 5, "max": 1}),
        "/reference_tests/1/expect/spike_count",
        "min exceeds max",
    ),
    (
        "bound value",
        _set(("reference_tests", 1, "expect", "spike_count"), {"min": "1"}),
        "/reference_tests/1/expect/spike_count/min",
        "min must be a number",
    ),
    (
        "refused by the DSL",
        _set(("model", "integration", "method"), "leapfrog"),
        "/model",
        "the Universal DSL refuses the model",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "location", "fragment"),
    [
        pytest.param(mutate, location, fragment, id=name)
        for name, mutate, location, fragment in CASES
    ],
)
def test_each_fault_is_reported_at_its_field(
    mutate: Mutation, location: str, fragment: str
) -> None:
    document = adex_candidate()
    mutate(document)
    diagnostics = _diagnostics(document)
    assert any(at == location and fragment in message for at, message in diagnostics), diagnostics
    assert not validate_candidate(document).valid


def test_too_many_reference_tests_are_refused() -> None:
    document = adex_candidate()
    first = document["reference_tests"][0]
    document["reference_tests"] = [
        {**first, "name": f"case {index}"} for index in range(MAX_REFERENCE_TESTS + 1)
    ]
    assert ("/reference_tests", f"at most {MAX_REFERENCE_TESTS} reference tests") in _diagnostics(
        document
    )


def test_every_fault_is_reported_at_once() -> None:
    document = adex_candidate()
    document["name"] = "AdExNeuron"
    document["units"]["state"]["v"] = "banana"
    document["authors"] = []
    locations = {location for location, _message in _diagnostics(document)}
    assert {"/name", "/units/state/v", "/authors"} <= locations


def test_a_parentless_candidate_needs_no_parent() -> None:
    document = adex_candidate()
    document["parent"] = None
    assert validate_candidate(document).valid


def test_a_pointer_escapes_slash_and_tilde_in_keys() -> None:
    document = adex_candidate()
    document["units"]["state"]["a/b~c"] = "mV"
    assert ("/units/state/a~1b~0c", "a/b~c is not a state entry") in _diagnostics(document)


def test_the_public_form_carries_each_located_diagnostic() -> None:
    document = adex_candidate()
    document["parent"] = "NoSuchNeuron"
    public = validate_candidate(document).to_public_dict()
    assert public["valid"] is False
    assert public["diagnostics"] == [
        {"location": "/parent", "message": "parent 'NoSuchNeuron' is not a catalogue model"}
    ]


def test_the_catalogue_checked_against_can_be_given() -> None:
    document = adex_candidate()
    assert not validate_candidate(document, catalogue=frozenset({"SlowAdaptationAdEx"})).valid
    assert validate_candidate(document, catalogue=frozenset({"AdExNeuron"})).valid


def test_a_model_without_threshold_or_reset_is_judged_by_the_dsl_alone() -> None:
    document = adex_candidate()
    del document["model"]["threshold"]
    del document["model"]["reset"]
    locations = [location for location, _message in _diagnostics(document)]
    # No field-level fault: whatever remains is the Universal DSL's own verdict.
    assert all(location == "/model" for location in locations)


def test_a_test_may_bound_the_final_state_alone() -> None:
    document = adex_candidate()
    del document["reference_tests"][0]["expect"]["spike_count"]
    assert validate_candidate(document).valid
