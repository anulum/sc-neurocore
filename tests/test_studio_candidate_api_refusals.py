# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual candidate API diagnostic provenance

"""Exercise candidate refusal bodies through the real Studio API router."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from tests.studio_candidate_support import adex_candidate

_DEEP_EXPRESSION = " + ".join(["v"] * 1100)
#: Refusal raised when the interpreter's parser itself gives up on the expression.
TOO_DEEP_TO_PARSE = "Equation expression is too deep to parse"


@pytest.fixture
def candidate_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Run the actual API with isolated job and audit storage.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned directory for jobs and audit records.
    monkeypatch : pytest.MonkeyPatch
        Restore the storage environment after the client closes.

    Yields
    ------
    starlette.testclient.TestClient
        Live in-process client with the application's lifespan active.
    """
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        yield client


@pytest.mark.parametrize("route", ["validate", "diff", "simulate", "review-packet"])
@pytest.mark.parametrize(
    ("path", "value", "location", "message"),
    [
        (("units", "state", "v"), "(", "/units/state/v", "'(' is not a unit"),
        (("units", "state", "v"), "mV)", "/units/state/v", "'mV)' is not a unit"),
        (("units", "state", "v"), "2*mV", "/units/state/v", "'2*mV' is not a unit"),
        (("units", "state", "v"), "mV/0", "/units/state/v", "'mV/0' is not a unit"),
        (("units", "state", "v"), "mV**(1j)", "/units/state/v", "'mV**(1j)' is not a unit"),
        (
            ("model", "dynamics", "v"),
            _DEEP_EXPRESSION,
            "/model/dynamics/v",
            f"Equation AST depth 1102 exceeds limit 20: {_DEEP_EXPRESSION!r}",
        ),
        (
            ("model", "dynamics", "v"),
            " + ".join(["v"] * 10000),
            "/model/dynamics/v",
            TOO_DEEP_TO_PARSE,
        ),
        (
            ("model", "dynamics", "v"),
            "sqrt()",
            "/model/dynamics/v",
            "function 'sqrt' does not accept 0 positional arguments",
        ),
        (
            ("model", "dynamics", "v"),
            "v.missing",
            "/model/dynamics/v",
            "Attribute 'missing' is not part of the equation grammar: 'v.missing'",
        ),
        (
            ("model", "dynamics", "v"),
            "math.exp(v)",
            "/model/dynamics/v",
            "Only a function under a plain name can be called in an equation: 'math.exp(v)'",
        ),
        (
            ("model", "integration", "substeps"),
            "caller-text-xyz",
            "/model",
            "the Universal DSL refuses the model: a model field is missing or invalid",
        ),
        (("model", "state", "v"), 10**400, "/model/state/v", "state.v must be finite"),
        (
            ("source", "citation"),
            "citation-" + chr(0xD800),
            "",
            "candidate text must contain valid Unicode",
        ),
        (
            ("model", "dynamics", "v"),
            chr(0xD800),
            "/model/dynamics/v",
            "the expression cannot be read",
        ),
    ],
)
def test_candidate_routes_refuse_real_parser_and_encoding_faults(
    candidate_client: TestClient,
    route: str,
    path: tuple[str, ...],
    value: object,
    location: str,
    message: str,
) -> None:
    """Keep located admission refusals free of interpreter diagnostics.

    Parameters
    ----------
    candidate_client : starlette.testclient.TestClient
        Actual candidate API with isolated storage.
    route : str
        Validation, comparison, simulation or review-packet endpoint.
    path : tuple of str
        Nested candidate field changed before admission.
    value : object
        Invalid unit, expression, model field or attribution text.
    location : str
        JSON pointer expected in the authored diagnostic.
    message : str
        Exact refusal text, including the parser-depth alternative.
    """
    document = adex_candidate()
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    response = candidate_client.post(
        f"/api/candidates/{route}",
        content=json.dumps({"candidate": document}, ensure_ascii=True).encode("ascii"),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == (200 if route == "validate" else 422)
    body = response.json()
    if route == "validate":
        validation = body
    else:
        assert body["detail"]["reason"] == "invalid_candidate"
        validation = body["detail"]["validation"]
    assert validation["valid"] is False
    if message == TOO_DEEP_TO_PARSE:
        # Which gate refuses a ten-thousand-term sum depends on the interpreter:
        # the parser's own recursion limit, or the validator's depth limit where
        # the parser accepts it. Both are authored refusals at this location.
        located = [
            row["message"] for row in validation["diagnostics"] if row["location"] == location
        ]
        assert any(
            text == TOO_DEEP_TO_PARSE or text.startswith("Equation AST depth ") for text in located
        )
    else:
        assert {"location": location, "message": message} in validation["diagnostics"]
    assert "TokenInfo" not in response.text
    assert "unexpected EOF" not in response.text
    assert "invalid literal" not in response.text
    assert "caller-text-xyz" not in response.text


def test_unicode_attribution_survives_valid_simulation_and_review(
    candidate_client: TestClient,
) -> None:
    """Retain Unicode attribution in an actual positive candidate round trip.

    Parameters
    ----------
    candidate_client : starlette.testclient.TestClient
        Actual candidate API with isolated storage.
    """
    document = adex_candidate()
    document["source"]["citation"] += " — Šotek"
    validation = candidate_client.post("/api/candidates/validate", json={"candidate": document})
    assert validation.status_code == 200
    assert validation.json()["valid"] is True
    simulation = candidate_client.post(
        "/api/candidates/simulate", json={"candidate": document, "steps": 10}
    )
    assert simulation.status_code == 200
    assert simulation.json()["diverged_at_step"] is None
    review = candidate_client.post("/api/candidates/review-packet", json={"candidate": document})
    assert review.status_code == 200
    assert review.json()["candidate"] == document
    assert review.json()["reference_tests_passed"] is True
    assert review.json()["candidate_sha256"] == validation.json()["candidate_sha256"]


@pytest.mark.parametrize("route", ["simulate", "review-packet"])
@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("1 / 0", "the equation divides by zero"),
        ("min(v)", "the equation value has an invalid type"),
        ("v[0]", "the equation value has an invalid type"),
        ("[v].real", "the equation reads an unavailable attribute"),
        ("[v]", "the equation value has an invalid type"),
        ("v + exp", "the equation value has an invalid type"),
    ],
)
def test_real_expression_failure_reports_a_failed_run_without_interpreter_text(
    candidate_client: TestClient, route: str, expression: str, reason: str
) -> None:
    """Report a failed admitted calculation while preserving review packet binding.

    Parameters
    ----------
    candidate_client : starlette.testclient.TestClient
        Actual candidate API with isolated storage.
    route : str
        Simulation or review-packet endpoint.
    expression : str
        Admitted expression that fails when evaluated against the model state.
    reason : str
        Authored runtime failure without Python's exception wording.
    """
    document = adex_candidate()
    document["model"]["dynamics"]["v"] = expression
    response = candidate_client.post(f"/api/candidates/{route}", json={"candidate": document})
    assert response.status_code == 200
    body = response.json()
    if route == "simulate":
        assert body["diverged_at_step"] == 0
        assert body["final_state"] is None
        assert body["divergence"] == reason
    else:
        assert body["reference_tests_passed"] is False
        assert body["candidate"] == document
        assert all(test["diverged_at_step"] == 0 for test in body["reference_tests"])
    assert "is not iterable" not in response.text
    assert "object has no attribute" not in response.text
    assert "float() argument" not in response.text
    assert "unsupported operand" not in response.text
