# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Running candidates, their reference tests, the review packet and the routes

"""A candidate runs under its own profile, and only when it is valid.

Simulations are real Universal DSL runs; the routes are exercised through the
Studio application, so a refused candidate is checked as the editor receives it.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.candidate_package import MAX_CANDIDATE_STEPS, candidate_sha256
from sc_neurocore.studio.candidate_run import (
    NOT_ESTABLISHED,
    REVIEW_PACKET_SCHEMA_VERSION,
    CandidateRejected,
    MAX_TRACE_POINTS,
    review_packet,
    run_reference_tests,
    simulate_candidate,
)
from tests.studio_candidate_support import adex_candidate


def _diverging() -> dict[str, Any]:
    """A valid candidate whose state leaves the finite numbers within a few steps."""
    document = adex_candidate()
    document["model"]["dynamics"]["w"] = "w * w * 1e6 + 1e6"
    document["reference_tests"] = [
        {
            "name": "stays bounded",
            "current": 0.0,
            "steps": 200,
            "expect": {"final_state": {"w": {"max": 1.0}}},
        }
    ]
    return document


class TestSimulation:
    def test_a_long_run_is_strided_to_a_bounded_trace(self) -> None:
        document = adex_candidate()
        result = simulate_candidate(document, current=800.0, steps=12_000)

        assert result["candidate"] == "SlowAdaptationAdEx"
        assert result["candidate_sha256"] == candidate_sha256(document)
        assert result["units"] == {"current": "pA", "time": "ms"}
        assert result["profile"]["method"] == "euler"
        assert result["profile"]["dt"] == 0.1
        assert result["sample_every"] == 3
        assert len(result["trace"]["v"]) == 4_000 <= MAX_TRACE_POINTS
        assert result["spike_count"] == len(result["spike_steps"]) > 0
        assert result["diverged_at_step"] is None
        assert set(result["final_state"]) == {"v", "w"}

    def test_a_diverging_run_says_where_instead_of_returning_nonsense(self) -> None:
        result = simulate_candidate(_diverging(), current=0.0, steps=200)
        assert result["diverged_at_step"] is not None
        assert "'w' became non-finite" in result["divergence"]
        assert result["final_state"] is None

    @pytest.mark.parametrize("steps", [0, MAX_CANDIDATE_STEPS + 1])
    def test_a_run_outside_the_bound_is_refused(self, steps: int) -> None:
        with pytest.raises(ValueError, match="steps must be from 1 to"):
            simulate_candidate(adex_candidate(), current=0.0, steps=steps)

    def test_an_invalid_candidate_is_not_run(self) -> None:
        document = adex_candidate()
        document["units"]["state"]["v"] = "banana"
        with pytest.raises(CandidateRejected) as refused:
            simulate_candidate(document, current=0.0, steps=10)
        assert refused.value.validation["valid"] is False
        with pytest.raises(CandidateRejected):
            run_reference_tests(document)
        with pytest.raises(CandidateRejected):
            review_packet(document)


class TestReferenceTests:
    def test_each_expectation_states_what_was_observed(self) -> None:
        results = run_reference_tests(adex_candidate())
        assert [result["name"] for result in results] == [
            "rests without drive",
            "fires under drive",
        ]
        assert all(result["passed"] for result in results)
        rest = results[0]["checks"]
        assert [check["quantity"] for check in rest] == ["spike_count", "final_state.v"]
        assert rest[0]["observed"] == 0
        assert -66.0 <= rest[1]["observed"] <= -64.0

    def test_a_failed_expectation_fails_its_test(self) -> None:
        document = adex_candidate()
        document["reference_tests"][1]["expect"]["spike_count"] = {"min": 10_000}
        results = run_reference_tests(document)
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False
        assert results[1]["checks"][0]["held"] is False

    def test_a_diverged_run_holds_no_final_state_bound(self) -> None:
        (result,) = run_reference_tests(_diverging())
        assert result["diverged_at_step"] is not None
        assert result["checks"] == [
            {
                "quantity": "final_state.w",
                "bounds": {"max": 1.0},
                "observed": None,
                "held": False,
            }
        ]
        assert result["passed"] is False


def test_the_review_packet_binds_everything_under_one_digest() -> None:
    document = adex_candidate()
    packet = review_packet(document)

    assert packet["schema_version"] == REVIEW_PACKET_SCHEMA_VERSION
    assert packet["candidate"] == document
    assert packet["candidate_sha256"] == candidate_sha256(document)
    assert packet["validation"]["valid"] is True
    assert packet["diff"]["status"] == "compared"
    assert packet["reference_tests_passed"] is True
    assert packet["not_established"] == list(NOT_ESTABLISHED)
    assert set(packet["environment"]) == {"sc_neurocore", "python"}
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    assert packet["packet_sha256"] == candidate_sha256(body)
    assert review_packet(document)["packet_sha256"] == packet["packet_sha256"]


class TestRoutes:
    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        return TestClient(create_app(), base_url="http://127.0.0.1")

    def test_validation_reports_located_diagnostics_with_200(self, client: TestClient) -> None:
        document = adex_candidate()
        document["parent"] = "NoSuchNeuron"
        response = client.post("/api/candidates/validate", json={"candidate": document})
        assert response.status_code == 200
        assert response.json()["diagnostics"] == [
            {"location": "/parent", "message": "parent 'NoSuchNeuron' is not a catalogue model"}
        ]

    @pytest.mark.parametrize("route", ["diff", "simulate", "review-packet"])
    def test_acting_on_an_invalid_candidate_is_refused_with_the_diagnostics(
        self, client: TestClient, route: str
    ) -> None:
        document = adex_candidate()
        document["authors"] = []
        response = client.post(f"/api/candidates/{route}", json={"candidate": document})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["reason"] == "invalid_candidate"
        assert {"location": "/authors", "message": "authors must list at least one entry"} in (
            detail["validation"]["diagnostics"]
        )

    def test_a_valid_candidate_is_diffed_simulated_and_reviewed(self, client: TestClient) -> None:
        document = adex_candidate()
        diff = client.post("/api/candidates/diff", json={"candidate": document})
        run = client.post(
            "/api/candidates/simulate",
            json={"candidate": document, "current": 800.0, "steps": 2000},
        )
        packet = client.post("/api/candidates/review-packet", json={"candidate": document})

        assert diff.status_code == run.status_code == packet.status_code == 200
        assert diff.json()["parent_schema"] == "adex"
        assert run.json()["spike_count"] > 0
        assert packet.json()["reference_tests_passed"] is True

    def test_the_request_bounds_are_enforced_before_the_candidate_is_read(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/candidates/simulate",
            json={"candidate": {}, "steps": MAX_CANDIDATE_STEPS + 1},
        )
        assert response.status_code == 422
        extra = client.post("/api/candidates/validate", json={"candidate": {}, "extra": 1})
        assert extra.status_code == 422
