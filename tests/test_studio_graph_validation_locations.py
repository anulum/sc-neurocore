# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Every graph validation failure carries where it happened

"""A refusal a caller cannot place is a refusal they have to hunt for.

Validation already reported every failure at once, which is right: nobody
should fix one field, resubmit, and meet the next. It also knew the request
field each message came from — ``projections[2].delay`` — and the route threw
that half away, leaving an editor to match prose against a diagram to find
which of six projections was meant.

These cases hold the route to reporting both, to reporting them for the same
failures in the same order, and to naming fields an editor can actually index
by.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph_spec import graph_issues


def population(identifier: str, **overrides: Any) -> dict[str, Any]:
    """Return one valid population, overridden where a case needs a defect."""
    node = {
        "count": 10,
        "drive": {"kind": "none"},
        "id": identifier,
        "label": identifier,
        "model": "SCLapicqueLIFNeuron",
        "neuron_type": "excitatory",
        "params": {},
        "position": {"x": 0, "y": 0},
        "type": "population",
    }
    node.update(overrides)
    return node


def projection(identifier: str, **overrides: Any) -> dict[str, Any]:
    """Return one valid projection, overridden where a case needs a defect."""
    edge = {
        "delay": 0.0,
        "id": identifier,
        "probability": 0.1,
        "rule": "random",
        "source": "p1",
        "target": "p2",
        "weight": 0.5,
    }
    edge.update(overrides)
    return edge


def graph(**overrides: Any) -> dict[str, Any]:
    """Return a valid two-population graph, overridden per case."""
    document = {
        "dt": 0.1,
        "duration": 100.0,
        "populations": [population("p1"), population("p2", neuron_type="inhibitory")],
        "projections": [projection("e1")],
        "seed": 42,
    }
    document.update(overrides)
    return document


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A Studio client for the graph routes."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


def _validate(client: TestClient, document: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/api/graph/validate", json=document)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


class TestAValidGraph:
    def test_it_is_valid_and_carries_no_issues(self, client: TestClient) -> None:
        body = _validate(client, graph())

        assert body["valid"] is True
        assert body["errors"] == []
        assert body["issues"] == []


class TestWhereAFailureHappened:
    def test_a_projection_failure_names_the_projection_by_index(self, client: TestClient) -> None:
        body = _validate(client, graph(projections=[projection("e1", weight=-4.0)]))

        assert body["valid"] is False
        assert [issue["field"] for issue in body["issues"]] == ["projections[0].weight"]

    def test_a_population_failure_names_the_population_by_index(self, client: TestClient) -> None:
        body = _validate(client, graph(populations=[population("p1"), population("p2", count=0)]))

        assert "populations[1].count" in [issue["field"] for issue in body["issues"]]

    def test_a_failure_about_the_run_names_its_own_field(self, client: TestClient) -> None:
        body = _validate(client, graph(dt=0.0))

        assert "dt" in [issue["field"] for issue in body["issues"]]

    def test_a_parameter_failure_names_the_parameter(self, client: TestClient) -> None:
        body = _validate(
            client, graph(populations=[population("p1", params={"not_a_parameter": 1.0})])
        )

        fields = [issue["field"] for issue in body["issues"]]
        assert any(field.startswith("populations[0].params") for field in fields), fields

    def test_the_index_is_the_position_in_the_request(self, client: TestClient) -> None:
        """An editor indexes the array it sent; anything else misplaces the message."""
        body = _validate(
            client,
            graph(
                projections=[
                    projection("e1"),
                    projection("e2", weight=-4.0),
                    projection("e3", delay=0.05),
                ]
            ),
        )
        located = {issue["field"]: issue["message"] for issue in body["issues"]}

        assert "e2" in located["projections[1].weight"]
        assert "e3" in located["projections[2].delay"]


class TestBothFormsAgree:
    def test_every_message_appears_in_both_lists_in_one_order(self, client: TestClient) -> None:
        """A caller reading either form must see the same refusal."""
        body = _validate(
            client,
            graph(
                dt=0.0,
                projections=[projection("e1", weight=-4.0, probability=1.5)],
            ),
        )

        assert body["errors"] == [issue["message"] for issue in body["issues"]]
        assert len(body["errors"]) > 1

    def test_every_failure_is_reported_at_once(self, client: TestClient) -> None:
        """Three defects in one projection, not one refusal at a time."""
        body = _validate(
            client,
            graph(projections=[projection("e1", weight=-4.0, probability=1.5, delay=0.05)]),
        )

        assert sorted(issue["field"] for issue in body["issues"]) == [
            "projections[0].delay",
            "projections[0].probability",
            "projections[0].weight",
        ]

    def test_the_route_reports_what_the_validator_found(self, client: TestClient) -> None:
        """The route adds the location and changes nothing else."""
        document = graph(projections=[projection("e1", weight=-4.0, delay=0.05)])
        body = _validate(client, document)

        assert body["issues"] == [
            {"field": issue.field, "message": issue.message} for issue in graph_issues(document)
        ]
