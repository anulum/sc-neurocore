# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What a population of one model may be given

"""An editor driven by a contract is only as honest as the contract it reads.

The canvas creates a population with a model's defaults and then has to let a
user change them. It cannot do that from a list of names: which constructor
fields are numerically overridable, their kind, their default, and the reason
each other field is not an input are all decided by the run contract. A browser
that guessed would be a second implementation of it, free to drift from the one
that validates the graph.

The failure these cases exist to prevent is subtler than a wrong list: a
contract that offers a field the graph then refuses. ``dt`` is overridable on
the class and refused as an override by the run contract, so a contract that
simply reported the class's overridable fields would have offered an input that
is rejected every time it is used.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.model_run_contract import DT_OVERRIDE_REASON
from sc_neurocore.studio.network_graph import (
    POPULATION_MODEL_CONTRACT_VERSION,
    available_models,
    population_model_admission,
    population_model_contract,
)
from sc_neurocore.studio.network_graph_spec import graph_issues

MODEL = "SCLapicqueLIFNeuron"


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A Studio client for the graph routes."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


def graph_with(params: dict[str, Any]) -> dict[str, Any]:
    """Return a one-population graph carrying ``params``."""
    return {
        "dt": 0.1,
        "duration": 100.0,
        "populations": [
            {
                "count": 10,
                "drive": {"kind": "none"},
                "id": "p1",
                "label": "p1",
                "model": MODEL,
                "neuron_type": "excitatory",
                "params": params,
                "position": {"x": 0, "y": 0},
                "type": "population",
            }
        ],
        "projections": [],
        "seed": 42,
    }


class TestWhatTheContractStates:
    def test_it_carries_its_own_version(self) -> None:
        contract = population_model_contract(MODEL)

        assert contract is not None
        assert contract["schema_version"] == POPULATION_MODEL_CONTRACT_VERSION
        assert contract["model"] == MODEL

    def test_every_parameter_carries_its_kind_and_default(self) -> None:
        contract = population_model_contract(MODEL)

        assert contract is not None
        assert contract["parameters"]
        for parameter in contract["parameters"]:
            assert set(parameter) == {"default", "kind", "name"}
            assert parameter["kind"] in ("float", "int")

    def test_parameters_are_ordered_so_an_editor_is_stable(self) -> None:
        contract = population_model_contract(MODEL)

        assert contract is not None
        names = [parameter["name"] for parameter in contract["parameters"]]
        assert names == sorted(names)

    def test_it_names_the_drive_parameter_the_protocol_delivers_through(self) -> None:
        contract = population_model_contract(MODEL)

        assert contract is not None
        assert contract["drive"]["kind"] == "float"
        assert isinstance(contract["drive"]["parameter"], str)
        assert contract["drive"]["parameter"]

    def test_a_field_that_is_not_an_input_says_why(self) -> None:
        contract = population_model_contract(MODEL)

        assert contract is not None
        reasons = {entry["name"]: entry["reason"] for entry in contract["unsupported"]}
        assert reasons
        assert all(reason for reason in reasons.values())


class TestTheContractMatchesWhatTheGraphAccepts:
    def test_every_offered_parameter_is_accepted_by_the_graph(self) -> None:
        """The point of the contract: nothing it offers is refused on use."""
        contract = population_model_contract(MODEL)

        assert contract is not None
        for parameter in contract["parameters"]:
            value = parameter["default"] if parameter["default"] is not None else 1.0
            issues = graph_issues(graph_with({parameter["name"]: value}))
            assert issues == [], (parameter, [issue.message for issue in issues])

    def test_dt_is_not_offered_and_says_why_in_the_run_contract_s_own_words(self) -> None:
        contract = population_model_contract(MODEL)

        assert contract is not None
        assert "dt" not in [parameter["name"] for parameter in contract["parameters"]]
        reasons = {entry["name"]: entry["reason"] for entry in contract["unsupported"]}
        assert reasons["dt"] == DT_OVERRIDE_REASON

    def test_the_graph_refuses_dt_with_that_same_reason(self) -> None:
        """One wording, so the contract and the refusal cannot drift apart."""
        issues = graph_issues(graph_with({"dt": 0.5}))

        assert [issue.field for issue in issues] == ["populations[0].params.dt"]
        assert DT_OVERRIDE_REASON in issues[0].message

    def test_a_field_the_contract_does_not_offer_is_refused_by_the_graph(self) -> None:
        issues = graph_issues(graph_with({"not_a_parameter": 1.0}))

        assert [issue.field for issue in issues] == ["populations[0].params.not_a_parameter"]


class TestAModelThatCannotFormAPopulation:
    def test_it_has_no_contract_rather_than_an_empty_one(self) -> None:
        """An empty contract would read as a model with no parameters."""
        refused = [
            name
            for name in ("AkidaNeuron", "LoihiNeuron", "not-a-model")
            if population_model_admission(name) is not None
        ]
        assert refused, "no inadmissible model to check"
        for name in refused:
            assert population_model_contract(name) is None

    def test_every_admissible_model_has_a_contract(self) -> None:
        for name in available_models():
            assert population_model_contract(name) is not None, name


class TestOverHttp:
    def test_the_route_answers_the_contract(self, client: TestClient) -> None:
        response = client.get(f"/api/graph/models/{MODEL}")

        assert response.status_code == 200
        assert response.json() == population_model_contract(MODEL)

    def test_a_model_that_cannot_form_a_population_answers_404(self, client: TestClient) -> None:
        response = client.get("/api/graph/models/AkidaNeuron")

        assert response.status_code == 404

    def test_an_unknown_name_answers_404_without_a_path(self, client: TestClient) -> None:
        response = client.get("/api/graph/models/not-a-model")

        assert response.status_code == 404
        assert "/" not in response.json()["detail"].replace("'not-a-model'", "")

    def test_the_listing_and_the_contract_agree(self, client: TestClient) -> None:
        listed = client.get("/api/graph/models").json()

        assert listed
        for name in listed[:5]:
            assert client.get(f"/api/graph/models/{name}").status_code == 200
