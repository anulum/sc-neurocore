# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the Studio catalogue query

"""The catalogue query filters on verified readiness and counts facets that drill down.

The filtering rules are held on listings built here, so each case states the
corpus it reasons about; the same code is then run on the registered catalogue
and through the HTTP route.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio import catalogue_query
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.catalogue_query import (
    CATALOGUE_QUERY_SCHEMA_VERSION,
    CatalogueQuery,
    CatalogueQueryRejected,
    query_catalogue,
)
from sc_neurocore.studio.model_catalogue import get_model_detail, list_models, model_facets


def _model(name: str, **fields: Any) -> dict[str, Any]:
    model: dict[str, Any] = {
        "name": name,
        "public_label": "",
        "aliases": [],
        "family": "Integrate-and-Fire",
        "category": "Integrate-and-Fire",
        "description": "",
        "behavior_tags": [],
        "identity_kind": "source-literature",
        "metadata_state": "available",
        "science_tier": 0,
        "verified_science_tier": 0,
        "verified_silicon_tier": None,
        "is_perfect_verified": False,
    }
    model.update(fields)
    return model


_LISTING = [
    _model("Lif", behavior_tags=["tonic"], science_tier=5, verified_science_tier=2),
    _model(
        "Adex",
        behavior_tags=["adapting", "bursting"],
        verified_science_tier=5,
        verified_silicon_tier=2,
        is_perfect_verified=True,
        aliases=["AdaptiveExponential"],
    ),
    _model("Hh", family="Conductance-based", verified_science_tier=3, verified_silicon_tier=0),
    _model(
        "ScLif",
        identity_kind="sc-compatibility",
        public_label="SC leaky integrator",
        verified_science_tier=3,
    ),
    _model("Broken", family="unknown", metadata_state="invalid"),
]


def _names(query: CatalogueQuery) -> frozenset[str]:
    index = catalogue_query._catalogue_index(_LISTING)
    return catalogue_query._admitted(index, query)


class TestFilters:
    def test_readiness_floors_read_the_verified_tier_not_the_declared_one(self) -> None:
        """Lif declares S5 and is proven only at S2; it is not "at least S3"."""
        assert _names(CatalogueQuery(min_verified_science=3)) == {"Adex", "Hh", "ScLif"}

    def test_a_model_not_enrolled_on_silicon_meets_no_silicon_floor(self) -> None:
        assert _names(CatalogueQuery(min_verified_silicon=1)) == {"Adex"}

    def test_verified_perfect_only_keeps_what_the_receipts_prove(self) -> None:
        assert _names(CatalogueQuery(verified_perfect_only=True)) == {"Adex"}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("adaptiveexp", {"Adex"}), ("SC LEAKY", {"ScLif"}), ("conductance", {"Hh"})],
    )
    def test_text_matches_name_label_alias_and_family(self, text, expected) -> None:
        assert _names(CatalogueQuery(text=text)) == expected

    def test_facet_filters_intersect(self) -> None:
        assert _names(CatalogueQuery(family="Integrate-and-Fire", behavior="bursting")) == {"Adex"}
        assert _names(CatalogueQuery(identity_kind="sc-compatibility")) == {"ScLif"}
        assert _names(CatalogueQuery(metadata_state="invalid")) == {"Broken"}
        assert _names(CatalogueQuery(family="no such family")) == frozenset()

    def test_facet_counts_drill_down_without_zeroing_the_other_values(self) -> None:
        """Choosing a family still shows what each other family would give."""
        index = catalogue_query._catalogue_index(_LISTING)
        query = CatalogueQuery(family="Conductance-based", min_verified_science=3)

        families = catalogue_query._counts(
            index, catalogue_query._admitted(index, query, skip="family"), "family"
        )
        behaviors = catalogue_query._counts(
            index, catalogue_query._admitted(index, query, skip="behavior"), "behavior"
        )

        assert families == {"Conductance-based": 1, "Integrate-and-Fire": 2}
        assert behaviors == {}

    def test_the_index_is_rebuilt_only_for_a_new_listing(self) -> None:
        first = catalogue_query._catalogue_index(_LISTING)
        assert catalogue_query._catalogue_index(_LISTING) is first
        assert catalogue_query._catalogue_index(list(_LISTING)) is not first


class TestParameters:
    def test_query_parameters_become_a_query(self) -> None:
        query = CatalogueQuery.from_params(
            {
                "text": "lif",
                "family": "F",
                "behavior": "tonic",
                "identity_kind": "source-literature",
                "metadata_state": "available",
                "min_verified_science": "3",
                "min_verified_silicon": "1",
                "verified_perfect_only": "true",
            }
        )
        assert query == CatalogueQuery(
            text="lif",
            family="F",
            behavior="tonic",
            identity_kind="source-literature",
            metadata_state="available",
            min_verified_science=3,
            min_verified_silicon=1,
            verified_perfect_only=True,
        )
        assert CatalogueQuery.from_params({}) == CatalogueQuery()

    @pytest.mark.parametrize(
        ("params", "reason"),
        [
            ({"tier": "3"}, "unknown catalogue query parameter: tier"),
            ({"min_verified_science": "6"}, "min_verified_science must be an integer from 0 to 5"),
            ({"min_verified_science": "-1"}, "min_verified_science must be an integer"),
            ({"min_verified_silicon": "H2"}, "min_verified_silicon must be an integer"),
            ({"verified_perfect_only": "yes"}, "verified_perfect_only must be true or false"),
        ],
    )
    def test_what_cannot_be_a_query_is_refused(self, params, reason) -> None:
        with pytest.raises(CatalogueQueryRejected, match=reason):
            CatalogueQuery.from_params(params)


class TestRegisteredCatalogue:
    def test_an_empty_query_returns_the_whole_corpus_at_its_revision(self) -> None:
        result = query_catalogue(CatalogueQuery())
        models = list_models()

        assert result["schema_version"] == CATALOGUE_QUERY_SCHEMA_VERSION
        assert result["total"] == result["matched"] == len(models)
        assert result["models"] == sorted(model["name"] for model in models)
        assert result["corpus_revision"] == model_facets()["corpus_revision"]
        assert sum(result["facets"]["identity_kind"].values()) == len(models)

    def test_every_matched_model_is_proven_at_the_floor(self) -> None:
        by_name = {model["name"]: model for model in list_models()}
        result = query_catalogue(CatalogueQuery(min_verified_science=3))

        assert result["matched"] > 0
        assert all(by_name[name]["verified_science_tier"] >= 3 for name in result["models"])
        declared_only = {
            name
            for name, model in by_name.items()
            if model["science_tier"] >= 3 and model["verified_science_tier"] < 3
        }
        assert not declared_only & set(result["models"])

    def test_the_list_carries_the_detail_views_verified_judgement(self) -> None:
        for model in list_models()[:12]:
            detail = get_model_detail(model["name"])
            assert detail is not None
            readiness = detail.get("readiness")
            expected = readiness["is_perfect_verified"] if readiness else False
            assert model["is_perfect_verified"] is expected, model["name"]


class TestRoute:
    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        return TestClient(create_app(), base_url="http://127.0.0.1")

    def test_the_route_answers_a_query(self, client: TestClient) -> None:
        response = client.get("/api/models/query", params={"min_verified_science": "3"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["matched"] == query_catalogue(CatalogueQuery(min_verified_science=3))["matched"]

    def test_the_route_names_why_a_query_is_refused(self, client: TestClient) -> None:
        response = client.get("/api/models/query", params={"min_verified_science": "9"})
        assert response.status_code == 422
        assert "from 0 to 5" in response.json()["detail"]["reason"]
