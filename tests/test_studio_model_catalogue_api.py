# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Descriptor-backed catalogue API tests

"""Tests for the descriptor-backed model catalogue API."""

from __future__ import annotations

from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.models import (
    _introspected_summary,
    get_model_detail,
    list_models,
    model_documentation,
    model_facets,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _adex_summary() -> dict[str, object]:
    return next(m for m in list_models() if m["name"] == "AdExNeuron")


def test_list_models_serves_declared_family_and_provenance() -> None:
    summary = _adex_summary()
    assert summary["family"] == "Integrate-and-Fire"
    assert summary["category"] == "Integrate-and-Fire"
    assert summary["category_slug"] == "integrate-and-fire"
    assert summary["category_source"] == "declared"
    provenance = cast(dict[str, object], summary["provenance"])
    assert provenance["doi"] == "10.1152/jn.00686.2005"


def test_no_model_falls_into_an_other_bucket() -> None:
    """Every model now declares a real family — the 'Other' bucket is gone."""

    categories = {str(m["category"]) for m in list_models()}
    assert "Other" not in categories
    assert all(m["category_source"] == "declared" for m in list_models())


def test_get_model_detail_serves_descriptor_parameters_and_dynamics() -> None:
    detail = get_model_detail("AdExNeuron")
    assert detail is not None
    param = detail["params"][0]
    assert set(param) >= {"name", "default", "unit", "range", "meaning"}
    assert "v" in detail["dynamics"]
    assert any(b["name"] == "python" for b in detail["backends"])
    assert detail["family"] == "Integrate-and-Fire"


def test_model_facets_cover_the_whole_catalogue() -> None:
    facets = model_facets()
    assert facets["total"] == len(list_models())
    assert sum(f["count"] for f in facets["families"]) == facets["total"]
    families = {f["family"] for f in facets["families"]}
    assert "Cerebellar" in families
    assert "Integrate-and-Fire" in families


def test_introspected_fallback_flags_inferred_category() -> None:
    """A model without a descriptor falls back to an inferred category."""

    summary = _introspected_summary("AdExNeuron")
    assert summary["category_source"] == "inferred"
    assert summary["name"] == "AdExNeuron"
    assert summary["n_params"] >= 1


def test_api_models_facets_endpoint(client: TestClient) -> None:
    response = client.get("/api/models/facets")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] >= 150
    assert any(f["family"] == "Cerebellar" for f in body["families"])


def test_api_models_endpoint_serves_family(client: TestClient) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    adex = next(m for m in models if m["name"] == "AdExNeuron")
    assert adex["family"] == "Integrate-and-Fire"


def test_api_model_detail_endpoint_serves_descriptor(client: TestClient) -> None:
    response = client.get("/api/models/AdExNeuron")
    assert response.status_code == 200
    detail = response.json()
    assert detail["category_slug"] == "integrate-and-fire"
    assert "dynamics" in detail


def test_model_documentation_serves_reference_markdown() -> None:
    doc = model_documentation("AdExNeuron")
    assert doc is not None
    assert doc["name"] == "AdExNeuron"
    assert doc["slug"] == "models/adex"
    assert doc["markdown"].lstrip().startswith("# AdExNeuron")
    assert model_documentation("DefinitelyNotARealModel") is None


def test_api_model_doc_endpoint(client: TestClient) -> None:
    ok = client.get("/api/models/HodgkinHuxleyNeuron/doc")
    assert ok.status_code == 200, ok.text
    assert ok.json()["markdown"].lstrip().startswith("# HodgkinHuxleyNeuron")
    missing = client.get("/api/models/DefinitelyNotARealModel/doc")
    assert missing.status_code == 404
