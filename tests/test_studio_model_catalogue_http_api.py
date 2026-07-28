# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused Studio model catalogue contracts

"""Focused descriptor-backed model catalogue contracts."""

from .studio_model_catalogue_support import *


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
    assert adex["validation_metric"] == "parity"
    assert adex["integration_method"] == "euler"
    assert adex["terminal_silicon_tier"] == "H1"
    assert adex["terminal_reason"].startswith("Point-neuron schema")


def test_api_model_detail_endpoint_serves_descriptor(client: TestClient) -> None:
    response = client.get("/api/models/AdExNeuron")
    assert response.status_code == 200
    detail = response.json()
    assert detail["category_slug"] == "integrate-and-fire"
    assert "dynamics" in detail
    assert detail["compile_configuration"] == {
        "schema_name": "adex",
        "default_integrator": "euler",
        "integrators": ["euler", "rk4"],
        "default_q_format": "Q8.8",
        "q_formats": ["Q8.8", "Q16.16"],
    }
