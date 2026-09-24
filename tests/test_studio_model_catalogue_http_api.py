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
    assert adex["terminal_silicon_tier"] == "H2"
    assert adex["terminal_reason"].startswith("Q16.16 co-simulation")
    expif = next(m for m in models if m["name"] == "ExpIFNeuron")
    assert expif["family"] == "Integrate-and-Fire"
    assert expif["validation_metric"] == "parity"


def test_api_model_detail_endpoint_serves_descriptor(client: TestClient) -> None:
    response = client.get("/api/models/AdExNeuron")
    assert response.status_code == 200
    detail = response.json()
    assert detail["category_slug"] == "integrate-and-fire"
    assert "dynamics" in detail
    configuration = detail["compile_configuration"]
    contracts = configuration.pop("numeric_contracts")
    # Q8.8 tops out below 128, so the AdEx capacitance C=200 would wrap to -56.
    assert configuration == {
        "schema_name": "adex",
        "default_integrator": "euler",
        "integrators": ["euler", "rk4"],
        "cosim_integrators": ["euler"],
        "default_q_format": "Q16.16",
        "q_formats": ["Q16.16"],
    }
    assert list(contracts) == ["Q8.8", "Q16.16"]
    refused = contracts["Q8.8"]
    assert refused["representable"] is False
    assert "parameter C=200.0 becomes -56.0" in refused["refusal"]
    offered = contracts["Q16.16"]
    assert (offered["representable"], offered["refusal"]) == (True, "")
    assert offered["bit_true_mirror"]["available"] is True
    assert offered["bit_true_mirror"]["arithmetic"]["q_format"] == "Q16.16"


def test_api_expif_detail_serves_source_receipt(client: TestClient) -> None:
    response = client.get("/api/models/ExpIFNeuron")
    assert response.status_code == 200
    detail = response.json()
    assert detail["reproducibility"]["reference_config"].endswith(
        "reference_receipts/expif_fourcaud_trocme_2003.json"
    )
    assert "source profile" in detail["dynamics"]["v"]


def test_api_lapicque_detail_serves_source_receipt_and_h2_boundary(
    client: TestClient,
) -> None:
    response = client.get("/api/models/LapicqueNeuron")
    assert response.status_code == 200
    detail = response.json()
    assert detail["reproducibility"]["reference_config"].endswith(
        "reference_receipts/lapicque_1907.json"
    )
    assert "first candidate" in detail["dynamics"]["excited"]
    assert detail["silicon_label"] == "H2"
    assert detail["readiness"]["terminal_silicon_tier"] == "H2"
