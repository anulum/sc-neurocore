# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused Studio model catalogue contracts

"""Focused descriptor-backed model catalogue contracts."""

from .studio_model_catalogue_support import *


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
