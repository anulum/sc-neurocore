# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Network Canvas (Block 5)

"""Provide real Studio HTTP fixtures for graph authoring and simulation."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import (
    GRAPH_ENVELOPE_FORMAT,
    GRAPH_ENVELOPE_VERSION,
    LEGACY_GRAPH_ENVELOPE_FORMAT,
    available_models,
    create_population,
    create_projection,
    envelope_to_graph,
    graph_to_envelope,
    simulate_graph,
    validate_graph,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide the real Studio API for graph construction and simulation."""
    return TestClient(create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"})


__all__ = [
    "annotations",
    "pytest",
    "fastapi",
    "TestClient",
    "create_app",
    "available_models",
    "create_population",
    "create_projection",
    "GRAPH_ENVELOPE_FORMAT",
    "GRAPH_ENVELOPE_VERSION",
    "LEGACY_GRAPH_ENVELOPE_FORMAT",
    "graph_to_envelope",
    "envelope_to_graph",
    "simulate_graph",
    "validate_graph",
    "client",
]
