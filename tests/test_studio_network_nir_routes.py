# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Caller-safe NIR and graph-envelope route refusals

"""Post malformed real NIR files and envelopes through the Studio HTTP boundary."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Iterator
from typing import Any

import nir
import numpy as np
import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import create_population
from sc_neurocore.studio.network_nir import STUDIO_METADATA_KEY


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Serve the production router with the supported loopback security profile."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as http:
        yield http


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("caller-text-xyz!", "the NIR import is not valid base64"),
        ("é", "the NIR import is not valid base64"),
        (
            base64.b64encode(b"caller-text-xyz").decode("ascii"),
            "the file is not a readable NIR graph",
        ),
    ],
)
def test_unreadable_file_has_only_the_authored_reason(
    client: TestClient, content: str, reason: str
) -> None:
    """Base64 and HDF5 failures must not expose interpreter or library diagnostics."""
    response = client.post("/api/graph/import-nir", json={"content_base64": content})
    assert response.status_code == 422
    assert response.json() == {"detail": {"reason": reason}}
    assert "caller-text-xyz" not in response.text


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("graph-json", "the graph: its Studio metadata is not JSON"),
        ("node-json", "node a: its Studio metadata is not JSON"),
        ("missing-field", "the file's Studio metadata is missing a required field"),
        ("wrong-index", "the file's Studio metadata has a field with the wrong type"),
    ],
)
def test_malformed_file_metadata_has_only_the_authored_reason(
    client: TestClient, mutation: str, reason: str
) -> None:
    """Read edited real HDF5 metadata without leaking JSON, key or type errors."""
    population = create_population(count=2)
    population["id"] = "a"
    exported = client.post(
        "/api/graph/export-nir", json={"populations": [population], "projections": []}
    )
    assert exported.status_code == 200
    graph = nir.read(io.BytesIO(base64.b64decode(exported.json()["content_base64"])))
    if mutation == "graph-json":
        graph.metadata[STUDIO_METADATA_KEY] = "{caller-text-xyz"
    elif mutation == "node-json":
        graph.nodes["a"].metadata[STUDIO_METADATA_KEY] = "{caller-text-xyz"
    elif mutation == "missing-field":
        metadata = json.loads(graph.metadata[STUDIO_METADATA_KEY])
        del metadata["duration"]
        graph.metadata[STUDIO_METADATA_KEY] = json.dumps(metadata)
    else:
        metadata = json.loads(graph.nodes["a"].metadata[STUDIO_METADATA_KEY])
        metadata["index"] = {"caller-text-xyz": 1}
        metadata["position"] = None
        graph.nodes["a"].metadata[STUDIO_METADATA_KEY] = json.dumps(metadata)
    buffer = io.BytesIO()
    nir.write(buffer, graph)
    response = client.post(
        "/api/graph/import-nir",
        json={"content_base64": base64.b64encode(buffer.getvalue()).decode("ascii")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": {"reason": reason}}
    assert "caller-text-xyz" not in response.text


def test_envelope_malformed_format_has_a_fixed_reason(client: TestClient) -> None:
    """An array format must be refused instead of causing an unhashable-type error."""
    response = client.post(
        "/api/graph/import-nir", json={"format": ["caller-text-xyz"], "nodes": {}, "edges": []}
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": {"reason": "a required graph field is missing or has the wrong JSON type"}
    }


def test_envelope_authored_refusal_still_arrives_verbatim(client: TestClient) -> None:
    """A deliberately authored envelope rule remains useful at the HTTP surface."""
    response = client.post("/api/graph/import-nir", json={"nodes": {"a": {}}, "edges": [{}]})
    assert response.status_code == 422
    assert response.json() == {
        "detail": {"reason": "Graph envelope edge 0 source must be a non-empty string"}
    }


def test_export_model_constructor_diagnostic_is_not_a_refusal(client: TestClient) -> None:
    """The graph refuses invalid joint parameters without echoing constructor errors."""
    population: dict[str, Any] = create_population(count=2, params={"v_threshold": 0.0})
    population["id"] = "a"
    response = client.post(
        "/api/graph/export-nir", json={"populations": [population], "projections": []}
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "reason": "populations[0].params: Population a parameters: "
            "the model cannot accept these parameters"
        }
    }


@pytest.mark.parametrize("resistance", ["caller-text-xyz", 0.0])
def test_foreign_nir_invalid_values_have_a_fixed_reason(
    client: TestClient, resistance: str | float
) -> None:
    """Numeric conversion and reciprocal failures in a real foreign IF file are 422."""
    value = resistance.encode("ascii") if isinstance(resistance, str) else resistance
    graph = nir.NIRGraph(nodes={"a": nir.IF(r=np.array([value]), v_threshold=np.ones(1))}, edges=[])
    buffer = io.BytesIO()
    nir.write(buffer, graph)
    response = client.post(
        "/api/graph/import-nir",
        json={"content_base64": base64.b64encode(buffer.getvalue()).decode("ascii")},
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "reason": "the NIR graph has a missing field or a field with the wrong type or value"
        }
    }
    assert "caller-text-xyz" not in response.text
