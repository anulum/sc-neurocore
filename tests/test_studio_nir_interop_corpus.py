# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR files written by other frameworks, and what the Studio does with them

"""An acceptance corpus of NIR graphs other tools wrote, with the Studio's recorded answer.

The files are the ones published with the NIR paper, at a pinned commit of the
NIR repository, written by Norse, Rockpool, Sinabs and snnTorch. The manifest
records, for each, what it holds and what the Studio's importer does with it.
Every file here is refused today, each for a stated reason; a change in what
the importer reads shows up as a failure here and is recorded by updating the
manifest, never by weakening a case.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

nir = pytest.importorskip("nir")
pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_nir import NIRMappingRefused, nir_file_to_graph

CORPUS = Path(__file__).parent / "fixtures" / "nir_interop"
MANIFEST: dict[str, Any] = json.loads((CORPUS / "corpus.json").read_text(encoding="utf-8"))
FILES: list[dict[str, Any]] = MANIFEST["files"]
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _ids(entry: dict[str, Any]) -> str:
    return str(entry["file"])


def test_the_corpus_names_its_source_and_carries_its_licence() -> None:
    source = MANIFEST["source"]
    assert MANIFEST["schema"] == "sc-neurocore.nir-interop-corpus.v1"
    assert source["repository"] == "https://github.com/neuromorphs/NIR"
    assert len(source["commit"]) == 40
    assert source["licence"] == "BSD-3-Clause"
    licence = (REPO_ROOT / source["licence_text"]).read_text(encoding="utf-8")
    assert "Copyright (c) 2023, NIR Team." in licence
    assert "Redistributions of source code must retain" in licence
    reuse = (REPO_ROOT / "REUSE.toml").read_text(encoding="utf-8")
    assert '"tests/fixtures/nir_interop/*.nir"' in reuse
    assert sorted(path.name for path in CORPUS.glob("*.nir")) == sorted(e["file"] for e in FILES)


@pytest.mark.parametrize("entry", FILES, ids=_ids)
def test_each_file_is_the_published_bytes(entry: dict[str, Any]) -> None:
    content = (CORPUS / entry["file"]).read_bytes()
    assert len(content) == entry["bytes"]
    assert hashlib.sha256(content).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("entry", [e for e in FILES if e["readable_by_nir"]], ids=_ids)
def test_the_reference_package_reads_the_nodes_the_manifest_lists(entry: dict[str, Any]) -> None:
    graph = nir.read(str(CORPUS / entry["file"]))
    assert sorted({type(node).__name__ for node in graph.nodes.values()}) == entry["nir_node_types"]


@pytest.mark.parametrize("entry", FILES, ids=_ids)
def test_the_studio_answers_each_file_as_recorded(entry: dict[str, Any]) -> None:
    content = base64.b64encode((CORPUS / entry["file"]).read_bytes()).decode()
    assert entry["studio_import"] == "refused"
    with pytest.raises(NIRMappingRefused) as refusal:
        nir_file_to_graph(content)
    assert entry["studio_reason_contains"] in str(refusal.value)


@pytest.mark.parametrize("entry", FILES, ids=_ids)
def test_the_import_route_refuses_with_the_reason_not_a_server_error(
    client: TestClient, entry: dict[str, Any]
) -> None:
    content = base64.b64encode((CORPUS / entry["file"]).read_bytes()).decode()
    response = client.post("/api/graph/import-nir", json={"content_base64": content})
    assert response.status_code == 422
    assert entry["studio_reason_contains"] in response.json()["detail"]["reason"]
