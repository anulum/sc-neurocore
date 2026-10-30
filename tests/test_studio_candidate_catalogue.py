# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical catalogue-to-candidate HTTP and execution custody

"""Copy actual model resources, then exercise ordinary candidate and workspace APIs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.neurons import universal_dsl
from sc_neurocore.neurons.model_identity import resolve_identity, schema_for_class
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.universal_dsl import load_schema
from sc_neurocore.studio.app import create_app
from tests.studio_candidate_jobs_support import finish_candidate_job


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Open the production application with exclusively owned job and project storage."""
    for key, folder in [
        ("JOB_ROOT", "jobs"),
        ("PROJECT_ROOT", "projects"),
        ("AUDIT_LOG_PATH", "audit.jsonl"),
    ]:
        monkeypatch.setenv("SC_NEUROCORE_STUDIO_" + key, str(tmp_path / folder))
    with TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ) as app:
        yield app


def test_every_bound_model_copies_its_exact_schema_and_order(client: TestClient) -> None:
    """Retain every actual bound schema, unit declaration and source without promotion."""
    source = universal_dsl._SCHEMA_DIR
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()
    }
    models_before = client.get("/api/models").json()
    bound = [name for name in _CLASS_TO_MODULE if resolve_identity(name).schema_profiles]
    assert bound
    for name in bound:
        response = client.get("/api/candidates/from-catalogue/" + name)
        assert response.status_code == 200, (name, response.text)
        result = response.json()
        schema = load_schema(schema_for_class(name))
        document = json.loads(result["candidate_text"])
        assert result["parent_model"] == name
        assert result["parent_schema"] == schema_for_class(name)
        assert (
            result["parent_schema_sha256"]
            == hashlib.sha256(
                json.dumps(
                    schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
        assert document["model"] == schema
        for section in ["state", "parameters", "dynamics", "reset"]:
            assert list(document["model"].get(section, {})) == list(schema.get(section, {}))
        assert document["parent"] == name and document["name"] not in _CLASS_TO_MODULE
        assert document["authors"] == [] and document["reference_tests"] == []
        assert list(document["units"]["state"]) == list(schema.get("state", {}))
        assert list(document["units"]["parameters"]) == list(schema.get("parameters", {}))
    assert client.get("/api/models").json() == models_before
    assert {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()
    } == before


def test_missing_metadata_is_explicit_in_the_draft(client: TestClient) -> None:
    """Keep authors and undeclared current units empty until the author supplies them."""
    document = json.loads(
        client.get("/api/candidates/from-catalogue/AdExNeuron").json()["candidate_text"]
    )
    assert document["authors"] == []
    assert document["units"]["current"] == ""
    assert document["units"]["state"] == {"v": "mV", "w": "pA"}
    assert document["units"]["time"] == "ms"
    assert document["source"]["doi"] == "10.1152/jn.00686.2005"
    response = client.post("/api/candidates/validate", json={"candidate": document})
    assert response.status_code == 200 and not response.json()["valid"]
    assert {d["location"] for d in response.json()["diagnostics"]} >= {"/authors", "/units/current"}


def test_completed_draft_uses_real_candidate_diff_jobs_and_artifacts(client: TestClient) -> None:
    """Validate and run the copied model through the durable public workflow."""
    text = client.get("/api/candidates/from-catalogue/AdExNeuron").json()["candidate_text"]
    document = json.loads(text)
    document["authors"] = ["Catalogue workflow test author"]
    document["units"]["current"] = "pA"
    document["reference_tests"] = [
        {
            "name": "no immediate spike at rest",
            "current": 0,
            "steps": 3,
            "expect": {"spike_count": {"max": 0}},
        }
    ]
    assert client.post("/api/candidates/validate", json={"candidate": document}).json()["valid"]
    diff = client.post("/api/candidates/diff", json={"candidate": document}).json()
    assert diff["status"] == "compared"
    for section in ["state", "parameters", "dynamics", "reset"]:
        assert all(row["status"] == "unchanged" for row in diff[section])
    assert diff["integration"]["status"] == "unchanged"
    for operation in ["simulate", "review-packet"]:
        body = {
            "candidate": document,
            **({"current": 0, "steps": 3} if operation == "simulate" else {}),
        }
        response = client.post("/api/candidates/" + operation, json=body)
        assert response.status_code == 202, response.text
        record = finish_candidate_job(client, response.json()["status_route"])
        assert record["status"] == "completed"
        for artifact in record["artifacts"]:
            data = client.get(
                response.json()["status_route"] + "/artifacts/" + artifact["relative_path"]
            )
            assert data.status_code == 200
            assert hashlib.sha256(data.content).hexdigest() == artifact["sha256"]
        if operation == "simulate":
            assert record["result"]["steps"] == 3
            assert record["result"]["spike_count"] == 0
        else:
            assert record["result"]["reference_tests_passed"]
            assert "catalogue promotion" in record["result"]["not_established"][0]


def test_unknown_and_unbound_catalogue_models_are_refused(client: TestClient) -> None:
    """Refuse unsupported identities without inventing an equation template."""
    response = client.get("/api/candidates/from-catalogue/NoSuchCatalogueModel")
    assert response.status_code == 404
    assert response.json() == {"detail": "Select a registered catalogue model."}
    unbound = [name for name in _CLASS_TO_MODULE if not resolve_identity(name).schema_profiles]
    assert unbound
    response = client.get("/api/candidates/from-catalogue/" + unbound[0])
    assert response.status_code == 404
    assert response.json() == {"detail": "This catalogue model has no canonical authoring schema."}


@pytest.mark.parametrize(
    "fault", ["missing", "json", "profile", "metadata", "science", "units", "nonfinite", "unicode"]
)
def test_source_faults_remain_server_faults(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Read damaged real schema copies without blaming or changing an author's draft."""
    schema = load_schema("adex")
    root = tmp_path / "schemas"
    root.mkdir()
    if fault in ["profile", "metadata", "science"]:
        schema[fault] = []
    elif fault == "units":
        schema["profile"]["units"] = []
    elif fault == "nonfinite":
        schema["state"]["v"] = float("nan")
    elif fault == "unicode":
        schema["metadata"]["author"] = chr(0xD800)
    if fault != "missing":
        (root / "adex.json").write_text("{" if fault == "json" else json.dumps(schema))
    monkeypatch.setattr(universal_dsl, "_SCHEMA_DIR", root)
    response = client.get("/api/candidates/from-catalogue/AdExNeuron")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error"}


def test_missing_source_fields_preserve_exact_numbers_and_declarations(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copy incomplete source metadata and exact integer tokens without guessing a repair."""
    schema: dict[str, Any] = load_schema("adex")
    schema["metadata"] = {"schema_version": 2, "url": "https://example.invalid/schema"}
    schema["profile"] = {"time_unit": "iteration"}
    schema.pop("science")
    schema["parameters"]["source_integer"] = 18446744073709551615
    schema["dynamics"] = dict(reversed(list(schema["dynamics"].items())))
    root = tmp_path / "schemas"
    root.mkdir()
    (root / "adex.json").write_text(json.dumps(schema))
    monkeypatch.setattr(universal_dsl, "_SCHEMA_DIR", root)
    result = client.get("/api/candidates/from-catalogue/AdExNeuron").json()
    document = json.loads(result["candidate_text"])
    assert '"source_integer": 18446744073709551615' in result["candidate_text"]
    assert document["model"] == schema
    assert list(document["model"]["dynamics"]) == ["w", "v"]
    assert document["source"] == {
        "citation": "SC-NeuroCore catalogue schema adex",
        "url": "https://example.invalid/schema",
    }
    assert document["assumptions"] == []
    assert document["units"]["time"] == "1"
    assert all(value == "" for value in document["units"]["parameters"].values())
