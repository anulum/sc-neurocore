# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate comparison resource integrity

"""Keep a genuinely corrupt parent schema out of candidate refusal diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.neurons import universal_dsl
from sc_neurocore.studio.app import create_app
from tests.studio_candidate_support import adex_candidate


@pytest.mark.parametrize("route", ["diff", "review-packet"])
def test_corrupt_parent_resource_remains_a_generic_server_failure(
    route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read an actual malformed schema without treating corruption as no parent."""
    document = adex_candidate()
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    raw = b'{"metadata": "private-corrupt-parent-marker",'
    parent = schemas / "adex.json"
    parent.write_bytes(raw)
    monkeypatch.setattr(universal_dsl, "_SCHEMA_DIR", schemas)
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    with TestClient(
        create_app(), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as client:
        response = client.post(f"/api/candidates/{route}", json={"candidate": document})
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert "private-corrupt-parent-marker" not in response.text
    assert parent.read_bytes() == raw
