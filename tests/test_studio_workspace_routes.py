# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio workspace routes over HTTP

"""The workspace guarantees have to hold over HTTP, not only in Python.

A conflict that is only a Python exception protects nobody: the browser is the
editor. These cases drive the real application — two clients saving the same
workspace, a delete followed by a restore, an export carried into a second
installation — and assert on the status codes and bodies a client actually
receives, including that no filesystem path reaches it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

STATE = {
    "experiment": {"name": "SCLapicqueLIFNeuron", "dt": 0.1},
    "graph": {"populations": []},
    "hardware_profile": {"target": "ice40"},
}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A Studio client whose workspaces live under this test's own directory."""
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path / "projects"))
    with TestClient(create_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


def _save(
    client: TestClient, name: str, state: dict[str, object], expected: int | None
) -> dict[str, object]:
    response = client.post(
        "/api/project/save",
        json={"name": name, "state": state, "expected_revision": expected},
    )
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


class TestSaveConflict:
    def test_a_second_editor_is_refused_instead_of_overwriting(self, client: TestClient) -> None:
        first = _save(client, "shared", STATE, None)
        assert first["revision"] == 1

        # Both editors hold revision 1. Bob saves; Alice then saves from the
        # same revision she loaded.
        _save(client, "shared", {"note": "bob"}, 1)
        refused = client.post(
            "/api/project/save",
            json={"name": "shared", "state": {"note": "alice"}, "expected_revision": 1},
        )

        assert refused.status_code == 409
        detail = refused.json()["detail"]
        assert detail["error"] == "workspace_conflict"
        assert detail["expected_revision"] == 1
        assert detail["actual_revision"] == 2
        assert "reload" in detail["reason"]

        # Bob's work is what is current; Alice's save did not happen.
        current = client.get("/api/project/load/shared")
        assert current.json()["state"] == {"note": "bob"}

    def test_claiming_a_new_workspace_over_an_existing_one_is_refused(
        self, client: TestClient
    ) -> None:
        _save(client, "shared", STATE, None)

        refused = client.post("/api/project/save", json={"name": "shared", "state": {"note": "x"}})

        assert refused.status_code == 409
        assert refused.json()["detail"]["expected_revision"] is None

    def test_a_non_integer_expected_revision_is_a_client_error(self, client: TestClient) -> None:
        response = client.post(
            "/api/project/save",
            json={"name": "shared", "state": STATE, "expected_revision": "one"},
        )

        assert response.status_code == 422

    def test_a_conflict_body_carries_no_filesystem_path(self, client: TestClient) -> None:
        _save(client, "shared", STATE, None)
        refused = client.post("/api/project/save", json={"name": "shared", "state": STATE})

        assert "/" not in refused.json()["detail"]["reason"]


class TestHistory:
    def test_every_revision_stays_readable_after_later_saves(self, client: TestClient) -> None:
        _save(client, "w", {"note": "one"}, None)
        _save(client, "w", {"note": "two"}, 1)
        _save(client, "w", {"note": "three"}, 2)

        listing = client.get("/api/project/w/revisions")
        assert listing.status_code == 200
        revisions = listing.json()["revisions"]
        assert [entry["revision"] for entry in revisions] == [1, 2, 3]
        assert [entry["parent"] for entry in revisions] == [None, 1, 2]

        # The first revision reads back as it was written, not as it was
        # superseded.
        assert client.get("/api/project/load/w?revision=1").json()["state"] == {"note": "one"}
        assert client.get("/api/project/load/w").json()["state"] == {"note": "three"}

    def test_an_absent_revision_is_reported_as_absent(self, client: TestClient) -> None:
        _save(client, "w", STATE, None)

        response = client.get("/api/project/load/w?revision=9")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_a_fork_branches_without_touching_its_source(self, client: TestClient) -> None:
        _save(client, "w", {"note": "one"}, None)
        _save(client, "w", {"note": "two"}, 1)

        forked = client.post("/api/project/w/fork", json={"new_name": "branch", "revision": 1})
        assert forked.status_code == 200, forked.text
        assert forked.json()["revision"] == 1

        assert client.get("/api/project/load/branch").json()["state"] == {"note": "one"}
        assert client.get("/api/project/load/w").json()["state"] == {"note": "two"}

    def test_forking_onto_a_live_workspace_is_refused(self, client: TestClient) -> None:
        _save(client, "w", STATE, None)
        _save(client, "taken", STATE, None)

        refused = client.post("/api/project/w/fork", json={"new_name": "taken"})

        assert refused.status_code == 409


class TestRecoverableDelete:
    def test_a_deleted_workspace_can_be_restored(self, client: TestClient) -> None:
        _save(client, "w", {"note": "one"}, None)
        _save(client, "w", {"note": "two"}, 1)

        deleted = client.delete("/api/project/w")
        assert deleted.status_code == 200, deleted.text
        assert client.get("/api/project/load/w").status_code == 404

        waiting = client.get("/api/project/deleted")
        assert waiting.status_code == 200
        entries = waiting.json()["deleted"]
        assert [entry["name"] for entry in entries] == ["w"]
        token = entries[0]["token"]

        restored = client.post("/api/project/restore", json={"token": token})
        assert restored.status_code == 200, restored.text

        # The whole history came back, not only the state at deletion.
        assert client.get("/api/project/load/w").json()["state"] == {"note": "two"}
        assert len(client.get("/api/project/w/revisions").json()["revisions"]) == 2

    def test_restoring_onto_a_live_name_is_refused(self, client: TestClient) -> None:
        _save(client, "w", STATE, None)
        token = None
        client.delete("/api/project/w")
        token = client.get("/api/project/deleted").json()["deleted"][0]["token"]
        _save(client, "w", {"note": "new work"}, None)

        refused = client.post("/api/project/restore", json={"token": token})

        assert refused.status_code == 409
        # The new work is untouched by the refusal.
        assert client.get("/api/project/load/w").json()["state"] == {"note": "new work"}

    def test_an_unknown_token_is_refused(self, client: TestClient) -> None:
        response = client.post("/api/project/restore", json={"token": "nothing-here"})

        assert response.status_code == 404


class TestTransfer:
    def test_export_and_import_carry_a_workspace_between_installations(
        self, client: TestClient
    ) -> None:
        _save(client, "w", STATE, None)

        exported = client.get("/api/project/w/export")
        assert exported.status_code == 200, exported.text
        document = exported.json()

        imported = client.post(
            "/api/project/import", json={"name": "arrived", "document": document}
        )
        assert imported.status_code == 200, imported.text
        assert imported.json()["revision"] == 1
        assert client.get("/api/project/load/arrived").json()["state"] == STATE

    def test_importing_something_that_is_not_a_workspace_is_refused(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/project/import", json={"name": "arrived", "document": {"nope": True}}
        )

        assert response.status_code == 422
        assert client.get("/api/project/load/arrived").status_code == 404

    def test_an_exported_document_carries_no_filesystem_path(self, client: TestClient) -> None:
        _save(client, "w", STATE, None)

        document = client.get("/api/project/w/export").json()

        assert "path" not in document
        assert "/tmp" not in str(document)
