# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.workspace_store import WorkspaceConflict, WorkspaceStore


@pytest.fixture
def store(tmp_path: Path) -> WorkspaceStore:
    """A store holding one workspace two editors have both loaded."""
    return WorkspaceStore(root=tmp_path)


def test_a_refused_edit_survives_as_its_own_branch(store: WorkspaceStore) -> None:
    """Both edits exist afterwards; neither editor is asked to retype anything.

    The refusal itself is right — a stale save must not overwrite the other
    editor's work. But refusing alone leaves the losing edit in one browser and
    nowhere else, and the conflict message asks for it to be reapplied. This
    keeps it.
    """
    base = store.save("column", {"note": "shared base"})
    store.save("column", {"note": "editor A"}, expected_revision=base.revision)

    with pytest.raises(WorkspaceConflict):
        store.save("column", {"note": "editor B"}, expected_revision=base.revision)

    branch = store.branch_conflicting_edit(
        "column", {"note": "editor B"}, base_revision=base.revision
    )

    assert store.load("column")["state"] == {"note": "editor A"}
    assert store.load(branch.name)["state"] == {"note": "editor B"}


def test_the_branch_names_what_it_diverged_from(store: WorkspaceStore) -> None:
    """A reader reconciling the two needs the source and the revision."""
    base = store.save("column", {"note": "shared base"})

    branch = store.branch_conflicting_edit(
        "column", {"note": "editor B"}, base_revision=base.revision
    )

    assert branch.name == "column (from revision 1)"
    assert branch.revision == 1


def test_branching_from_a_revision_that_does_not_exist_is_refused(
    store: WorkspaceStore,
) -> None:
    """An edit that diverged from nothing is not a divergence."""
    store.save("column", {"note": "shared base"})

    with pytest.raises(KeyError):
        store.branch_conflicting_edit("column", {"note": "b"}, base_revision=99)


def test_branching_never_overwrites_an_existing_workspace(
    store: WorkspaceStore,
) -> None:
    """Two conflicts in a row must not let the second bury the first."""
    base = store.save("column", {"note": "shared base"})
    store.branch_conflicting_edit("column", {"note": "editor B"}, base_revision=base.revision)

    with pytest.raises(WorkspaceConflict):
        store.branch_conflicting_edit("column", {"note": "editor C"}, base_revision=base.revision)

    assert store.load("column (from revision 1)")["state"] == {"note": "editor B"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A Studio client whose workspaces live under this test's own directory."""
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path / "projects"))
    with TestClient(create_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


def test_the_route_keeps_the_edit_the_save_route_refused(client: TestClient) -> None:
    """End to end over HTTP: refused by save, kept by the branch route."""
    base = client.post("/api/project/save", json={"name": "column", "state": {"note": "base"}})
    assert base.status_code == 200
    revision = base.json()["revision"]

    assert (
        client.post(
            "/api/project/save",
            json={"name": "column", "state": {"note": "A"}, "expected_revision": revision},
        ).status_code
        == 200
    )

    refused = client.post(
        "/api/project/save",
        json={"name": "column", "state": {"note": "B"}, "expected_revision": revision},
    )
    assert refused.status_code == 409

    kept = client.post(
        "/api/project/column/branch-refused-edit",
        json={"state": {"note": "B"}, "base_revision": revision},
    )
    assert kept.status_code == 200
    branch = kept.json()["branched"]

    assert client.get("/api/project/load/column").json()["state"] == {"note": "A"}
    assert client.get(f"/api/project/load/{branch}").json()["state"] == {"note": "B"}


def test_the_route_refuses_a_base_revision_that_does_not_exist(client: TestClient) -> None:
    """An edit that diverged from nothing is refused, not stored."""
    client.post("/api/project/save", json={"name": "column", "state": {"note": "base"}})

    response = client.post(
        "/api/project/column/branch-refused-edit",
        json={"state": {"note": "B"}, "base_revision": 99},
    )

    assert response.status_code == 404


def test_the_route_requires_a_state_and_a_base_revision(client: TestClient) -> None:
    """Both are needed to describe a divergence; neither is guessed."""
    client.post("/api/project/save", json={"name": "column", "state": {"note": "base"}})

    assert (
        client.post(
            "/api/project/column/branch-refused-edit", json={"base_revision": 1}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/project/column/branch-refused-edit", json={"state": {"note": "B"}}
        ).status_code
        == 422
    )
