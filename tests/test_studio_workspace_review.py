# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Review comments stay bound to the revision they were written on

"""A comment names its revision and that revision's digest, and says if either is gone."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from sc_neurocore.studio.workspace_review import (
    MAX_COMMENT_CHARS,
    REVIEW_FILE,
    REVIEW_SCHEMA_VERSION,
    add_comment,
    list_comments,
)
from sc_neurocore.studio.workspace_store import WorkspaceStore, state_digest

STATE = {"sourceMode": "model", "selectedModelName": "AdExNeuron"}


@pytest.fixture
def store(tmp_path: Path) -> WorkspaceStore:
    workspaces = WorkspaceStore(root=tmp_path / "projects", clock=lambda: 1000.0)
    workspaces.save("study", STATE, expected_revision=None)
    workspaces.save("study", {**STATE, "selectedModelName": "LIFNeuron"}, expected_revision=1)
    return workspaces


def test_a_comment_is_bound_to_its_revision_and_digest(store: WorkspaceStore) -> None:
    first = add_comment(store, "study", 1, author="alice", body="  Why AdEx here?  ")
    reply = add_comment(
        store, "study", 1, author="bob", body="Adaptation.", reply_to=first.comment_id
    )
    add_comment(store, "study", 2, author="alice", body="LIF is simpler.")

    assert first.body == "Why AdEx here?"
    assert first.state_sha256 == state_digest(STATE)
    assert first.created_at == 1000.0
    listed = list_comments(store, "study", revision=1)
    assert listed["schema_version"] == REVIEW_SCHEMA_VERSION
    assert [(c["author"], c["reply_to"], c["revision_status"]) for c in listed["comments"]] == [
        ("alice", None, "matches"),
        ("bob", first.comment_id, "matches"),
    ]
    assert len(list_comments(store, "study")["comments"]) == 3
    assert reply.revision == 1


def test_a_revision_that_changed_or_vanished_is_marked(store: WorkspaceStore) -> None:
    add_comment(store, "study", 1, author="alice", body="On one.")
    add_comment(store, "study", 2, author="alice", body="On two.")
    revisions = store.workspace_dir("study") / "revisions"
    first = revisions / "1.json"
    document = json.loads(first.read_text(encoding="utf-8"))
    document["state"]["selectedModelName"] = "Tampered"
    first.write_text(json.dumps(document), encoding="utf-8")
    (revisions / "2.json").unlink()
    # A blank line (a hand edit) is not a comment.
    with (store.workspace_dir("study") / REVIEW_FILE).open("a", encoding="utf-8") as handle:
        handle.write("\n")

    statuses = [c["revision_status"] for c in list_comments(store, "study")["comments"]]
    assert statuses == ["changed", "missing"]


@pytest.mark.parametrize("body", ["", "   ", "x" * (MAX_COMMENT_CHARS + 1)])
def test_an_empty_or_oversized_comment_is_refused(store: WorkspaceStore, body: str) -> None:
    with pytest.raises(ValueError, match=f"1 to {MAX_COMMENT_CHARS} characters"):
        add_comment(store, "study", 1, author="alice", body=body)


def test_a_reply_must_answer_a_comment_on_the_same_revision(store: WorkspaceStore) -> None:
    other = add_comment(store, "study", 2, author="alice", body="On two.")
    with pytest.raises(ValueError, match="is not a comment on revision 1"):
        add_comment(store, "study", 1, author="bob", body="Reply.", reply_to=other.comment_id)
    with pytest.raises(ValueError, match="is not a comment on revision 1"):
        add_comment(store, "study", 1, author="bob", body="Reply.", reply_to="nothing")


def test_a_missing_revision_or_workspace_is_refused(store: WorkspaceStore) -> None:
    with pytest.raises(KeyError):
        add_comment(store, "study", 9, author="alice", body="Nowhere.")
    with pytest.raises(KeyError):
        list_comments(store, "absent")
    assert list_comments(store, "study")["comments"] == []
    assert not (store.workspace_dir("study") / REVIEW_FILE).exists()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A Studio whose workspaces live under this test's directory."""
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path / "projects"))
    with TestClient(create_app(), base_url="http://127.0.0.1") as test_client:
        yield test_client


def _saved(client: TestClient) -> None:
    response = client.post(
        "/api/project/save", json={"name": "study", "state": STATE, "expected_revision": None}
    )
    assert response.status_code == 200, response.text


def test_the_routes_comment_list_and_refuse(client: TestClient) -> None:
    _saved(client)
    posted = client.post("/api/project/study/revisions/1/comments", json={"body": "Looks right."})
    assert posted.status_code == 200
    assert posted.json()["author"] == "local"
    listed = client.get("/api/project/study/comments", params={"revision": 1})
    assert [c["body"] for c in listed.json()["comments"]] == ["Looks right."]

    assert (
        client.post("/api/project/study/revisions/7/comments", json={"body": "x"}).status_code
        == 404
    )
    assert (
        client.post("/api/project/study/revisions/1/comments", json={"body": " "}).status_code
        == 422
    )
    assert client.get("/api/project/absent/comments").status_code == 404


def test_the_author_is_the_authenticated_principal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(tmp_path / "projects"))
    settings = StudioRuntimeSettings(enforce_route_policies=True, allow_header_principal=True)
    headers = {"X-Studio-Principal": "reviewer-7", "X-Studio-Roles": "studio.admin"}
    with TestClient(create_app(settings), base_url="http://127.0.0.1", headers=headers) as client:
        _saved(client)
        posted = client.post("/api/project/study/revisions/1/comments", json={"body": "Signed."})
    assert posted.status_code == 200, posted.text
    assert posted.json()["author"] == "reviewer-7"
