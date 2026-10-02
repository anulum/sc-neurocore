# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Revision-comment refusals through public persistence and HTTP

"""Review failures retain their meaning without exposing stored-data diagnostics."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread

import pytest
from starlette.testclient import TestClient

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.project import comment_on_revision, review_comments
from sc_neurocore.studio.workspace_review import MAX_COMMENT_CHARS, REVIEW_FILE
from sc_neurocore.studio.workspace_store import WorkspaceStore, state_digest

STATE = {"sourceMode": "model", "selectedModelName": "AdExNeuron"}
POST = "/api/project/study/revisions/1/comments"
GET = "/api/project/study/comments"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Path]]:
    """Save two real revisions through the app in an isolated project directory."""
    root = tmp_path / "projects"
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(root))
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        for revision, state in enumerate([STATE, {**STATE, "selectedModelName": "LIFNeuron"}]):
            response = client.post(
                "/api/project/save",
                json={
                    "name": "study",
                    "state": state,
                    "expected_revision": revision or None,
                },
            )
            assert response.status_code == 200, response.text
        yield client, root / "study"


@pytest.mark.parametrize(
    "body",
    ["", " \t\n ", "é" * (MAX_COMMENT_CHARS + 1)],
    ids=["empty", "whitespace", "oversized-unicode"],
)
def test_deliberate_body_refusal_is_marked_and_explained(
    workspace: tuple[TestClient, Path], body: str
) -> None:
    """An invalid comment has a ValueError-compatible authored refusal and no write."""
    client, directory = workspace
    with pytest.raises(AuthoredRefusal, match="1 to 4000 characters"):
        comment_on_revision("study", 1, author="alice", body=body)
    response = client.post(POST, json={"body": body})
    assert response.status_code == 422
    assert response.json() == {"detail": "a comment holds 1 to 4000 characters"}
    assert not (directory / REVIEW_FILE).exists()


@pytest.mark.parametrize("cross_revision", [False, True])
def test_reply_refusal_preserves_existing_comments(
    workspace: tuple[TestClient, Path], cross_revision: bool
) -> None:
    """Unknown and other-revision replies keep the explanation and append nothing."""
    client, directory = workspace
    first = client.post("/api/project/study/revisions/2/comments", json={"body": "On LIF."}).json()
    reply_to = first["comment_id"] if cross_revision else "unknown-comment"
    before = (directory / REVIEW_FILE).read_bytes()
    with pytest.raises(AuthoredRefusal, match="is not a comment on revision 1"):
        comment_on_revision("study", 1, author="alice", body="Reply", reply_to=reply_to)
    response = client.post(POST, json={"body": "Reply", "reply_to": reply_to})
    assert response.status_code == 422
    assert response.json() == {"detail": f"{reply_to} is not a comment on revision 1"}
    assert (directory / REVIEW_FILE).read_bytes() == before


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("bad_name", ["%20", "bad%5Cname"])
def test_invalid_name_has_the_deliberate_project_explanation(
    workspace: tuple[TestClient, Path], method: str, bad_name: str
) -> None:
    """Both comment endpoints translate the actual project-name refusal."""
    client, directory = workspace
    url = f"/api/project/{bad_name}/comments"
    if method == "POST":
        url = f"/api/project/{bad_name}/revisions/1/comments"
    response = client.request(method, url, json={"body": "Comment"} if method == "POST" else None)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid project name"}
    assert not (directory / REVIEW_FILE).exists()


@pytest.mark.parametrize(
    ("method", "url", "message"),
    [
        ("GET", "/api/project/absent/comments", "Project 'absent' not found"),
        ("POST", "/api/project/absent/revisions/1/comments", "Project 'absent' has no revision 1"),
        ("POST", "/api/project/study/revisions/99/comments", "Project 'study' has no revision 99"),
    ],
)
def test_actual_missing_target_keeps_404(
    workspace: tuple[TestClient, Path], method: str, url: str, message: str
) -> None:
    """A genuinely missing target stays distinct from corrupted review storage."""
    client, directory = workspace
    response = client.request(method, url, json={"body": "Comment"} if method == "POST" else None)
    assert response.status_code == 404
    assert response.json() == {"detail": message}
    assert not (directory / REVIEW_FILE).exists()


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    "payload",
    [b"{PRIVATE-REVIEW-FAULT\n", b"{}\n", b"[]\n", b"null\n", b"\xffPRIVATE-REVIEW-FAULT\n"],
    ids=["malformed-json", "missing-fields", "array", "null", "invalid-utf8"],
)
def test_unreadable_comment_storage_is_a_generic_internal_error(
    workspace: tuple[TestClient, Path], method: str, payload: bytes
) -> None:
    """Real corrupt bytes produce safe 500s and are retained for recovery."""
    client, directory = workspace
    path = directory / REVIEW_FILE
    path.write_bytes(payload)
    response = client.request(
        method,
        GET if method == "GET" else POST,
        json={"body": "Reply", "reply_to": "unknown-comment"} if method == "POST" else None,
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error"}
    assert path.read_bytes() == payload
    with pytest.raises(RuntimeError, match="review") as raised:
        review_comments("study")
    assert not isinstance(raised.value, AuthoredRefusal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comment_id", None),
        ("comment_id", ""),
        ("revision", "1"),
        ("revision", True),
        ("revision", 0),
        ("state_sha256", "PRIVATE-REVIEW-FAULT"),
        ("state_sha256", "g" * 64),
        ("author", []),
        ("created_at", float("nan")),
        ("created_at", True),
        ("created_at", "PRIVATE-REVIEW-FAULT"),
        ("body", {}),
        ("body", " "),
        ("body", "x" * (MAX_COMMENT_CHARS + 1)),
        ("body", "\ud800"),
        ("reply_to", 4),
    ],
)
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_malformed_stored_record_never_becomes_a_public_comment(
    workspace: tuple[TestClient, Path], field: str, value: object, method: str
) -> None:
    """Every stored wire field must satisfy the producer's actual record shape."""
    client, directory = workspace
    posted = client.post(POST, json={"body": "A valid review."})
    assert posted.status_code == 200
    record: dict[str, object] = posted.json()
    record[field] = value
    payload = (json.dumps(record) + "\n").encode()
    path = directory / REVIEW_FILE
    path.write_bytes(payload)
    response = client.request(
        method,
        GET if method == "GET" else POST,
        json={"body": "Reply", "reply_to": "unknown-comment"} if method == "POST" else None,
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error"}
    assert path.read_bytes() == payload


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_directory_at_review_file_is_an_error(
    workspace: tuple[TestClient, Path], method: str
) -> None:
    """A directory at review.jsonl cannot mean that the workspace has no comments."""
    client, directory = workspace
    path = directory / REVIEW_FILE
    path.mkdir()
    response = client.request(
        method,
        GET if method == "GET" else POST,
        json={"body": "Comment"} if method == "POST" else None,
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error"}
    assert path.is_dir() and not list(path.iterdir())


@pytest.mark.parametrize("schema_number", ["PRIVATE-REVISION-FAULT", 999])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_corrupt_saved_revision_is_not_a_comment_refusal(
    workspace: tuple[TestClient, Path], schema_number: str | int, method: str
) -> None:
    """Stored schema faults never escape through the user's comment explanation."""
    client, directory = workspace
    if method == "GET":
        assert client.post(POST, json={"body": "An existing review."}).status_code == 200
    review_path = directory / REVIEW_FILE
    before = review_path.read_bytes() if review_path.exists() else None
    path = directory / "revisions" / "1.json"
    document = json.loads(path.read_text())
    document["schema_number"] = schema_number
    payload = json.dumps(document).encode()
    path.write_bytes(payload)
    response = client.request(
        method,
        GET if method == "GET" else POST,
        json={"body": "A valid review."} if method == "POST" else None,
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error"}
    assert path.read_bytes() == payload
    assert (review_path.read_bytes() if review_path.exists() else None) == before


@pytest.mark.parametrize("name", [None, 3, "", ".", "..", "a/b", "a\\b", "/root"])
def test_public_facade_rejects_names_from_dynamic_input(
    workspace: tuple[TestClient, Path], name: object
) -> None:
    """Public facade calls from decoded JSON keep the deliberate name marker."""
    arguments = json.loads(json.dumps({"name": name}))
    with pytest.raises(AuthoredRefusal, match="Invalid project name"):
        review_comments(**arguments)


@pytest.mark.parametrize("field", ["body", "reply_to"])
def test_unpaired_unicode_is_a_safe_request_refusal(
    workspace: tuple[TestClient, Path], field: str
) -> None:
    """Invalid text cannot break the JSON error response or append a review."""
    client, directory = workspace
    payload = {"body": "Valid", field: "\ud800"}
    response = client.post(
        POST, content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    expected = (
        "comment text and author must be valid UTF-8"
        if field == "body"
        else "\\ud800 is not a comment on revision 1"
    )
    assert response.json() == {"detail": expected}
    assert not (directory / REVIEW_FILE).exists()


def test_a_new_root_comment_cannot_append_to_corrupt_history(
    workspace: tuple[TestClient, Path],
) -> None:
    """Reading existing records protects roots as well as replies from bad history."""
    client, directory = workspace
    path = directory / REVIEW_FILE
    before = b"PRIVATE-REVIEW-FAULT\n"
    path.write_bytes(before)
    response = client.post(POST, json={"body": "A new root."})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error"}
    assert path.read_bytes() == before


def test_invalid_author_text_is_refused_before_creating_storage(
    workspace: tuple[TestClient, Path],
) -> None:
    """A dynamic library author cannot produce an undecodable review file."""
    _, directory = workspace
    with pytest.raises(AuthoredRefusal, match="valid UTF-8"):
        comment_on_revision("study", 1, author="\ud800", body="A valid review.")
    assert not (directory / REVIEW_FILE).exists()


def test_required_nullable_reply_field_and_forward_compatible_extra_field(
    workspace: tuple[TestClient, Path],
) -> None:
    """The nullable reply member remains required while unrelated extra data is ignored."""
    client, directory = workspace
    posted = client.post(POST, json={"body": "A valid review."})
    assert posted.status_code == 200
    record: dict[str, object] = posted.json()
    record["future_metadata"] = {"note": "preserve existing reader compatibility"}
    path = directory / REVIEW_FILE
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    listed = client.get(GET)
    assert listed.status_code == 200
    assert "future_metadata" not in listed.json()["comments"][0]
    del record["reply_to"]
    payload = (json.dumps(record) + "\n").encode()
    path.write_bytes(payload)
    response = client.get(GET)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error"}
    assert path.read_bytes() == payload


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_actual_workspace_contention_keeps_the_retryable_503(
    workspace: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """A second real SQLite writer preserves the shared boundary's retry semantics."""
    client, directory = workspace
    monkeypatch.setattr("sc_neurocore.studio.project._LOCK_TIMEOUT", 0.05)
    held, release = Event(), Event()
    store = WorkspaceStore(root=directory.parent)

    def hold() -> None:
        """Hold the public workspace lock until the bounded HTTP request finishes."""
        with store.lock("study"):
            held.set()
            assert release.wait(10)

    writer = Thread(target=hold)
    writer.start()
    try:
        assert held.wait(5)
        response = client.request(
            method,
            GET if method == "GET" else POST,
            json={"body": "Comment"} if method == "POST" else None,
        )
        assert response.status_code == 503
        assert response.json()["detail"]["error"] == "workspace_busy"
        assert not (directory / REVIEW_FILE).exists()
    finally:
        release.set()
        writer.join(5)
    assert not writer.is_alive()


def test_valid_unicode_boundary_and_replies_remain_append_only(
    workspace: tuple[TestClient, Path],
) -> None:
    """The largest valid comment and its reply retain state, ordering and bytes."""
    client, directory = workspace
    first = client.post(POST, json={"body": "  " + "é" * MAX_COMMENT_CHARS + "  "})
    assert first.status_code == 200, first.text
    record = first.json()
    assert record["body"] == "é" * MAX_COMMENT_CHARS
    assert record["state_sha256"] == state_digest(STATE)
    assert record["revision"] == 1 and record["author"] == "local"
    before = (directory / REVIEW_FILE).read_bytes()
    reply = client.post(
        POST, json={"body": "Adaptation matters.", "reply_to": record["comment_id"]}
    )
    assert reply.status_code == 200, reply.text
    assert (directory / REVIEW_FILE).read_bytes().startswith(before)
    other = client.post("/api/project/study/revisions/2/comments", json={"body": "On LIF."})
    assert other.status_code == 200
    listed = client.get(GET, params={"revision": 1})
    assert listed.status_code == 200
    entries = listed.json()["comments"]
    assert [c["comment_id"] for c in entries] == [record["comment_id"], reply.json()["comment_id"]]
    assert [c["revision_status"] for c in entries] == ["matches", "matches"]
    assert entries[1]["reply_to"] == record["comment_id"]
    assert len(client.get(GET).json()["comments"]) == 3
