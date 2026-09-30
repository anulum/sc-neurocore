# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Identity refusals through authenticated Studio requests

"""Exercise identity persistence, refusals and Unicode login through real HTTP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.identity import update_studio_identity_record
from sc_neurocore.studio.platform.identity_passwords import (
    make_browser_user_password_verifier,
    verify_browser_user_password,
)
from sc_neurocore.studio.platform.identity_refusals import StudioIdentityRefused
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings

_USERS = "/api/studio/identity/browser-users"
_MUTATIONS: list[tuple[str, str, dict[str, object]]] = [
    ("POST", _USERS, {"username": "new", "principal_id": "new", "password": "password"}),
    ("PATCH", "/api/studio/identity/service-accounts/admin", {"active": True}),
    ("PATCH", f"{_USERS}/operator", {"active": True}),
    ("POST", f"{_USERS}/operator/password", {"password": "replacement"}),
]


def _client(tmp_path: Path) -> tuple[TestClient, Path]:
    """Configure actual private identities, durable auditing and enforced policies."""
    identity_path = tmp_path / "identities.json"
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "admin",
                        "roles": ["studio.admin"],
                        "token_sha256": hashlib.sha256(b"test-admin").hexdigest(),
                    }
                ],
                "browser_users": [
                    {
                        "username": "operator",
                        "principal_id": "human-operator",
                        "roles": ["studio.admin"],
                        "password_pbkdf2_sha256": make_browser_user_password_verifier("password"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    identity_path.chmod(0o600)
    app = create_app(
        StudioRuntimeSettings(
            allow_header_principal=False,
            enforce_route_policies=True,
            identity_file_path=str(identity_path),
            audit_log_path=str(tmp_path / "audit.jsonl"),
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    return TestClient(
        app, base_url="http://127.0.0.1", headers={"Authorization": "Bearer test-admin"}
    ), identity_path


@pytest.mark.parametrize(("method", "route", "payload"), _MUTATIONS)
def test_generated_store_error_has_fixed_response(
    tmp_path: Path, method: str, route: str, payload: dict[str, object]
) -> None:
    """Real codec errors never escape any identity mutation or rewrite its file."""
    client, path = _client(tmp_path)
    path.write_bytes(b"\xff")
    response = client.request(method, route, json={"roles": ["studio.viewer"], **payload})
    assert response.status_code == 422
    assert response.json() == {"detail": "Studio identity request could not be validated."}
    assert path.read_bytes() == b"\xff"


@pytest.mark.parametrize(("method", "route", "payload"), _MUTATIONS[:3])
@pytest.mark.parametrize("expiry", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_utc_expiry_overflow_is_authored_refusal(
    tmp_path: Path, method: str, route: str, payload: dict[str, object], expiry: str
) -> None:
    """Valid ISO instants outside UTC years 1–9999 are refused before persistence."""
    client, path = _client(tmp_path)
    before = path.read_bytes()
    response = client.request(
        method, route, json={"roles": ["studio.admin"], **payload, "expires_at_utc": expiry}
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": "Studio identity expires_at_utc must be within the supported UTC date range."
    }
    assert path.read_bytes() == before


def test_unicode_username_creation_login_and_password_rotation(tmp_path: Path) -> None:
    """Unicode users authenticate after store reload and rotation revokes old sessions."""
    client, _ = _client(tmp_path)
    created = client.post(
        _USERS,
        json={
            "username": "Žofia",
            "principal_id": "human-zofia",
            "roles": ["studio.viewer"],
            "password": "heslo-漢字",
        },
    )
    assert created.status_code == 200
    assert created.json()["username"] == "Žofia"
    for username, password in [("未知", "password"), ("Žofia", "wrong"), ("operator", "wrong")]:
        denied = client.post(
            "/api/studio/auth/login", json={"username": username, "password": password}
        )
        assert denied.status_code == 401
        assert denied.json() == {"detail": "invalid_browser_login"}
    login = client.post(
        "/api/studio/auth/login", json={"username": "Žofia", "password": "heslo-漢字"}
    )
    assert login.status_code == 200
    bearer = {"Authorization": f"Bearer {login.json()['access_token']}"}
    session = client.get("/api/studio/auth/session", headers=bearer)
    assert session.status_code == 200
    assert session.json()["principal_id"] == "human-zofia"
    rotated = client.post(f"{_USERS}/Žofia/password", json={"password": "nové-heslo"})
    assert rotated.status_code == 200
    assert client.get("/api/studio/auth/session", headers=bearer).status_code == 401
    assert (
        client.post(
            "/api/studio/auth/login", json={"username": "Žofia", "password": "heslo-漢字"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/studio/auth/login", json={"username": "Žofia", "password": "nové-heslo"}
        ).status_code
        == 200
    )
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "studio.identity.browser_user.password.rotate" in audit
    assert "heslo-漢字" not in audit
    assert "nové-heslo" not in audit


def test_authored_duplicate_and_validation_messages_preserved(tmp_path: Path) -> None:
    """Duplicate conflict classification and deliberate validation text remain stable."""
    client, path = _client(tmp_path)
    before = path.read_bytes()
    duplicate = client.post(
        _USERS,
        json={
            "username": "operator",
            "principal_id": "other",
            "password": "password",
            "roles": ["studio.viewer"],
        },
    )
    assert duplicate.status_code == 409
    assert duplicate.json() == {"detail": "Studio browser user username already exists."}
    for method, route, payload in _MUTATIONS[:3]:
        invalid = client.request(
            method,
            route,
            json={"roles": ["studio.viewer"], **payload, "expires_at_utc": "caller-text-xyz"},
        )
        assert invalid.status_code == 422
        assert invalid.json() == {
            "detail": "Studio identity expires_at_utc must be an ISO timestamp."
        }
    assert path.read_bytes() == before


def test_empty_password_is_a_domain_refusal() -> None:
    """Library callers retain ValueError compatibility without exposing generated text."""
    with pytest.raises(
        StudioIdentityRefused, match="Studio browser-user password must not be empty."
    ):
        make_browser_user_password_verifier("")


@pytest.mark.parametrize(
    ("method", "route", "payload", "detail"),
    [
        (
            "POST",
            _USERS,
            {"username": "new", "principal_id": " "},
            "Studio identity principal_id must be a non-empty string.",
        ),
        (
            "POST",
            _USERS,
            {"username": " ", "principal_id": "new"},
            "Studio browser user username must be a non-empty string.",
        ),
        (
            "POST",
            f"{_USERS}/%20/password",
            {},
            "Studio browser user username must be a non-empty string.",
        ),
    ],
)
def test_authored_identity_field_refusals(
    tmp_path: Path, method: str, route: str, payload: dict[str, object], detail: str
) -> None:
    """Real producer refusals retain authored text on create and password rotation."""
    client, path = _client(tmp_path)
    before = path.read_bytes()
    response = client.request(
        method, route, json={"roles": ["studio.viewer"], "password": "password", **payload}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    assert path.read_bytes() == before


def test_empty_roles_are_refused_by_public_library_mutation(tmp_path: Path) -> None:
    """Direct library consumers also receive the deliberate validation contract."""
    _, path = _client(tmp_path)
    before = path.read_bytes()
    with pytest.raises(
        StudioIdentityRefused, match="Studio identity roles must be a non-empty list."
    ):
        update_studio_identity_record(
            path, principal_id="admin", roles=[], active=True, expires_at_utc=None
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("digest", ["１" * 64, "g" * 64, "0" * 63])
def test_invalid_password_digest_fails_authentication(digest: str) -> None:
    """Malformed stored digests fail closed through the public verifier API."""
    verifier = f"pbkdf2_sha256$390000${'0' * 32}${digest}"
    assert not verify_browser_user_password("password", verifier)
