# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio identity route failure tests

"""Exercise identity-route failures through the public Studio API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api import identity
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    AuditSinkError,
    JsonlAuditSink,
    StudioIdentityLifecycleError,
    StudioRuntimeSettings,
    make_browser_user_password_verifier,
)


def _write_identity_store(path: Path) -> None:
    """Write one valid service account and browser administrator."""

    path.write_text(
        json.dumps(
            {
                "browser_users": [
                    {
                        "active": True,
                        "expires_at_utc": None,
                        "password_pbkdf2_sha256": make_browser_user_password_verifier(
                            "operator-password"
                        ),
                        "principal_id": "human-operator",
                        "roles": ["studio.admin", "studio.viewer"],
                        "username": "operator",
                    }
                ],
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "active": True,
                        "expires_at_utc": None,
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin", "studio.viewer"],
                        "token_sha256": hashlib.sha256(b"admin-token").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _configured_client(
    tmp_path: Path,
    *,
    browser_login_max_failures: int = 5,
) -> TestClient:
    """Return a policy-disabled client with a valid persistent identity store."""

    identity_path = tmp_path / "studio-identities.json"
    _write_identity_store(identity_path)
    application = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(tmp_path / "audit" / "studio.jsonl"),
            browser_login_cooldown_seconds=60.0,
            browser_login_failure_window_seconds=300.0,
            browser_login_max_failures=browser_login_max_failures,
            enforce_route_policies=False,
            identity_file_path=str(identity_path),
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    return TestClient(application, base_url="http://127.0.0.1")


def _unconfigured_client(tmp_path: Path) -> TestClient:
    """Return a client without a persistent identity store."""

    application = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(tmp_path / "audit" / "studio.jsonl"),
            enforce_route_policies=False,
            identity_file_path=None,
            job_root_path=str(tmp_path / "jobs"),
        )
    )
    return TestClient(application, base_url="http://127.0.0.1")


def _raise_audit_error(self: JsonlAuditSink, event: object) -> None:
    """Raise a redaction sentinel instead of appending an audit row."""

    del self, event
    raise AuditSinkError("private/audit/path")


def test_login_requires_an_identity_store(tmp_path: Path) -> None:
    """Browser login fails closed when persistent identities are unavailable."""

    response = _unconfigured_client(tmp_path).post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "operator-password"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "identity_store_unavailable"


@pytest.mark.parametrize(
    ("credentials", "expected_stage"),
    [
        ({"username": "operator", "password": "wrong-password"}, "deny"),
        ({"username": "operator", "password": "operator-password"}, "allow"),
    ],
)
def test_login_audit_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    credentials: dict[str, str],
    expected_stage: str,
) -> None:
    """Both denied and successful authentication require durable audit evidence."""

    client = _configured_client(tmp_path)
    monkeypatch.setattr(JsonlAuditSink, "record", _raise_audit_error)
    response = client.post("/api/studio/auth/login", json=credentials)

    assert response.status_code == 503, expected_stage
    assert response.json()["detail"] == "audit_append_failed"
    assert "private/audit/path" not in response.text


def test_throttled_login_audit_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-throttled denial is not returned without its audit record."""

    client = _configured_client(tmp_path, browser_login_max_failures=1)
    priming = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "wrong-password"},
    )
    assert priming.status_code == 429
    monkeypatch.setattr(JsonlAuditSink, "record", _raise_audit_error)

    response = client.post(
        "/api/studio/auth/login",
        json={"username": "operator", "password": "wrong-password"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "audit_append_failed"


def test_logout_audit_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session revocation requires durable audit evidence."""

    client = _configured_client(tmp_path)
    monkeypatch.setattr(JsonlAuditSink, "record", _raise_audit_error)
    response = client.post("/api/studio/auth/logout")

    assert response.status_code == 503
    assert response.json()["detail"] == "audit_append_failed"


@pytest.mark.parametrize(
    ("route", "function_name"),
    [
        ("/api/studio/identity/service-accounts", "list_studio_identity_public_records"),
        ("/api/studio/identity/browser-users", "list_studio_browser_user_public_records"),
        (
            "/api/studio/identity/service-accounts/svc-admin",
            "list_studio_identity_public_records",
        ),
        (
            "/api/studio/identity/browser-users/operator",
            "list_studio_browser_user_public_records",
        ),
    ],
)
def test_identity_reads_hide_store_parse_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    function_name: str,
) -> None:
    """Malformed-store details map to a stable health response."""

    def _raise_value_error(path: Path) -> object:
        del path
        raise ValueError("private identity parse detail")

    monkeypatch.setattr(identity, function_name, _raise_value_error)
    response = _configured_client(tmp_path).get(route)

    assert response.status_code == 503
    assert response.json()["detail"] == "identity_store_unhealthy"
    assert "private identity parse detail" not in response.text


@pytest.mark.parametrize(
    ("method", "route", "payload"),
    [
        ("GET", "/api/studio/identity/browser-users", None),
        (
            "POST",
            "/api/studio/identity/browser-users",
            {
                "password": "new-password",
                "principal_id": "human-new",
                "roles": ["studio.viewer"],
                "username": "new-user",
            },
        ),
        ("GET", "/api/studio/identity/browser-users/operator", None),
        ("GET", "/api/studio/identity/service-accounts/svc-admin", None),
        (
            "PATCH",
            "/api/studio/identity/service-accounts/svc-admin",
            {"active": True, "roles": ["studio.admin"]},
        ),
        (
            "PATCH",
            "/api/studio/identity/browser-users/operator",
            {"active": True, "roles": ["studio.admin"]},
        ),
        (
            "POST",
            "/api/studio/identity/browser-users/operator/password",
            {"password": "rotated-password"},
        ),
    ],
)
def test_identity_admin_routes_require_a_store(
    tmp_path: Path,
    method: str,
    route: str,
    payload: dict[str, object] | None,
) -> None:
    """Every identity administration route fails closed without a store."""

    response = _unconfigured_client(tmp_path).request(method, route, json=payload)

    assert response.status_code == 409
    assert response.json()["detail"] == "identity_store_unavailable"


@pytest.mark.parametrize(
    ("route", "expected_detail"),
    [
        (
            "/api/studio/identity/browser-users/missing-user",
            "identity_browser_user_not_found",
        ),
        (
            "/api/studio/identity/service-accounts/missing-service",
            "identity_service_account_not_found",
        ),
    ],
)
def test_identity_record_lookup_returns_stable_not_found(
    tmp_path: Path,
    route: str,
    expected_detail: str,
) -> None:
    """Unknown public record lookups return type-specific errors."""

    response = _configured_client(tmp_path).get(route)

    assert response.status_code == 404
    assert response.json()["detail"] == expected_detail


def test_service_account_lookup_returns_public_record(tmp_path: Path) -> None:
    """A service-account lookup returns its token-free public projection."""

    response = _configured_client(tmp_path).get("/api/studio/identity/service-accounts/svc-admin")

    assert response.status_code == 200
    assert response.json() == {
        "active": True,
        "expires_at_utc": None,
        "principal_id": "svc-admin",
        "roles": ["studio.admin", "studio.viewer"],
    }
    assert "token" not in response.text


@pytest.mark.parametrize(
    ("function_name", "method", "route", "payload", "error_type", "expected_status"),
    [
        (
            "update_studio_identity_record",
            "PATCH",
            "/api/studio/identity/service-accounts/svc-admin",
            {"active": True, "roles": ["studio.admin"]},
            KeyError,
            404,
        ),
        (
            "update_studio_identity_record",
            "PATCH",
            "/api/studio/identity/service-accounts/svc-admin",
            {"active": True, "roles": ["studio.admin"]},
            ValueError,
            422,
        ),
        (
            "update_studio_browser_user_record",
            "PATCH",
            "/api/studio/identity/browser-users/operator",
            {"active": True, "roles": ["studio.admin"]},
            KeyError,
            404,
        ),
        (
            "update_studio_browser_user_record",
            "PATCH",
            "/api/studio/identity/browser-users/operator",
            {"active": True, "roles": ["studio.admin"]},
            StudioIdentityLifecycleError,
            409,
        ),
        (
            "update_studio_browser_user_record",
            "PATCH",
            "/api/studio/identity/browser-users/operator",
            {"active": True, "roles": ["studio.admin"]},
            ValueError,
            422,
        ),
        (
            "rotate_studio_browser_user_password",
            "POST",
            "/api/studio/identity/browser-users/operator/password",
            {"password": "rotated-password"},
            KeyError,
            404,
        ),
        (
            "rotate_studio_browser_user_password",
            "POST",
            "/api/studio/identity/browser-users/operator/password",
            {"password": "rotated-password"},
            ValueError,
            422,
        ),
    ],
)
def test_identity_mutation_maps_lifecycle_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    method: str,
    route: str,
    payload: dict[str, object],
    error_type: type[Exception],
    expected_status: int,
) -> None:
    """Identity mutation exceptions retain the bounded public vocabulary."""

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise error_type("bounded lifecycle detail")

    monkeypatch.setattr(identity, function_name, _raise)
    response = _configured_client(tmp_path).request(method, route, json=payload)

    assert response.status_code == expected_status
    if error_type is KeyError:
        assert response.json()["detail"].endswith("not_found")
    else:
        assert response.json()["detail"] == "bounded lifecycle detail"


@pytest.mark.parametrize(
    ("method", "route", "payload"),
    [
        (
            "POST",
            "/api/studio/identity/browser-users",
            {
                "password": "new-password",
                "principal_id": "human-new",
                "roles": ["studio.viewer"],
                "username": "new-user",
            },
        ),
        (
            "PATCH",
            "/api/studio/identity/service-accounts/svc-admin",
            {"active": True, "roles": ["studio.admin", "studio.viewer"]},
        ),
        (
            "PATCH",
            "/api/studio/identity/browser-users/operator",
            {"active": True, "roles": ["studio.admin", "studio.viewer"]},
        ),
        (
            "POST",
            "/api/studio/identity/browser-users/operator/password",
            {"password": "rotated-password"},
        ),
    ],
)
def test_identity_mutation_audit_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    route: str,
    payload: dict[str, object],
) -> None:
    """A successful identity mutation is not acknowledged without audit evidence."""

    client = _configured_client(tmp_path)
    monkeypatch.setattr(JsonlAuditSink, "record", _raise_audit_error)
    response = client.request(method, route, json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "audit_append_failed"
    assert "private/audit/path" not in response.text
