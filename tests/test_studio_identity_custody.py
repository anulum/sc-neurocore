# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real identity and login custody regressions

"""Exercise the lab identity contract through real stores, HTTP and CLI processes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.bootstrap import bootstrap_studio_admin_identity
from sc_neurocore.studio.platform.identity import (
    add_studio_browser_user_record,
    load_studio_identity_store,
    rotate_studio_browser_user_password,
)
from sc_neurocore.studio.platform.identity_io import identity_store_mutation
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings


def _lab(
    path: Path, *, browser_login_max_failures: int = 5, browser_login_max_buckets: int = 1024
) -> tuple[TestClient, str]:
    """Provision a real private lab with a distinct service and browser principal."""
    identity = path / "identities.json"
    bootstrap = bootstrap_studio_admin_identity(identity)
    add_studio_browser_user_record(
        identity,
        username="operator",
        principal_id="human-operator",
        roles=["studio.admin"],
        password="operator-password",
    )
    app = create_app(
        StudioRuntimeSettings(
            deployment_profile="production",
            enforce_route_policies=True,
            allow_header_principal=False,
            identity_file_path=str(identity),
            audit_log_path=str(path / "audit.jsonl"),
            job_root_path=str(path / "jobs"),
            browser_login_max_failures=browser_login_max_failures,
            browser_login_max_buckets=browser_login_max_buckets,
        )
    )
    return TestClient(
        app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ), bootstrap.bearer_token


def test_all_failed_logins_have_the_same_public_response(tmp_path: Path) -> None:
    """Unknown, disabled, expired and wrong-password users share a single refusal."""
    client, _ = _lab(tmp_path)
    identity = tmp_path / "identities.json"
    for name, active, expiry in [
        ("disabled", False, None),
        ("expired", True, "2020-01-01T00:00:00Z"),
    ]:
        add_studio_browser_user_record(
            identity,
            username=name,
            principal_id=f"human-{name}",
            roles=["studio.viewer"],
            password="operator-password",
            active=active,
            expires_at_utc=expiry,
        )
    # CLI/library changes take effect at the documented process restart.
    client, _token = _existing_lab(tmp_path)
    with client:
        for name, password in [
            ("unknown", "operator-password"),
            ("disabled", "operator-password"),
            ("expired", "operator-password"),
            ("operator", "incorrect-password"),
        ]:
            response = client.post(
                "/api/studio/auth/login", json={"username": name, "password": password}
            )
            assert response.status_code == 401
            assert response.json() == {"detail": "invalid_browser_login"}


def test_legacy_password_still_logs_in_under_current_verification_work(tmp_path: Path) -> None:
    """A real stored legacy digest remains usable for login and subsequent rotation."""
    _lab(tmp_path)
    path = tmp_path / "identities.json"
    stored = json.loads(path.read_text())
    salt = bytes(range(16))
    digest = hashlib.pbkdf2_hmac("sha256", b"legacy", salt, 100_000).hex()
    stored["browser_users"][0]["password_pbkdf2_sha256"] = (
        f"pbkdf2_sha256$100000${salt.hex()}${digest}"
    )
    path.write_text(json.dumps(stored))
    client, _ = _existing_lab(tmp_path)
    with client:
        response = client.post(
            "/api/studio/auth/login", json={"username": "operator", "password": "legacy"}
        )
        assert response.status_code == 200
        assert response.json()["principal_id"] == "human-operator"


def _existing_lab(path: Path) -> tuple[TestClient, FastAPI]:
    """Rebuild the actual API from the same durable identities, as after a restart."""
    app = create_app(
        StudioRuntimeSettings(
            deployment_profile="production",
            enforce_route_policies=True,
            allow_header_principal=False,
            identity_file_path=str(path / "identities.json"),
            audit_log_path=str(path / "audit.jsonl"),
            job_root_path=str(path / "jobs"),
        )
    )
    return TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}), app


def test_concurrent_http_logins_share_the_failure_budget(tmp_path: Path) -> None:
    """Concurrent real HTTP requests cannot all enter password verification."""
    client, _ = _lab(tmp_path, browser_login_max_failures=3)
    with client:

        def login(_: int) -> int:
            return int(
                client.post(
                    "/api/studio/auth/login",
                    json={"username": "operator", "password": "incorrect-password"},
                ).status_code
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            statuses = list(executor.map(login, range(12)))
        assert statuses.count(401) == 2
        assert statuses.count(429) == 10
        assert (
            cast(FastAPI, client.app)
            .state.studio_browser_login_throttle.snapshot()
            .locked_bucket_count
            == 1
        )


def test_unknown_login_registry_has_a_finite_capacity(tmp_path: Path) -> None:
    """Distinct ordinary failed logins stay within the configured registry cap."""
    client, _ = _lab(tmp_path, browser_login_max_buckets=4)
    with client:
        for index in range(9):
            response = client.post(
                "/api/studio/auth/login",
                json={"username": f"missing-{index}", "password": "incorrect-password"},
            )
            assert response.status_code == 401
            assert (
                cast(FastAPI, client.app)
                .state.studio_browser_login_throttle.snapshot()
                .active_bucket_count
                <= 4
            )


def test_cli_and_http_mutations_both_persist(tmp_path: Path) -> None:
    """A real CLI addition and an API update preserve both changes in one store."""
    client, token = _lab(tmp_path)
    command = [
        sys.executable,
        "-m",
        "sc_neurocore.cli",
        "studio-add-browser-user",
        "--identity-file",
        str(tmp_path / "identities.json"),
        "--username",
        "analyst",
        "--principal-id",
        "human-analyst",
        "--role",
        "studio.viewer",
        "--password-stdin",
    ]
    with client, ThreadPoolExecutor(max_workers=2) as executor:
        added = executor.submit(
            subprocess.run,
            command,
            input="analyst-password\n",
            text=True,
            capture_output=True,
            timeout=30,
        )
        changed = client.patch(
            "/api/studio/identity/browser-users/operator",
            headers={"authorization": f"Bearer {token}"},
            json={"roles": ["studio.viewer"], "active": True},
        )
        completed = added.result(timeout=35)
        assert completed.returncode == 0
        assert "Restart a running Studio API" in completed.stderr
        assert changed.status_code == 200
    store = load_studio_identity_store(tmp_path / "identities.json")
    assert {user.username: user.roles for user in store.browser_users} == {
        "operator": frozenset({"studio.viewer"}),
        "analyst": frozenset({"studio.viewer"}),
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX owner and directory modes")
def test_group_writable_identity_parent_is_refused(tmp_path: Path) -> None:
    """The public loader refuses a store in a directory writable by its group."""
    _lab(tmp_path)
    tmp_path.chmod(0o770)
    try:
        with pytest.raises(ValueError, match="not writable by group or other"):
            load_studio_identity_store(tmp_path / "identities.json")
    finally:
        tmp_path.chmod(0o700)


def test_identity_symlink_is_refused_before_read(tmp_path: Path) -> None:
    """The public loader applies the documented no-symbolic-link store rule."""
    _lab(tmp_path)
    alias = tmp_path / "alias.json"
    alias.symlink_to(tmp_path / "identities.json")
    with pytest.raises(ValueError, match="must not be a symbolic link"):
        load_studio_identity_store(alias)


def test_identity_validation_never_echoes_credentials(tmp_path: Path) -> None:
    """Schema errors omit nested credentials and enforce new-password length."""
    client, token = _lab(tmp_path)
    with client:
        for payload in [{"password": {"token": "private-marker-19"}}, {"password": "tiny"}]:
            response = client.post(
                "/api/studio/identity/browser-users/operator/password",
                headers={"authorization": f"Bearer {token}"},
                json=payload,
            )
            assert response.status_code == 422
            assert "private-marker-19" not in response.text
            assert "tiny" not in response.text
            assert all(
                "input" not in error and "ctx" not in error for error in response.json()["detail"]
            )
        assert (
            client.get("/api/health")
            .headers["content-security-policy"]
            .startswith("default-src 'self'")
        )


def test_browser_principal_cannot_reuse_service_identity(tmp_path: Path) -> None:
    """A browser user cannot be provisioned under the service-account principal."""
    client, token = _lab(tmp_path)
    before = (tmp_path / "identities.json").read_bytes()
    with client:
        response = client.post(
            "/api/studio/identity/browser-users",
            headers={"authorization": f"Bearer {token}"},
            json={
                "username": "second",
                "principal_id": "svc-studio-admin",
                "roles": ["studio.viewer"],
                "password": "operator-password",
            },
        )
    assert response.status_code == 409
    assert (tmp_path / "identities.json").read_bytes() == before


def test_password_rotation_library_refuses_a_short_replacement(tmp_path: Path) -> None:
    """Apply the same password minimum to direct library rotation and HTTP administration."""
    _lab(tmp_path)
    path = tmp_path / "identities.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="at least 12 characters"):
        rotate_studio_browser_user_password(path, username="operator", password="short")
    assert path.read_bytes() == before


def test_identity_parent_must_be_a_directory(tmp_path: Path) -> None:
    """Refuse a regular file used as an identity parent through the public loader."""
    parent = tmp_path / "ordinary-file"
    parent.write_text("not a directory")
    with pytest.raises(ValueError, match="parent path must be a directory"):
        load_studio_identity_store(parent / "identities.json")


@pytest.mark.parametrize(
    "method,route,payload",
    [
        (
            "POST",
            "/api/studio/identity/browser-users",
            {
                "username": "analyst",
                "principal_id": "human-analyst",
                "roles": ["studio.viewer"],
                "password": "analyst-password",
            },
        ),
        (
            "PATCH",
            "/api/studio/identity/service-accounts/svc-studio-admin",
            {"roles": ["studio.viewer"], "active": True},
        ),
        (
            "PATCH",
            "/api/studio/identity/browser-users/operator",
            {"roles": ["studio.viewer"], "active": True},
        ),
        (
            "POST",
            "/api/studio/identity/browser-users/operator/password",
            {"password": "rotated-operator-password"},
        ),
    ],
)
def test_busy_identity_writer_refuses_then_retries_without_losing_state(
    tmp_path: Path, method: str, route: str, payload: dict[str, object]
) -> None:
    """Report a real competing writer's bounded wait and retry the same mutation afterwards."""
    client, token = _lab(tmp_path)
    identity = tmp_path / "identities.json"
    before = identity.read_bytes()
    headers = {"authorization": f"Bearer {token}"}
    with client:
        with identity_store_mutation(identity):
            refused = client.request(method, route, json=payload, headers=headers)
            assert refused.status_code == 503
            assert refused.json() == {"detail": "identity_store_busy"}
            assert identity.read_bytes() == before
        retried = client.request(method, route, json=payload, headers=headers)
        assert retried.status_code == 200, retried.text
    assert identity.read_bytes() != before
