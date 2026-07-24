# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity browser-user auth

"""Browser-user authentication and malformed-user rejection contracts."""

from __future__ import annotations

from tests.studio_identity_support import *  # noqa: F403


def test_browser_user_authentication_rejects_empty_expired_and_unknown_users(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": [
                {
                    "active": True,
                    "expires_at_utc": "2020-01-01T00:00:00Z",
                    "password_pbkdf2_sha256": make_browser_user_password_verifier(
                        "operator-password"
                    ),
                    "principal_id": "human-operator",
                    "roles": ["studio.admin"],
                    "username": "operator",
                }
            ],
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))

    empty = authenticator.authenticate_browser_user("", "operator-password")
    expired = authenticator.authenticate_browser_user("operator", "operator-password")
    unknown = authenticator.authenticate_browser_user("missing", "operator-password")

    assert empty.failure_reason == "invalid_browser_login"
    assert expired.failure_reason == "expired_browser_user"
    assert unknown.failure_reason == "invalid_browser_login"


@pytest.mark.parametrize(
    ("browser_user", "match"),
    [
        ("not-an-object", "browser user 0"),
        (
            {
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "human-operator",
                "roles": ["studio.admin"],
                "username": "",
            },
            "username",
        ),
        (
            {
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "",
                "roles": ["studio.admin"],
                "username": "operator",
            },
            "principal_id",
        ),
        (
            {
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "human-operator",
                "roles": [],
                "username": "operator",
            },
            "roles",
        ),
        (
            {
                "password_pbkdf2_sha256": "not-a-verifier",
                "principal_id": "human-operator",
                "roles": ["studio.admin"],
                "username": "operator",
            },
            "password verifier",
        ),
        (
            {
                "active": "yes",
                "password_pbkdf2_sha256": make_browser_user_password_verifier("pw"),
                "principal_id": "human-operator",
                "roles": ["studio.admin"],
                "username": "operator",
            },
            "active",
        ),
    ],
)
def test_studio_identity_store_rejects_malformed_browser_users(
    tmp_path: Path,
    browser_user: object,
    match: str,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": [browser_user],
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )

    with pytest.raises(ValueError, match=match):
        load_studio_identity_store(identity_path)


def test_studio_identity_store_rejects_non_list_browser_users(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": "operator",
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )

    with pytest.raises(ValueError, match="browser_users"):
        load_studio_identity_store(identity_path)


def test_browser_user_mutations_preserve_non_target_users(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_payload(
        identity_path,
        {
            "browser_users": [
                {
                    "active": True,
                    "expires_at_utc": None,
                    "password_pbkdf2_sha256": make_browser_user_password_verifier(
                        "operator-password"
                    ),
                    "principal_id": "human-operator",
                    "roles": ["studio.admin"],
                    "username": "operator",
                },
                {
                    "active": True,
                    "expires_at_utc": None,
                    "password_pbkdf2_sha256": make_browser_user_password_verifier(
                        "analyst-password"
                    ),
                    "principal_id": "human-analyst",
                    "roles": ["studio.viewer"],
                    "username": "analyst",
                },
            ],
            "schema_version": "sc-neurocore.studio.identity.v1",
            "service_accounts": [],
        },
    )

    update_studio_browser_user_record(
        identity_path,
        active=True,
        expires_at_utc=None,
        roles=["studio.admin", "studio.viewer"],
        username="operator",
    )
    rotate_studio_browser_user_password(
        identity_path,
        password="rotated-password",
        username="operator",
    )
    store = load_studio_identity_store(identity_path)
    authenticator = StudioIdentityAuthenticator(store)

    assert authenticator.authenticate_browser_user("operator", "rotated-password").principal
    assert authenticator.authenticate_browser_user("analyst", "analyst-password").principal


def test_add_browser_user_rejects_whitespace_username(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    _write_identity_file(identity_path, "admin-token")

    with pytest.raises(ValueError, match="whitespace"):
        add_studio_browser_user_record(
            identity_path,
            password="operator-password",
            principal_id="human-operator",
            roles=["studio.viewer"],
            username="bad user",
        )


@pytest.mark.parametrize(
    "encoded_verifier",
    [
        "sha1$390000$" + ("0" * 32) + "$" + ("0" * 64),
        "pbkdf2_sha256$not-int$" + ("0" * 32) + "$" + ("0" * 64),
        "pbkdf2_sha256$1$" + ("0" * 32) + "$" + ("0" * 64),
        "pbkdf2_sha256$390000$00$" + ("0" * 64),
        "pbkdf2_sha256$390000$" + ("0" * 32) + "$not-a-hash",
    ],
)
def test_browser_password_verifier_rejects_malformed_encodings(
    encoded_verifier: str,
) -> None:
    assert not verify_browser_user_password("password", encoded_verifier)
