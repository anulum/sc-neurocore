# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (browser_user_records) from former test_studio_identity_bootstrap.py

from __future__ import annotations

from tests.studio_identity_bootstrap_support import *  # noqa: F403

def test_add_studio_browser_user_record_preserves_service_account(
    tmp_path: Path,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)

    public_record = add_studio_browser_user_record(
        identity_path,
        username="operator",
        principal_id="human-operator",
        roles=("studio.viewer", "studio.admin", "studio.viewer"),
        password="browser-secret",
        expires_at_utc="2030-01-01T00:00:00+00:00",
    )
    store = load_studio_identity_store(identity_path)
    authenticator = StudioIdentityAuthenticator(store)
    auth_result = authenticator.authenticate_browser_user("operator", "browser-secret")
    payload = json.loads(identity_path.read_text(encoding="utf-8"))

    assert public_record.username == "operator"
    assert public_record.principal_id == "human-operator"
    assert public_record.roles == ("studio.admin", "studio.viewer")
    assert public_record.expires_at_utc == "2030-01-01T00:00:00Z"
    assert len(store.service_accounts) == 1
    assert len(store.browser_users) == 1
    assert payload["browser_users"][0]["password_pbkdf2_sha256"].startswith("pbkdf2_sha256$")
    assert "browser-secret" not in identity_path.read_text(encoding="utf-8")
    assert auth_result.principal is not None
    assert auth_result.principal.principal_id == "human-operator"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"username": " "}, "username"),
        ({"username": "human operator"}, "whitespace"),
        ({"roles": ()}, "roles"),
        ({"password": ""}, "password"),
    ],
)
def test_add_studio_browser_user_rejects_invalid_inputs(
    tmp_path: Path,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)

    arguments: dict[str, Any] = {
        "username": "operator",
        "principal_id": "human-operator",
        "roles": ("studio.viewer",),
        "password": "browser-secret",
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=match):
        add_studio_browser_user_record(identity_path, **arguments)


def test_add_studio_browser_user_rejects_duplicate_username(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)
    add_studio_browser_user_record(
        identity_path,
        username="operator",
        principal_id="human-operator",
        roles=("studio.viewer",),
        password="browser-secret",
    )

    with pytest.raises(ValueError, match="already exists"):
        add_studio_browser_user_record(
            identity_path,
            username="operator",
            principal_id="human-operator-2",
            roles=("studio.viewer",),
            password="browser-secret-2",
        )


