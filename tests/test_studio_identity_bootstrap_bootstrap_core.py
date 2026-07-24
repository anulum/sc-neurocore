# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (bootstrap_core) from former test_studio_identity_bootstrap.py

from __future__ import annotations

from tests.studio_identity_bootstrap_support import *  # noqa: F403


def test_bootstrap_studio_admin_identity_creates_authenticating_store(tmp_path: Path) -> None:
    identity_path = tmp_path / "private" / "studio-identities.json"

    result = bootstrap_studio_admin_identity(
        identity_path,
        token_factory=_fixed_token,
        expires_at_utc="2030-01-01T00:00:00+00:00",
    )

    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    account = payload["service_accounts"][0]
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))
    auth_result = authenticator.authenticate_authorization_header(f"Bearer {result.bearer_token}")

    assert result.principal_id == DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID
    assert result.roles == DEFAULT_STUDIO_ADMIN_ROLES
    assert result.parent_directory_created is True
    assert result.token_sha256 == hashlib.sha256(b"generated-admin-token").hexdigest()
    assert result.expires_at_utc == "2030-01-01T00:00:00Z"
    assert account["token_sha256"] == result.token_sha256
    assert "generated-admin-token" not in identity_path.read_text(encoding="utf-8")
    assert auth_result.principal is not None
    assert auth_result.principal.roles == frozenset(DEFAULT_STUDIO_ADMIN_ROLES)
    if os.name == "posix":
        assert result.file_permissions_hardened is True
        assert identity_path.stat().st_mode & 0o777 == 0o600
        assert identity_path.parent.stat().st_mode & 0o777 == 0o700


def test_bootstrap_public_dict_excludes_bearer_token(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"

    result = bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)
    public_payload = result.to_public_dict()

    assert public_payload["principal_id"] == DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID
    assert public_payload["roles"] == list(DEFAULT_STUDIO_ADMIN_ROLES)
    assert "bearer_token" not in public_payload
    assert "generated-admin-token" not in json.dumps(public_payload)


def test_bootstrap_refuses_existing_identity_without_overwrite(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)

    assert identity_path.read_text(encoding="utf-8") == "existing"


def test_bootstrap_overwrite_replaces_existing_identity(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text("existing", encoding="utf-8")

    result = bootstrap_studio_admin_identity(
        identity_path,
        principal_id="svc-admin-2",
        roles=("studio.viewer", "studio.admin", "studio.admin"),
        overwrite=True,
        token_factory=_fixed_token,
    )
    payload = json.loads(identity_path.read_text(encoding="utf-8"))

    assert result.principal_id == "svc-admin-2"
    assert result.roles == ("studio.viewer", "studio.admin")
    assert payload["service_accounts"][0]["principal_id"] == "svc-admin-2"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"principal_id": " "}, "principal_id"),
        ({"principal_id": "svc admin"}, "whitespace"),
        ({"roles": ()}, "role"),
        ({"roles": ("studio.viewer",)}, "studio.admin"),
        ({"roles": ("studio.admin", "")}, "role"),
        ({"token_bytes": MIN_BOOTSTRAP_TOKEN_BYTES - 1}, "at least"),
        ({"expires_at_utc": ""}, "expiry"),
        ({"expires_at_utc": "not-a-date"}, "ISO-8601"),
        ({"expires_at_utc": "2030-01-01T00:00:00"}, "timezone"),
        ({"token_factory": lambda _: ""}, "invalid token"),
    ],
)
def test_bootstrap_rejects_invalid_inputs(
    tmp_path: Path,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        bootstrap_studio_admin_identity(
            tmp_path / "studio-identities.json",
            **kwargs,
        )


def test_bootstrap_rejects_parent_path_that_is_file(tmp_path: Path) -> None:
    parent_path = tmp_path / "not-a-directory"
    parent_path.write_text("file", encoding="utf-8")

    with pytest.raises(ValueError, match="parent path"):
        bootstrap_studio_admin_identity(parent_path / "studio-identities.json")


def test_bootstrap_create_failure_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_path = tmp_path / "studio-identities.json"

    class BrokenHandle:
        def __init__(self, descriptor: int) -> None:
            self._descriptor = descriptor

        def __enter__(self) -> "BrokenHandle":
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> Literal[False]:
            os.close(self._descriptor)
            return False

        def write(self, text: str) -> int:
            del text
            raise OSError("simulated write failure")

    def broken_fdopen(
        descriptor: int,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
    ) -> BrokenHandle:
        del mode, buffering, encoding
        return BrokenHandle(descriptor)

    monkeypatch.setattr("sc_neurocore.studio.platform.bootstrap.os.fdopen", broken_fdopen)

    with pytest.raises(OSError, match="simulated write failure"):
        bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)

    assert not identity_path.exists()


def test_bootstrap_replace_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text("existing", encoding="utf-8")

    def failing_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr("sc_neurocore.studio.platform.bootstrap.os.replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        bootstrap_studio_admin_identity(
            identity_path,
            overwrite=True,
            token_factory=_fixed_token,
        )

    assert identity_path.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.glob(".studio-identities.json.*")) == []


def test_bootstrap_permission_hardening_reports_false_without_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr("sc_neurocore.studio.platform.bootstrap.os.name", "nt")

    hardened = bootstrap._chmod_owner_only(identity_path, directory=False)

    assert hardened is False
