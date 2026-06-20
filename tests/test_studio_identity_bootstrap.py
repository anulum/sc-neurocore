# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio identity bootstrap tests

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import pytest

import sc_neurocore.studio.platform.bootstrap as bootstrap
from sc_neurocore.cli import main
from sc_neurocore.studio.platform.bootstrap import (
    DEFAULT_STUDIO_ADMIN_PRINCIPAL_ID,
    DEFAULT_STUDIO_ADMIN_ROLES,
    MIN_BOOTSTRAP_TOKEN_BYTES,
    bootstrap_studio_admin_identity,
)
from sc_neurocore.studio.platform.identity import (
    StudioIdentityAuthenticator,
    add_studio_browser_user_record,
    load_studio_identity_store,
)


def _fixed_token(_: int) -> str:
    return "generated-admin-token"


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


def test_studio_bootstrap_admin_cli_writes_identity_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sc-neurocore",
            "studio-bootstrap-admin",
            "--identity-file",
            str(identity_path),
            "--principal-id",
            "svc-cli-admin",
            "--role",
            "studio.admin",
        ],
    )

    exit_code = main()
    output = json.loads(capsys.readouterr().out)
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))
    auth_result = authenticator.authenticate_authorization_header(
        f"Bearer {output['bearer_token']}"
    )

    assert exit_code == 0
    assert output["principal_id"] == "svc-cli-admin"
    assert output["environment"] == f"SC_NEUROCORE_STUDIO_IDENTITY_FILE={identity_path}"
    assert auth_result.principal is not None
    assert auth_result.principal.principal_id == "svc-cli-admin"
    assert output["bearer_token"] not in identity_path.read_text(encoding="utf-8")


def test_studio_bootstrap_admin_cli_requires_identity_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["sc-neurocore", "studio-bootstrap-admin"])

    exit_code = main()

    assert exit_code == 1
    assert "--identity-file" in capsys.readouterr().out


def test_studio_add_browser_user_cli_writes_password_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sc-neurocore",
            "studio-add-browser-user",
            "--identity-file",
            str(identity_path),
            "--username",
            "operator",
            "--principal-id",
            "human-operator",
            "--role",
            "studio.viewer",
            "--password-stdin",
        ],
    )
    monkeypatch.setattr("sys.stdin", _StringStdin("browser-secret\n"))

    exit_code = main()
    output = json.loads(capsys.readouterr().out)
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(identity_path))
    auth_result = authenticator.authenticate_browser_user("operator", "browser-secret")

    assert exit_code == 0
    assert output["browser_user"]["username"] == "operator"
    assert output["browser_user"]["principal_id"] == "human-operator"
    assert "password" not in json.dumps(output)
    assert "browser-secret" not in identity_path.read_text(encoding="utf-8")
    assert auth_result.principal is not None
    assert auth_result.principal.roles == frozenset({"studio.viewer"})


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["sc-neurocore", "studio-add-browser-user"], "--identity-file"),
        (
            [
                "sc-neurocore",
                "studio-add-browser-user",
                "--identity-file",
                "identity.json",
            ],
            "--username",
        ),
        (
            [
                "sc-neurocore",
                "studio-add-browser-user",
                "--identity-file",
                "identity.json",
                "--username",
                "operator",
            ],
            "--role",
        ),
        (
            [
                "sc-neurocore",
                "studio-add-browser-user",
                "--identity-file",
                "identity.json",
                "--username",
                "operator",
                "--role",
                "studio.viewer",
            ],
            "--password-stdin",
        ),
    ],
)
def test_studio_add_browser_user_cli_requires_operational_inputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    match: str,
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = main()

    assert exit_code == 1
    assert match in capsys.readouterr().out


class _StringStdin:
    """Small stdin stand-in for CLI password input tests."""

    def __init__(self, text: str) -> None:
        self._text = text

    def readline(self) -> str:
        """Return the configured input once."""

        return self._text
