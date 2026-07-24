# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (cli) from former test_studio_identity_bootstrap.py

from __future__ import annotations

from tests.studio_identity_bootstrap_support import *  # noqa: F403

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


def test_studio_preflight_cli_prints_release_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)
    (tmp_path / "audit").mkdir()
    (tmp_path / "jobs").mkdir()
    _configure_release_preflight_env(monkeypatch, tmp_path, identity_path)
    monkeypatch.setattr(sys, "argv", ["sc-neurocore", "studio-preflight"])

    exit_code = main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["passed"] is True
    assert output["schema_version"] == "studio.preflight.v1"
    assert str(tmp_path) not in json.dumps(output)


def test_studio_preflight_cli_writes_report_to_explicit_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    output_path = tmp_path / "reports" / "studio-preflight.json"
    bootstrap_studio_admin_identity(identity_path, token_factory=_fixed_token)
    (tmp_path / "audit").mkdir()
    (tmp_path / "jobs").mkdir()
    _configure_release_preflight_env(monkeypatch, tmp_path, identity_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sc-neurocore", "studio-preflight", "--output", str(output_path)],
    )

    exit_code = main()
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert output["passed"] is True


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


