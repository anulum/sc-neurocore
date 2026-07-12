# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli studio tests

"""Exercise cli studio behaviour through the public CLI."""

from __future__ import annotations

import io
from pathlib import Path
import sys
from unittest import mock

import pytest

from tests.cli_test_support import run_cli


def test_studio_launches_uvicorn(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        mock.patch("uvicorn.run") as m_uvicorn,
        mock.patch("webbrowser.open") as m_browser,
    ):
        rc = run_cli("studio")
    assert rc == 0
    m_uvicorn.assert_called_once()
    m_browser.assert_called_once_with("http://127.0.0.1:8001")


def test_studio_missing_fastapi(capsys: pytest.CaptureFixture[str]) -> None:
    with mock.patch.dict("sys.modules", {"uvicorn": None}):
        rc = run_cli("studio")
    assert rc == 1
    assert "pip install" in capsys.readouterr().out


def test_studio_command_routes_custom_port() -> None:
    with (
        mock.patch("uvicorn.run") as run_server,
        mock.patch("webbrowser.open") as open_browser,
    ):
        rc = run_cli("studio", "--port", "9000")
    assert rc == 0
    open_browser.assert_called_once_with("http://127.0.0.1:9000")
    assert run_server.call_args.kwargs["port"] == 9000


def test_studio_backup_plan_reports_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Backup-plan validation failures return status one."""
    import sc_neurocore.studio.platform as platform

    monkeypatch.setattr(
        platform,
        "build_studio_backup_plan",
        mock.Mock(side_effect=ValueError("invalid durable target")),
    )
    assert run_cli("studio-backup-plan") == 1
    assert "invalid durable target" in capsys.readouterr().out


def test_studio_bootstrap_reports_identity_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bootstrap I/O failures do not leak past the CLI boundary."""
    import sc_neurocore.studio.platform as platform

    monkeypatch.setattr(
        platform,
        "bootstrap_studio_admin_identity",
        mock.Mock(side_effect=OSError("identity write failed")),
    )
    assert (
        run_cli(
            "studio-bootstrap-admin",
            "--identity-file",
            str(tmp_path / "identities.json"),
        )
        == 1
    )
    assert "identity write failed" in capsys.readouterr().out


def test_studio_deployment_profile_reports_builder_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Profile builder validation errors return status one."""
    import sc_neurocore.studio.platform as platform

    monkeypatch.setattr(
        platform,
        "build_studio_deployment_profile_package",
        mock.Mock(side_effect=ValueError("invalid profile fixture")),
    )
    assert run_cli("studio-deployment-profile") == 1
    assert "invalid profile fixture" in capsys.readouterr().out


def test_studio_add_browser_user_reports_store_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Browser-user store failures remain a controlled command error."""
    import sc_neurocore.studio.platform as platform

    monkeypatch.setattr(sys, "stdin", io.StringIO("correct horse battery staple\n"))
    monkeypatch.setattr(
        platform,
        "add_studio_browser_user_record",
        mock.Mock(side_effect=ValueError("identity store rejected user")),
    )
    assert (
        run_cli(
            "studio-add-browser-user",
            "--identity-file",
            str(tmp_path / "identities.json"),
            "--username",
            "operator",
            "--role",
            "admin",
            "--password-stdin",
        )
        == 1
    )
    assert "identity store rejected user" in capsys.readouterr().out
