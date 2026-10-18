# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio deployment profile tests

"""Verify emitted Studio profiles against their runtime and operator contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.cli import main
from sc_neurocore.studio.platform.deployment_profiles import (
    STUDIO_DEPLOYMENT_PROFILE_SCHEMA_VERSION,
    StudioDeploymentPackageName,
    build_studio_deployment_profile_package,
    list_studio_deployment_profile_packages,
)
from sc_neurocore.studio.platform.settings import build_default_studio_runtime_settings


def test_lab_package_builds_the_single_operator_loopback_settings() -> None:
    """Build usable lab runtime settings directly from the emitted package."""
    package = build_studio_deployment_profile_package("lab")
    settings = build_default_studio_runtime_settings(package.environment)

    assert settings.deployment_profile == "production"
    assert settings.enforce_route_policies is True
    assert settings.allow_header_principal is False
    assert settings.allowed_hosts == ("127.0.0.1", "localhost")
    assert settings.cors_allowed_origins == (
        "http://127.0.0.1:8001",
        "http://localhost:8001",
    )
    assert settings.websocket_allowed_origins == settings.cors_allowed_origins
    assert settings.identity_file_path == package.environment["SC_NEUROCORE_STUDIO_IDENTITY_FILE"]
    assert settings.audit_log_path is not None
    assert settings.audit_rotation_bytes == 104857600
    assert settings.audit_retained_files == 14
    assert "one API process per identity store" in package.security_controls


@pytest.mark.parametrize(
    ("name", "runtime_profile"),
    [("local", "development"), ("lab", "production"), ("server", "production")],
)
def test_studio_deployment_profile_packages_are_secret_free(
    name: StudioDeploymentPackageName,
    runtime_profile: str,
) -> None:
    """Emit each declared deployment package without embedding operator secrets."""
    package = build_studio_deployment_profile_package(name)
    payload = package.to_public_dict()
    encoded = json.dumps(payload)

    assert payload["schema_version"] == STUDIO_DEPLOYMENT_PROFILE_SCHEMA_VERSION
    assert payload["name"] == name
    assert payload["runtime_profile"] == runtime_profile
    assert "password" not in encoded.lower()
    assert "token" not in encoded.lower()
    assert "/home/" not in encoded
    assert "/media/" not in encoded
    assert "*" not in json.dumps(payload["environment"])


def test_studio_deployment_profile_lab_and_server_are_fail_closed() -> None:
    """Require identity enforcement in the emitted lab and server templates."""
    for name in ("lab", "server"):
        package = build_studio_deployment_profile_package(name)
        environment = package.environment

        assert environment["SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE"] == "production"
        assert environment["SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES"] == "true"
        assert environment["SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL"] == "false"
        assert "SC_NEUROCORE_STUDIO_IDENTITY_FILE" in environment
        assert "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH" in environment
        assert "SC_NEUROCORE_STUDIO_JOB_ROOT" in environment
        assert int(environment["SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS"]) > 0
        assert int(environment["SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES"]) > 0
        assert "process CPU and memory ceilings for EDA jobs" in package.security_controls
        assert package.preflight_command.endswith("--output studio-preflight.json")
        assert "<identity-file>" in package.backup_items


def test_studio_deployment_profile_local_is_loopback_development() -> None:
    """Keep the local template on explicit loopback development settings."""
    package = build_studio_deployment_profile_package("local")
    environment = package.environment

    assert environment["SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE"] == "development"
    assert environment["SC_NEUROCORE_STUDIO_ALLOWED_HOSTS"] == "127.0.0.1,localhost"
    assert environment["SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL"] == "true"
    assert environment["SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES"] == "false"
    assert package.required_operator_inputs == ("<local-job-root>",)


def test_studio_deployment_profile_env_lines_are_quoted_and_sorted() -> None:
    """Emit deterministic shell environment lines with quoted values."""
    package = build_studio_deployment_profile_package("server")
    env_lines = package.to_env_lines()

    assert env_lines == tuple(sorted(env_lines))
    assert env_lines[0].startswith("export SC_NEUROCORE_STUDIO_ALLOWED_HOSTS=")
    assert all(line.endswith("'") for line in env_lines)
    assert any("SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES='104857600'" in line for line in env_lines)


def test_list_studio_deployment_profile_packages_returns_all_profiles() -> None:
    """List the local, lab and server deployment templates in stable order."""
    packages = list_studio_deployment_profile_packages()

    assert tuple(package.name for package in packages) == ("local", "lab", "server")


def test_studio_deployment_profile_rejects_unknown_name() -> None:
    """Refuse a deployment-template name outside the declared catalogue."""
    unknown_name: Any = "staging"

    with pytest.raises(ValueError, match="local, lab, or server"):
        build_studio_deployment_profile_package(unknown_name)


def test_studio_deployment_profile_cli_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Emit the selected deployment package through the real CLI."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["sc-neurocore", "studio-deployment-profile", "--studio-profile", "lab"],
    )

    exit_code = main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["name"] == "lab"
    assert output["runtime_profile"] == "production"
    assert output["environment"]["SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL"] == "false"


def test_studio_deployment_profile_cli_writes_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write the selected shell environment without duplicating it on stdout."""
    output_path = tmp_path / "studio-server.env"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sc-neurocore",
            "studio-deployment-profile",
            "--studio-profile",
            "server",
            "--format",
            "env",
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    output = output_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE='production'" in output
    assert "SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES='104857600'" in output
