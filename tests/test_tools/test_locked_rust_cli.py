# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real Cargo audit configuration refusal

"""Reject database settings that could falsely pass an unanswered Rust audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.security_scan.locked_dependency_inventory import cargo_audit_configuration

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "settings",
    [
        "fetch=false\nstale=false",
        "fetch=true\nstale=true",
        'fetch=true\nstale=false\nurl="https://example.invalid/advisory-db"',
        'fetch=true\nstale=false\npath="local-advisory-db"',
    ],
)
@pytest.mark.parametrize("home_config", [False, True])
def test_public_cli_refuses_unqualified_rust_database(
    tmp_path: Path, settings: str, home_config: bool
) -> None:
    """Reject actual tracked Cargo settings before executing an unanswered audit.

    Parameters
    ----------
    tmp_path : Path
        Fresh real Git checkout and retained refusal packet.
    settings : str
        Database settings that disable freshness or override source custody.
    home_config : bool
        Whether to exercise relative Cargo-home configuration or project precedence.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    shutil.copy(ROOT / "Cargo.lock", checkout / "Cargo.lock")
    configuration_dir = checkout / ("cargo-home" if home_config else ".cargo")
    configuration_dir.mkdir()
    (configuration_dir / "audit.toml").write_text("[database]\n" + settings + "\n")
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(checkout)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True, timeout=30)
    target = tmp_path / "packet"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.security_scan.locked_dependency_audit",
            "--repo-root",
            str(checkout),
            "--ecosystem",
            "rust",
            "--output-dir",
            str(target),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            **({"CARGO_HOME": "cargo-home"} if home_config else {}),
        },
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == []
    assert report["audits"] == [{"path": "Cargo.lock", "passed": False, "error_type": "ValueError"}]
    assert not (target / "lock-1").exists()
    assert (checkout / "Cargo.lock").read_bytes() == (ROOT / "Cargo.lock").read_bytes()


def test_cargo_configuration_binds_effective_project_and_home_inputs(tmp_path: Path) -> None:
    """Follow actual scanner precedence and bind every effective configuration digest.

    Parameters
    ----------
    tmp_path : Path
        Real filesystem directories used only as explicit configuration inputs.
    """
    checkout, cargo_home = tmp_path / "checkout", tmp_path / "cargo-home"
    checkout.mkdir()
    cargo_home.mkdir()
    assert cargo_audit_configuration(checkout, cargo_home) == {}
    home_input = cargo_home / "audit.toml"
    home_input.write_text("[database]\nfetch=true\nstale=false\n")
    home_binding = cargo_audit_configuration(checkout, cargo_home)
    assert home_binding["path"] == str(home_input)
    (checkout / ".cargo").mkdir()
    project_input = checkout / ".cargo/audit.toml"
    project_input.write_text("# Project defaults\n")
    project_binding = cargo_audit_configuration(checkout, cargo_home)
    assert project_binding["path"] == str(project_input)
    assert project_binding["sha256"] != home_binding["sha256"]
    project_input.write_text("# Updated project defaults\n")
    assert cargo_audit_configuration(checkout, cargo_home) != project_binding
    project_input.unlink()
    assert cargo_audit_configuration(checkout, cargo_home) == home_binding
    project_input.symlink_to(home_input)
    with pytest.raises(ValueError, match="regular configuration"):
        cargo_audit_configuration(checkout, cargo_home)
