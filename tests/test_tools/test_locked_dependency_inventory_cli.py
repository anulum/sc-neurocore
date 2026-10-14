# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real committed dependency input contracts

"""Exercise committed-input discovery and report refusal through public entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import yaml

from tools.security_scan.locked_dependency_inventory import (
    cargo_audit_configuration,
    go_queries,
    inventory_locks,
    julia_queries,
    pixi_package_count,
    require_cargo_audit_configuration,
)

ROOT = Path(__file__).resolve().parents[2]


def test_real_checkout_inventory_preserves_all_lock_bytes() -> None:
    """Discover all maintained ecosystems and retain checksum inputs without mutation."""
    locks = inventory_locks(ROOT)
    for ecosystem in ("python", "rust", "npm", "go", "go-sum", "julia", "pixi"):
        assert any(lock.ecosystem == ecosystem for lock in locks)
    assert {lock.path for lock in locks if lock.ecosystem == "go"} == {
        "src/sc_neurocore/accel/go/go.mod",
        "src/sc_neurocore/accel/go/services/aer_router/go.mod",
        "src/sc_neurocore/accel/go/services/hil_debugger/go.mod",
        "src/sc_neurocore/accel/go/services/services/go.mod",
        "src/sc_neurocore/accel/go/services/services_ext/go.mod",
    }
    assert len([lock for lock in locks if lock.ecosystem == "rust"]) >= 16
    assert len([lock for lock in locks if lock.ecosystem == "python"]) >= 22
    assert inventory_locks(ROOT) == locks


def test_public_cli_refuses_a_directory_without_git_custody(tmp_path: Path) -> None:
    """A real Git discovery failure returns a retained nonzero audit decision.

    Parameters
    ----------
    tmp_path : Path
        Fresh filesystem directory, outside the audited checkout.
    """
    target = tmp_path / "packet"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.security_scan.locked_dependency_audit",
            "--repo-root",
            str(tmp_path),
            "--ecosystem",
            "rust",
            "--output-dir",
            str(target),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == ["CalledProcessError"]


@pytest.mark.parametrize(
    "condition",
    [
        "untracked-profile",
        "unsupported-lock",
        "missing-lock",
        "symlink-lock",
        "orphan-sum",
        "no-lock",
    ],
)
def test_public_cli_refuses_incomplete_tracked_inputs(tmp_path: Path, condition: str) -> None:
    """Refuse incomplete lock custody in a real Git checkout before contacting a scanner.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory holding an independent checkout and retained audit packet.
    condition : str
        Missing tracking, unsupported format, nonregular input, orphan checksum
        or absent required ecosystem to exercise through the audit command.

    Notes
    -----
    Requirement profiles are copied unchanged from the maintained checkout.
    Git and filesystem operations are real; no scanner response is substituted.
    Each refusal retains a failing summary and leaves the input bytes unchanged.
    A missing ecosystem also retains the complete Python input inventory.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(checkout)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "add", "requirements"], cwd=checkout, check=True, capture_output=True, timeout=30
    )
    if condition == "untracked-profile":
        subprocess.run(
            ["git", "rm", "--cached", "requirements/runtime.txt"],
            cwd=checkout,
            check=True,
            capture_output=True,
            timeout=30,
        )
    elif condition != "no-lock":
        name = {"unsupported-lock": "unknown.lock", "orphan-sum": "go.sum"}.get(
            condition, "Cargo.lock"
        )
        lock = checkout / name
        if condition == "symlink-lock":
            lock.symlink_to("requirements/runtime.txt")
        else:
            lock.write_text("tracked dependency input\n")
        subprocess.run(
            ["git", "add", name], cwd=checkout, check=True, capture_output=True, timeout=30
        )
        if condition == "missing-lock":
            lock.unlink()
    target = tmp_path / "packet"
    process = subprocess.run(
        [
            sys.executable,
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == ["ValueError"]
    assert report["audits"] == []
    if condition == "no-lock":
        assert len(report["inputs"]) >= 22
        assert {row["ecosystem"] for row in report["inputs"]} == {"python"}
    else:
        assert report["inputs"] == []
    assert (checkout / "requirements/runtime.txt").read_bytes() == (
        ROOT / "requirements/runtime.txt"
    ).read_bytes()


def test_julia_queries_retain_registry_build_versions() -> None:
    """Query every Julia manifest entry without stripping a package build suffix."""
    manifest = ROOT / "src/sc_neurocore/accel/julia/sc_compte_wm_network/Manifest.toml"
    queries = julia_queries(manifest.read_bytes())
    assert {
        "package": {"ecosystem": "Julia", "name": "FFTW_jll"},
        "version": "3.3.12+0",
    } in queries
    assert {
        "package": {"ecosystem": "Julia", "name": "JLLWrappers"},
        "version": "1.8.0",
    } in queries
    for unsupported in ("1.0", "2.2", "3.0"):
        with pytest.raises(ValueError, match="version-2"):
            julia_queries(f'manifest_format = "{unsupported}"\n[deps]\n'.encode())
    with pytest.raises(ValueError, match="exact registry"):
        julia_queries(
            b'manifest_format = "2.0"\n[[deps.Local]]\npath = "local"\nversion = "1.0.0"\n'
        )


@pytest.mark.parametrize(
    ("manifest", "refusal"),
    [
        (b'manifest_format = "2.0"\n[deps]\nExample = "1.0.0"\n', "entry is invalid"),
        (b'manifest_format = "2.0"\n[deps]\nExample = []\n', "entry is invalid"),
        (b'manifest_format = "2.0"\n[deps]\nExample = ["1.0.0"]\n', "identity is invalid"),
        (b'manifest_format = "2.0"\n[deps]\n', "manifest is empty"),
    ],
)
def test_julia_refuses_unidentified_and_empty_manifests(manifest: bytes, refusal: str) -> None:
    """A manifest without exact per-entry identities cannot produce an empty clean audit.

    Parameters
    ----------
    manifest : bytes
        Version-2 manifest whose dependency table lacks auditable entries.
    refusal : str
        Refusal that names the unavailable identity.
    """
    with pytest.raises(ValueError, match=refusal):
        julia_queries(manifest)


def test_go_queries_bind_standard_library_requirements_and_checksum_versions() -> None:
    """Query the toolchain, every requirement and every checksum-only module version."""
    payload = {
        "Module": {"Path": "example.invalid/service"},
        "Go": "1.27.1",
        "Require": [{"Path": "example.invalid/library", "Version": "v1.4.0"}],
    }
    checksum = (
        "example.invalid/library v1.4.0 h1:AAAA=\n"
        "example.invalid/library v1.4.0/go.mod h1:BBBB=\n"
        "example.invalid/indirect v0.9.1/go.mod h1:CCCC=\n"
    )
    assert go_queries(0, payload, checksum) == (
        {"package": {"ecosystem": "Go", "name": "example.invalid/indirect"}, "version": "0.9.1"},
        {"package": {"ecosystem": "Go", "name": "example.invalid/library"}, "version": "1.4.0"},
        {"package": {"ecosystem": "Go", "name": "stdlib"}, "version": "1.27.1"},
    )
    assert go_queries(0, {"Go": "1.27.1", "Require": None}, None) == (
        {"package": {"ecosystem": "Go", "name": "stdlib"}, "version": "1.27.1"},
    )


@pytest.mark.parametrize(
    ("returncode", "payload", "checksum", "refusal"),
    [
        (1, {"Go": "1.27.1"}, None, "unreplaced module identities"),
        (0, ["1.27.1"], None, "unreplaced module identities"),
        (
            0,
            {"Go": "1.27.1", "Replace": [{"Old": {"Path": "a"}, "New": {"Path": "./b"}}]},
            None,
            "unreplaced module identities",
        ),
        (0, {"Require": []}, None, "standard library version"),
        (0, {"Go": ""}, None, "standard library version"),
        (0, {"Go": "1.27.1", "Require": {"Path": "a"}}, None, "requirements are invalid"),
        (0, {"Go": "1.27.1", "Require": ["a v1.0.0"]}, None, "requirement is invalid"),
        (0, {"Go": "1.27.1", "Require": [{"Path": "a"}]}, None, "exact version"),
        (0, {"Go": "1.27.1", "Require": [{"Path": 1, "Version": "v1.0.0"}]}, None, "exact version"),
        (
            0,
            {"Go": "1.27.1", "Require": [{"Path": "a", "Version": "latest"}]},
            None,
            "exact version",
        ),
        (0, {"Go": "1.27.1"}, "a v1.0.0\n", "checksum input is invalid"),
        (0, {"Go": "1.27.1"}, "a 1.0.0 h1:AAAA=\n", "checksum input is invalid"),
        (0, {"Go": "1.27.1"}, "a v1.0.0 sha256:AAAA=\n", "checksum input is invalid"),
    ],
)
def test_go_queries_refuse_incomplete_parser_output(
    returncode: int, payload: object, checksum: str | None, refusal: str
) -> None:
    """A failed, replaced or inexact module description cannot bound an audit.

    Parameters
    ----------
    returncode : int
        Exit status reported for the manifest parser.
    payload : object
        Parser output that is not a complete exact module description.
    checksum : str or None
        Checksum input, valid or malformed, recorded beside the manifest.
    refusal : str
        Refusal that names the unavailable identity.
    """
    with pytest.raises(ValueError, match=refusal):
        go_queries(returncode, payload, checksum)


def test_pixi_package_count_binds_the_maintained_lock() -> None:
    """Count the real maintained lock and refuse inputs without a package record list."""
    raw = (ROOT / "src/sc_neurocore/accel/mojo/pixi.lock").read_bytes()
    assert pixi_package_count(raw) == len(yaml.safe_load(raw)["packages"]) > 0
    for unavailable in (b"[]\n", b"version: 6\n", b"packages: {}\n"):
        with pytest.raises(ValueError, match="package inventory"):
            pixi_package_count(unavailable)
    with pytest.raises(yaml.YAMLError):
        pixi_package_count(b"packages: [\n")


def test_cargo_settings_refuse_invalid_database_and_drift(tmp_path: Path) -> None:
    """Reject a non-table database setting and any change after the binding was taken.

    Parameters
    ----------
    tmp_path : Path
        Real filesystem directories used only as explicit configuration inputs.
    """
    checkout, cargo_home = tmp_path / "checkout", tmp_path / "cargo-home"
    (checkout / ".cargo").mkdir(parents=True)
    cargo_home.mkdir()
    project_input = checkout / ".cargo/audit.toml"
    project_input.write_text('database = "advisory-db"\n')
    with pytest.raises(ValueError, match="database settings are invalid"):
        cargo_audit_configuration(checkout, cargo_home)
    project_input.unlink()
    absent = cargo_audit_configuration(checkout, cargo_home)
    require_cargo_audit_configuration(checkout, cargo_home, absent)
    project_input.write_text("[database]\nfetch = true\n")
    with pytest.raises(ValueError, match="changed during advisory execution"):
        require_cargo_audit_configuration(checkout, cargo_home, absent)
    present = cargo_audit_configuration(checkout, cargo_home)
    require_cargo_audit_configuration(checkout, cargo_home, present)
    project_input.write_text("[database]\nfetch = true\nstale = false\n")
    with pytest.raises(ValueError, match="changed during advisory execution"):
        require_cargo_audit_configuration(checkout, cargo_home, present)
    project_input.write_text("[database]\nstale = true\n")
    with pytest.raises(ValueError, match="refuses stale acceptance"):
        require_cargo_audit_configuration(checkout, cargo_home, present)
