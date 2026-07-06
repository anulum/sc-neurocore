# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest


def _load_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    spec = importlib.util.spec_from_file_location("security_scanner_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_contains_required_security_scanners() -> None:
    tool = _load_tool()

    manifest = tool.build_scanner_manifest()
    required = {
        "pip-audit",
        "osv-scanner",
        "cargo-audit",
        "cargo-deny",
        "gitleaks",
        "semgrep",
        "trivy fs",
        "syft-cyclonedx",
        "reuse",
        "actionlint",
        "pyright",
        "mypy",
        "bandit",
        "ruff",
        "benchmark-regression",
        "cargo-fuzz-nightly",
    }

    scanner_names = {scanner["name"] for scanner in manifest["scanners"]}
    assert required <= scanner_names


def test_scanner_records_ownership_and_noise_fields() -> None:
    tool = _load_tool()
    manifest = tool.build_scanner_manifest()

    for scanner in manifest["scanners"]:
        assert isinstance(scanner["ecosystem"], str) and scanner["ecosystem"]
        assert isinstance(scanner["cadence"], str) and scanner["cadence"]
        assert isinstance(scanner["blocking_policy"], str)
        assert scanner["blocking_policy"] in {"blocking", "allowed_to_fail"}
        assert isinstance(scanner["command"], str) and scanner["command"]
        assert isinstance(scanner["inputs"], list)
        assert isinstance(scanner["owner"], str) and scanner["owner"]
        assert isinstance(scanner["noise"], str)


def test_non_blocking_scanners_require_rationale() -> None:
    tool = _load_tool()
    manifest = tool.build_scanner_manifest()

    for scanner in manifest["scanners"]:
        if scanner["blocking_policy"] == "allowed_to_fail":
            assert isinstance(scanner["allowed_to_fail_rationale"], str)


def test_manifest_validation_and_deterministic_json() -> None:
    tool = _load_tool()

    first = json.dumps(tool.build_scanner_manifest(), sort_keys=True)
    second = json.dumps(tool.build_scanner_manifest(), sort_keys=True)
    assert first == second

    payload = json.loads(first)
    report = tool.validate_scanner_manifest(payload)
    assert report["passed"]
    assert not any(finding["level"] == "error" for finding in report["findings"])


def test_manifest_validation_reports_schema_and_scanner_list_errors() -> None:
    tool = _load_tool()

    report = tool.validate_scanner_manifest(
        {"schema_version": "wrong-version", "scanners": "not-a-list"}
    )

    messages = {finding["message"] for finding in report["findings"]}
    assert not report["passed"]
    assert "schema_version is missing or not equal to expected version" in messages
    assert "scanners must be a list" in messages


def test_manifest_validation_reports_malformed_scanner_entries() -> None:
    tool = _load_tool()
    payload = tool.build_scanner_manifest()
    payload["scanners"] = [
        "not-an-object",
        {
            "name": "pip-audit",
            "ecosystem": "python",
            "cadence": "on-demand",
            "blocking_policy": "allowed_to_fail",
            "command": "pip-audit",
            "inputs": [],
            "owner": "owner",
            "noise": "low",
            "pinned_version": "pip-audit==2.9.0",
        },
        {
            "name": "pip-audit",
            "ecosystem": "python",
            "cadence": "on-demand",
            "blocking_policy": "blocking",
            "command": "pip-audit",
            "inputs": ["bad-input", {"path": 42}],
            "owner": "owner",
            "noise": "low",
            "pinned_version": "pip-audit==2.9.0",
        },
        {"name": "custom", "inputs": [{"path": "pyproject.toml"}]},
    ]

    report = tool.validate_scanner_manifest(payload)
    messages = "\n".join(finding["message"] for finding in report["findings"])

    assert not report["passed"]
    assert "scanner entry must be an object" in messages
    assert "scanner custom missing fields:" in messages
    assert "duplicate scanner name pip-audit" in messages
    assert "uses allowed_to_fail but has no rationale" in messages
    assert "scanner pip-audit has no declared inputs" in messages
    assert "scanner pip-audit input entry must be object" in messages
    assert "scanner pip-audit input.path must be string" in messages
    assert "required scanner missing from manifest: cargo-fuzz-nightly" in messages


def test_required_scanner_contract_includes_every_manifest_scanner() -> None:
    tool = _load_tool()

    manifest = tool.build_scanner_manifest()
    scanner_names = {scanner["name"] for scanner in manifest["scanners"]}

    assert scanner_names == tool._required_scanner_names()


def test_json_output_scanners_are_release_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tool = _load_tool()

    release_manifest_path = repo_root / "security" / "release_artifacts_manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {
        artifact["path"]
        for artifact in release_manifest["artifacts"]
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
    }

    for scanner in tool.build_scanner_manifest()["scanners"]:
        command = scanner["command"]
        if "security/" not in command:
            continue
        expected_paths = set()
        tokens = command.replace("=", " ").split()
        for index, token in enumerate(tokens):
            if index > 0 and tokens[index - 1] in {
                "--baseline",
                "--cache-dir",
                "--current",
                "--output-dir",
            }:
                continue
            if token.startswith("security/"):
                expected_paths.add(token)
            elif token in {"--output", "--output-file", "--report-path", "--file"}:
                output_path = tokens[index + 1]
                if output_path.startswith("security/"):
                    expected_paths.add(output_path)

        assert expected_paths <= artifact_paths, scanner["name"]


def test_bandit_scanner_excludes_vendored_tool_environments() -> None:
    tool = _load_tool()
    manifest = tool.build_scanner_manifest()

    bandit = next(scanner for scanner in manifest["scanners"] if scanner["name"] == "bandit")

    assert "src/sc_neurocore/accel/mojo/.pixi" in bandit["command"]
    assert "--severity-level medium" in bandit["command"]
    assert "security/bandit.json" in bandit["command"]


def test_cli_writes_manifest_and_can_validate() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    tool = _load_tool()

    with _tempdir() as tmpdir:
        manifest_path = tmpdir / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--output",
                str(manifest_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == tool.SCAN_MANIFEST_SCHEMA_VERSION

        validate = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--validate",
                str(manifest_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert validate.returncode == 0


def test_main_prints_writes_and_validates_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    manifest_path = tmp_path / "manifest.json"
    invalid_path = tmp_path / "invalid.json"

    assert tool.main([]) == 0
    stdout = capsys.readouterr().out
    assert "cargo-fuzz-nightly" in stdout

    assert tool.main(["--output", str(manifest_path)]) == 0
    assert manifest_path.exists()
    assert tool.main(["--validate", str(manifest_path)]) == 0
    valid_report = capsys.readouterr().out
    assert '"passed": true' in valid_report

    invalid_path.write_text(json.dumps({"scanners": []}), encoding="utf-8")
    assert tool.main(["--validate", str(invalid_path)]) == 1
    invalid_report = capsys.readouterr().out
    assert "schema_version is missing or not equal to expected version" in invalid_report


def test_script_entrypoint_runs_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    monkeypatch.setattr(sys, "argv", [str(tool_path)])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(tool_path), run_name="__main__")

    assert exit_info.value.code == 0
    assert "cargo-fuzz-nightly" in capsys.readouterr().out


@contextmanager
def _tempdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)
