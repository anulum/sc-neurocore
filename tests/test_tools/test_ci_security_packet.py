# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "security_scan" / "ci_security_packet.py"
RELEASE_SCHEMA_VERSION = "sc-neurocore.release-security-artifact-index.v1"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location("ci_security_packet", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_release_manifest(root: Path, artifacts: list[dict[str, object]]) -> None:
    release_root = root / "security"
    release_root.mkdir(parents=True)
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "artifacts": artifacts,
    }
    (release_root / "release_artifacts_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_model_data_matrix(root: Path) -> None:
    (root / "security").mkdir(parents=True, exist_ok=True)
    (root / "security" / "model_data_license_matrix.json").write_text(
        "{}\n",
        encoding="utf-8",
    )


def _read_summary(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def test_ci_security_packet_builds_expected_outputs_and_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_release_manifest(
        repo_root,
        [
            {
                "id": "scanner_manifest",
                "path": "security/security_scanner_manifest.json",
                "required": True,
            },
            {
                "id": "model_data_license_matrix",
                "path": "security/model_data_license_matrix.json",
                "required": True,
            },
            {
                "id": "optional_report",
                "path": "security/optional_report.json",
                "required": False,
            },
        ],
    )
    _write_model_data_matrix(repo_root)

    output_dir = tmp_path / "packet"
    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(tool, "_project_root", lambda: repo_root)
        assert tool.main(["--output-dir", str(output_dir)]) == 0

    summary = _read_summary(capsys.readouterr().out)
    assert summary["schema_version"] == tool.CI_SECURITY_PACKET_SCHEMA_VERSION
    assert summary["output_dir"] == str(output_dir.resolve())
    assert summary["missing_required"] == []
    assert summary["missing_optional"] == ["optional_report"]

    artifact_ids = [entry["id"] for entry in summary["artifact_paths"]]
    assert artifact_ids == sorted(artifact_ids)

    assert (output_dir / "security_scanner_manifest.json").exists()
    assert (output_dir / "security" / "security_scanner_manifest.json").exists()
    assert (output_dir / "model_data_license_matrix.json").exists()
    assert (output_dir / "security" / "model_data_license_matrix.json").exists()
    assert (output_dir / "python_code_scanner_plan.json").exists()
    assert (output_dir / "rust_supply_chain_scanner_plan.json").exists()
    assert (output_dir / "release_security_artifact_index.json").exists()


def test_ci_security_packet_does_not_execute_scanner_binaries(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_release_manifest(
        repo_root,
        [
            {
                "id": "scanner_manifest",
                "path": "security/security_scanner_manifest.json",
                "required": True,
            },
            {
                "id": "model_data_license_matrix",
                "path": "security/model_data_license_matrix.json",
                "required": True,
            },
        ],
    )
    _write_model_data_matrix(repo_root)

    call_log = {"python": 0, "rust": 0}

    def fake_python_plan(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        call_log["python"] += 1
        return {
            "schema_version": "sc-neurocore.python-code-scanner-plan.v1",
            "scanner_count": 0,
            "scanners": [],
        }

    def fake_rust_plan(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        call_log["rust"] += 1
        return {
            "schema_version": "sc-neurocore.security-supply-chain-plan.v1",
            "include_heavy": False,
            "scanner_count": 0,
            "scanners": [],
        }

    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(tool, "_project_root", lambda: repo_root)
        monkeypatch.setattr(tool, "build_scanner_plan", fake_python_plan)
        monkeypatch.setattr(tool, "build_rust_supply_chain_plan", fake_rust_plan)

        assert tool.main(["--output-dir", str(tmp_path / "packet")]) == 0

    assert call_log["python"] == 1
    assert call_log["rust"] == 1


def test_cli_include_heavy_and_fail_flag_forwards_to_missing_required_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_release_manifest(
        repo_root,
        [
            {
                "id": "scanner_manifest",
                "path": "security/security_scanner_manifest.json",
                "required": True,
            },
            {
                "id": "model_data_license_matrix",
                "path": "security/model_data_license_matrix.json",
                "required": True,
            },
            {
                "id": "required_missing",
                "path": "security/required_missing.json",
                "required": True,
            },
        ],
    )
    _write_model_data_matrix(repo_root)

    call_log: dict[str, bool | None] = {"include_heavy": None}

    def fake_rust_plan(
        _manifest_payload: dict[str, object],
        *,
        repo_root: Path,
        include_heavy: bool,
        has_executable: Any = None,
    ) -> dict[str, Any]:
        call_log["include_heavy"] = include_heavy
        return {
            "schema_version": "sc-neurocore.security-supply-chain-plan.v1",
            "include_heavy": include_heavy,
            "scanner_count": 0,
            "scanners": [],
        }

    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(tool, "_project_root", lambda: repo_root)
        monkeypatch.setattr(
            tool,
            "build_scanner_plan",
            lambda *_args, **_kwargs: {
                "schema_version": "sc-neurocore.python-code-scanner-plan.v1",
                "scanner_count": 0,
                "scanners": [],
            },
        )
        monkeypatch.setattr(tool, "build_rust_supply_chain_plan", fake_rust_plan)
        monkeypatch.setattr(
            tool,
            "build_artifact_index",
            lambda *_args, **_kwargs: {
                "schema_version": tool.RELEASE_SECURITY_ARTIFACT_INDEX_SCHEMA_VERSION,
                "required_count": 2,
                "optional_count": 0,
                "missing_required": ["required_missing"],
                "missing_optional": [],
                "artifacts": [],
            },
        )

        exit_code = tool.main(
            [
                "--output-dir",
                str(tmp_path / "packet"),
                "--include-heavy",
                "--fail-on-missing-required",
            ]
        )

    assert exit_code == 1
    summary = _read_summary(capsys.readouterr().out)
    assert summary["missing_required"] == ["required_missing"]
    assert call_log["include_heavy"] is True


def test_fail_on_missing_required_includes_scanner_plan_input_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_release_manifest(
        repo_root,
        [
            {
                "id": "scanner_manifest",
                "path": "security/security_scanner_manifest.json",
                "required": True,
            },
            {
                "id": "model_data_license_matrix",
                "path": "security/model_data_license_matrix.json",
                "required": True,
            },
        ],
    )
    _write_model_data_matrix(repo_root)

    python_plan = {
        "schema_version": "sc-neurocore.python-code-scanner-plan.v1",
        "scanner_count": 1,
        "scanners": [
            {
                "name": "pip-audit",
                "run_class": "missing_required_input",
                "executable": "pip-audit",
                "missing_required_inputs": ["requirements/release.txt"],
            }
        ],
    }
    rust_plan = {
        "schema_version": "sc-neurocore.security-supply-chain-plan.v1",
        "include_heavy": False,
        "scanner_count": 1,
        "scanners": [
            {
                "name": "cargo-audit",
                "run_class": "missing_required_input",
                "executable": "cargo",
                "missing_required_inputs": ["Cargo.lock"],
            }
        ],
    }

    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(tool, "_project_root", lambda: repo_root)
        monkeypatch.setattr(tool, "build_scanner_plan", lambda *_args, **_kwargs: python_plan)
        monkeypatch.setattr(
            tool,
            "build_rust_supply_chain_plan",
            lambda *_args, **_kwargs: rust_plan,
        )

        exit_code = tool.main(
            [
                "--output-dir",
                str(tmp_path / "packet"),
                "--fail-on-missing-required",
            ]
        )

    assert exit_code == 1
    summary = _read_summary(capsys.readouterr().out)
    assert summary["missing_required_scanner_inputs"] == [
        {
            "inputs": ["requirements/release.txt"],
            "plan": "python_code_scanner_plan",
            "scanner": "pip-audit",
        },
        {
            "inputs": ["Cargo.lock"],
            "plan": "rust_supply_chain_scanner_plan",
            "scanner": "cargo-audit",
        },
    ]
