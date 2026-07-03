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


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MANIFEST = REPO_ROOT / "security" / "release_artifacts_manifest.json"


EXPECTED_ARTIFACT_IDS = {
    "actionlint",
    "bandit",
    "benchmark_regression",
    "cargo_audit",
    "cargo_deny",
    "cargo_fuzz_summary",
    "gitleaks",
    "hypothesis_fuzz_summary",
    "lightweight_scanner_summary",
    "model_data_license_matrix",
    "mypy",
    "osv_scanner",
    "osv_scanner_summary",
    "pip_audit",
    "python_compliance_summary",
    "pyright",
    "release_security_sweep_summary",
    "reuse",
    "ruff",
    "rust_proptest_summary",
    "rust_scanner_summary",
    "scanner_manifest",
    "semgrep",
    "semgrep_summary",
    "syft_cyclonedx",
    "syft_cyclonedx_summary",
    "supply_chain_audit",
    "trivy_fs",
    "typing_scanner_summary",
}

EXPECTED_VULNERABILITY_STATUS_IDS = {
    "cargo_audit",
    "osv_scanner",
    "pip_audit",
    "trivy_fs",
}


def _load_tool() -> Any:
    tool_path = REPO_ROOT / "tools" / "security_scan" / "release_security_artifact_index.py"
    spec = importlib.util.spec_from_file_location("release_security_artifact_index", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_artifacts_manifest_exists_and_is_valid() -> None:
    tool = _load_tool()
    assert CANONICAL_MANIFEST.exists()

    payload = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    report = tool.validate_artifact_manifest(payload)

    assert report["passed"]
    assert report["errors"] == 0
    assert payload["schema_version"] == tool.RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION

    artifact_ids = {entry["id"] for entry in payload["artifacts"]}
    assert artifact_ids == EXPECTED_ARTIFACT_IDS

    vulnerability_status_ids = {entry["id"] for entry in payload["vulnerability_status"]}
    assert vulnerability_status_ids == EXPECTED_VULNERABILITY_STATUS_IDS
    assert all("scanner" in entry for entry in payload["vulnerability_status"])

    required_ids = {entry["id"] for entry in payload["artifacts"] if bool(entry["required"])}
    assert required_ids == {"scanner_manifest", "model_data_license_matrix"}
    required_paths = {
        entry["id"]: entry["path"] for entry in payload["artifacts"] if bool(entry["required"])
    }
    assert required_paths == {
        "scanner_manifest": "security/security_scanner_manifest.json",
        "model_data_license_matrix": "security/model_data_license_matrix.json",
    }


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    tool = _load_tool()
    payload = {
        "schema_version": tool.RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "artifacts": entries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest_with_vulnerability_status(
    path: Path,
    entries: list[dict[str, object]],
    vulnerability_status: list[dict[str, object]],
) -> None:
    tool = _load_tool()
    payload = {
        "schema_version": tool.RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "artifacts": entries,
        "vulnerability_status": vulnerability_status,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_root_with_artifacts(root: Path) -> None:
    security = root / "security"
    security.mkdir(parents=True)
    (security / "scanner_manifest.json").write_text("{}", encoding="utf-8")
    (security / "model_data_license_matrix.json").write_text("{}", encoding="utf-8")


def test_cli_fails_on_missing_required_when_requested(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "index.json"
    root = tmp_path / "release-root"

    _write_manifest(
        manifest,
        [
            {
                "id": "scanner_manifest",
                "path": "security/scanner_manifest.json",
                "required": True,
            },
            {
                "id": "model_data_license_matrix",
                "path": "security/model_data_license_matrix.json",
                "required": True,
            },
            {
                "id": "optional",
                "path": "security/optional.json",
                "required": False,
            },
        ],
    )

    root.mkdir()
    (root / "security").mkdir()
    (root / "security" / "scanner_manifest.json").write_text("{}", encoding="utf-8")

    exit_code = tool.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(root),
            "--output",
            str(output),
            "--fail-on-missing-required",
        ]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["missing_required"] == ["model_data_license_matrix"]
    assert report["missing_optional"] == ["optional"]


def test_optional_missing_does_not_fail_without_required_miss(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "index.json"
    root = tmp_path / "release-root"

    _write_manifest(
        manifest,
        [
            {
                "id": "scanner_manifest",
                "path": "security/scanner_manifest.json",
                "required": True,
            },
            {
                "id": "model_data_license_matrix",
                "path": "security/model_data_license_matrix.json",
                "required": True,
            },
            {
                "id": "benchmark_regression",
                "path": "security/benchmark_regression.json",
                "required": False,
            },
        ],
    )
    _required_root_with_artifacts(root)

    exit_code = tool.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(root),
            "--output",
            str(output),
            "--fail-on-missing-required",
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["missing_required"] == []
    assert report["missing_optional"] == ["benchmark_regression"]


def test_vulnerability_status_summarises_present_scanner_outputs(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = {
        "schema_version": tool.RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "artifacts": [],
        "vulnerability_status": [
            {
                "id": "trivy_fs",
                "path": "security/trivy_fs.json",
                "required": False,
                "scanner": "trivy",
            },
            {
                "id": "pip_audit",
                "path": "security/pip_audit.json",
                "required": False,
                "scanner": "pip-audit",
            },
        ],
    }
    root = tmp_path / "release-root"
    security = root / "security"
    security.mkdir(parents=True)
    (security / "trivy_fs.json").write_text(
        json.dumps(
            {
                "Results": [
                    {
                        "Vulnerabilities": [
                            {"VulnerabilityID": "CVE-1", "Severity": "HIGH"},
                            {"VulnerabilityID": "CVE-2", "Severity": "MEDIUM"},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (security / "pip_audit.json").write_text(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "demo",
                        "vulns": [{"id": "PYSEC-1", "fix_versions": ["2.0"]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = tool.build_artifact_index(manifest, root=root)

    assert report["missing_optional_vulnerability_status"] == []
    assert report["vulnerability_summary"] == {
        "present_status_count": 2,
        "unresolved_count": 3,
        "unresolved_by_severity": {"HIGH": 1, "MEDIUM": 1, "UNKNOWN": 1},
    }
    entries = {entry["id"]: entry for entry in report["vulnerability_status"]}
    assert entries["trivy_fs"]["unresolved_count"] == 2
    assert entries["pip_audit"]["unresolved_count"] == 1


def test_required_vulnerability_status_failure_is_separate_from_artifacts(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "index.json"
    root = tmp_path / "release-root"
    root.mkdir()

    _write_manifest_with_vulnerability_status(
        manifest,
        [],
        [
            {
                "id": "pip_audit",
                "path": "security/pip_audit.json",
                "required": True,
                "scanner": "pip-audit",
            }
        ],
    )

    assert (
        tool.main(
            [
                "--manifest",
                str(manifest),
                "--root",
                str(root),
                "--output",
                str(output),
                "--fail-on-missing-required",
            ]
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["missing_required"] == []
    assert report["missing_required_vulnerability_status"] == ["pip_audit"]


def test_build_and_cli_output_are_deterministic_and_sorted(tmp_path: Path) -> None:
    tool = _load_tool()
    manifest = {
        "schema_version": tool.RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "artifacts": [
            {"id": "ruff", "path": "security/ruff.json", "required": False},
            {"id": "actionlint", "path": "security/actionlint.json", "required": False},
            {"id": "mypy", "path": "security/mypy", "required": False},
        ],
    }
    root = tmp_path / "release-root"
    (root / "security").mkdir(parents=True)
    (root / "security" / "mypy").mkdir()

    first = tool.build_artifact_index(manifest, root=root)
    second = tool.build_artifact_index(manifest, root=root)

    assert first == second
    assert first["schema_version"] == tool.RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION
    assert [entry["id"] for entry in first["artifacts"]] == [
        "actionlint",
        "mypy",
        "ruff",
    ]
    assert first["missing_optional"] == ["actionlint", "ruff"]

    cli_output = tmp_path / "index.json"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest["artifacts"])
    assert (
        tool.main(
            [
                "--manifest",
                str(manifest_path),
                "--root",
                str(root),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )

    with cli_output.open("r", encoding="utf-8") as handle:
        cli_payload = json.load(handle)
    assert cli_payload == first
