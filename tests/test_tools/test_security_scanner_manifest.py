# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


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


@contextmanager
def _tempdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)
