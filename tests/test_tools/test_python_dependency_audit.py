# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete dependency report and CLI acceptance

"""Exercise exact report validation and real fail-closed CLI output."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools.security_scan.python_dependency_audit import (
    run_dependency_audit,
    validate_dependency_report,
)

REPO = Path(__file__).resolve().parents[2]


def test_real_repository_removal_during_execution_invalidates_coverage(tmp_path: Path) -> None:
    """A real filesystem loss cannot reuse stale reports or qualify input coverage."""
    checkout = tmp_path / "checkout"
    shutil.copytree(REPO / "requirements", checkout / "requirements")
    parked = tmp_path / "retained-inputs"
    calls = 0

    def execute(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Remove the owned checkout once and run the actual bounded subprocess."""
        nonlocal calls
        calls += 1
        if calls == 1:
            checkout.rename(parked)
        return subprocess.run(command, **kwargs)

    packet = tmp_path / "packet"
    report = run_dependency_audit(repo_root=checkout, output_dir=packet, run_command=execute)
    assert calls > 0 and parked.is_dir() and not checkout.exists()
    assert report["passed"] is False and report["coverage_complete"] is False
    assert report["dependencies"] == []
    assert report["errors"] == ["Python audit inputs changed during execution."]
    assert all(
        batch["returncode"] is None
        and batch["failure_kind"] == "FileNotFoundError"
        and batch["report_sha256"] is None
        and batch["coverage_complete"] is False
        for profile in report["profiles"]
        for batch in profile["batches"]
    )
    assert json.loads((packet / "security/pip_audit.json").read_text()) == json.loads(
        json.dumps(report)
    )


def test_report_retains_findings_and_accepts_normalised_package_identity() -> None:
    """A known finding remains present when the report identity is canonicalised."""
    finding = {"id": "PYSEC-2021-108", "fix_versions": ["1.26.5"]}
    payload = {"dependencies": [{"name": "urllib3", "version": "1.26.4", "vulns": [finding]}]}
    result = validate_dependency_report(payload, {("urllib3", "1.26.4")})
    assert result[("urllib3", "1.26.4")]["vulns"] == [finding]
    assert validate_dependency_report(
        {"dependencies": [{"name": "tomli_w", "version": "1.2.0", "vulns": []}]},
        {("tomli-w", "1.2.0")},
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"dependencies": {}},
        {"dependencies": []},
        {"dependencies": [None]},
        {"dependencies": [{"name": "numpy", "skip_reason": "not audited"}]},
        {"dependencies": [{"name": 1, "version": "2.5.3", "vulns": []}]},
        {"dependencies": [{"name": "numpy", "version": "invalid", "vulns": []}]},
        {"dependencies": [{"name": "numpy", "version": "2.5.2", "vulns": []}]},
        {"dependencies": [{"name": "numpy", "version": "2.5.3", "vulns": None}]},
        {"dependencies": [{"name": "numpy", "version": "2.5.3", "vulns": [None]}]},
        {
            "dependencies": [
                {"name": "numpy", "version": "2.5.3", "vulns": [{"id": "", "fix_versions": []}]}
            ]
        },
        {
            "dependencies": [
                {
                    "name": "numpy",
                    "version": "2.5.3",
                    "vulns": [{"id": "GHSA-example", "fix_versions": [1]}],
                }
            ]
        },
        {"dependencies": [{"name": "numpy", "version": "2.5.3", "vulns": []}] * 2},
    ],
)
def test_incomplete_skipped_duplicate_or_malformed_reports_fail(payload: object) -> None:
    """A successful process cannot turn invalid coverage into accepted evidence."""
    with pytest.raises(ValueError):
        validate_dependency_report(payload, {("numpy", "2.5.3")})


def test_real_cli_replaces_stale_green_report_when_inventory_is_missing(tmp_path: Path) -> None:
    """Run the real entry point without scanner mocks or external services."""
    packet = tmp_path / "packet"
    security = packet / "security"
    security.mkdir(parents=True)
    (security / "pip_audit.json").write_text('{"coverage_complete":true,"passed":true}')
    retained = security / "unrelated-evidence.txt"
    retained.write_text("keep\n")
    process = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/security_scan/python_dependency_audit.py"),
            "--repo-root",
            str(tmp_path / "missing-repository"),
            "--output-dir",
            str(packet),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads((security / "pip_audit.json").read_text())
    assert report["coverage_complete"] is False and report["passed"] is False
    assert report["profiles"] == [] and report["errors"]
    assert json.loads(process.stdout) == report
    assert retained.read_text() == "keep\n"
