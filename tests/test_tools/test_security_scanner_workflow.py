# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Security Scanner Workflow CI Wiring

from __future__ import annotations

import re
from pathlib import Path

import yaml


def _load_workflow() -> dict:
    workflow_path = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "security-scanners.yml"
    )
    assert workflow_path.exists()
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def _workflow_events(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True, {}))


def test_security_scanner_workflow_triggers_and_scope() -> None:
    workflow = _load_workflow()
    on = _workflow_events(workflow)

    assert "workflow_dispatch" in on
    assert "pull_request" in on
    assert "push" in on

    push_paths = set(on["push"].get("paths", []))
    pull_request_paths = set(on["pull_request"].get("paths", []))
    for required in {
        ".github/workflows/security-scanners.yml",
        "security/**",
        "tools/security_scan/**",
        "requirements/**",
    }:
        assert required in push_paths
        assert required in pull_request_paths


def test_security_scanner_workflow_invokes_packet_builder() -> None:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    assert jobs

    scan_job = jobs["security-scanner-manifest"]
    steps = scan_job["steps"]

    run_steps = [step["run"] for step in steps if isinstance(step, dict) and "run" in step]
    assert any("tools/security_scan/ci_security_packet.py" in step for step in run_steps)
    assert any(
        "--output-dir" in step and "security/ci-security-packet" in step for step in run_steps
    )
    assert not any("--output security/ci-security-packet" in step for step in run_steps)


def test_security_scanner_workflow_checks_manifest_consistency_if_present() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    all_runs = [step["run"] for step in steps if isinstance(step, dict) and "run" in step]

    run_text = "\n".join(all_runs)
    for banned in {
        "pip-audit",
        "osv-scanner",
        "cargo-audit",
        "cargo-deny",
        "gitleaks",
        "semgrep",
        "trivy",
        "syft",
        "reuse",
        "actionlint",
        "pyright",
        "mypy",
        "ruff",
        "cargo fuzz",
    }:
        assert banned not in run_text

    manifest_validation = [
        step for step in all_runs if "security_scanner_manifest.py --validate" in step
    ]
    if manifest_validation:
        for command in manifest_validation:
            assert re.search(r"--validate\s+([^\s#]+)", command) is not None


def test_security_scanner_workflow_uploads_artifact() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["security-scanner-manifest"]["steps"]

    assert any(
        isinstance(step, dict)
        and step.get("uses", "").startswith("actions/upload-artifact@")
        and "security/ci-security-packet" in step.get("with", {}).get("path", "")
        for step in steps
    )
