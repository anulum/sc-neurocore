# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_release_workflow() -> dict[str, Any]:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert workflow_path.exists()
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return cast(dict[str, Any], workflow)


def test_release_workflow_builds_security_packet_before_release() -> None:
    workflow = _load_release_workflow()
    steps = workflow["jobs"]["release"]["steps"]
    run_steps = [step["run"] for step in steps if isinstance(step, dict) and "run" in step]

    assert any("tools/security_scan/release_security_sweep.py" in step for step in run_steps)
    assert any(
        "--output-dir" in step and "security/ci-security-packet" in step for step in run_steps
    )
    assert any("--include-fuzz" in step for step in run_steps)


def test_release_workflow_installs_real_release_sweep_tools() -> None:
    workflow = _load_release_workflow()
    steps = workflow["jobs"]["release"]["steps"]
    run_text = "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)

    assert (
        "python -m pip install --require-hashes -r requirements/security-scanners.txt" in run_text
    )
    assert "python -m pip install --require-hashes -r requirements/semgrep.txt" in run_text
    assert "python -m pip install --require-hashes -r requirements/fuzz.txt" in run_text
    assert "go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.12" in run_text
    assert "go install github.com/google/osv-scanner/v2/cmd/osv-scanner@v2.3.8" in run_text
    assert "go install github.com/anchore/syft/cmd/syft@v1.20.0" in run_text
    assert "cargo install cargo-audit --version 0.22.1 --locked" in run_text
    assert "cargo install cargo-deny --version 0.19.6 --locked" in run_text
    assert "cargo +stable install cargo-fuzz --version 0.13.1 --locked" in run_text


def test_release_workflow_attaches_cve_status_to_github_release() -> None:
    workflow = _load_release_workflow()
    steps = workflow["jobs"]["release"]["steps"]
    release_steps = [
        step
        for step in steps
        if isinstance(step, dict)
        and str(step.get("uses", "")).startswith("softprops/action-gh-release@")
    ]

    assert len(release_steps) == 1
    files = release_steps[0]["with"]["files"]
    assert "security/ci-security-packet/release_security_artifact_index.json" in files
    assert "security/ci-security-packet/security/release_security_sweep_summary.json" in files
