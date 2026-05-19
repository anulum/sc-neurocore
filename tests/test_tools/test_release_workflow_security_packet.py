# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_release_workflow() -> dict:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert workflow_path.exists()
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def test_release_workflow_builds_security_packet_before_release() -> None:
    workflow = _load_release_workflow()
    steps = workflow["jobs"]["release"]["steps"]
    run_steps = [step["run"] for step in steps if isinstance(step, dict) and "run" in step]

    assert any("tools/security_scan/ci_security_packet.py" in step for step in run_steps)
    assert any(
        "--output-dir" in step and "security/ci-security-packet" in step for step in run_steps
    )
    assert any("--fail-on-missing-required" in step for step in run_steps)


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
