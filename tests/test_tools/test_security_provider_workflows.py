# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Security Provider Workflow Evidence Tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_workflow(name: str) -> dict[str, Any]:
    workflow_path = REPO_ROOT / ".github" / "workflows" / name
    assert workflow_path.exists()
    return cast(dict[str, Any], yaml.safe_load(workflow_path.read_text(encoding="utf-8")))


def _run_steps(workflow: dict[str, Any], job_name: str) -> str:
    job = workflow["jobs"][job_name]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    return "\n".join(
        step["run"] for step in steps if isinstance(step, dict) and isinstance(step.get("run"), str)
    )


def _uses_steps(workflow: dict[str, Any], job_name: str) -> list[dict[str, Any]]:
    job = workflow["jobs"][job_name]
    assert isinstance(job, dict)
    steps = job["steps"]
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict) and isinstance(step.get("uses"), str)]


def test_codeql_workflow_uploads_stable_release_evidence_artifact() -> None:
    workflow = _load_workflow("codeql.yml")
    steps = _uses_steps(workflow, "analyze")
    run_text = _run_steps(workflow, "analyze")

    analyze_steps = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("github/codeql-action/analyze@")
    ]
    assert len(analyze_steps) == 1
    assert analyze_steps[0].get("with", {}).get("output") == "security/codeql-results"
    assert "mkdir -p security/codeql-results" in run_text

    assert any(
        str(step.get("uses", "")).startswith("actions/upload-artifact@")
        and step.get("with", {}).get("name") == "codeql-results"
        and step.get("with", {}).get("path") == "security/codeql-results"
        for step in steps
    )


def test_scorecard_workflow_uploads_stable_release_evidence_artifact() -> None:
    workflow = _load_workflow("scorecard.yml")
    steps = _uses_steps(workflow, "analysis")
    run_text = _run_steps(workflow, "analysis")

    scorecard_steps = [
        step for step in steps if str(step.get("uses", "")).startswith("ossf/scorecard-action@")
    ]
    assert len(scorecard_steps) == 1
    assert scorecard_steps[0].get("with", {}).get("results_file") == (
        "security/scorecard-results.sarif"
    )
    assert scorecard_steps[0].get("with", {}).get("results_format") == "sarif"
    assert "mkdir -p security" in run_text

    assert any(
        str(step.get("uses", "")).startswith("actions/upload-artifact@")
        and step.get("with", {}).get("name") == "scorecard-results"
        and step.get("with", {}).get("path") == "security/scorecard-results.sarif"
        for step in steps
    )
    assert any(
        str(step.get("uses", "")).startswith("github/codeql-action/upload-sarif@")
        and step.get("with", {}).get("sarif_file") == "security/scorecard-results.sarif"
        for step in steps
    )


def test_release_artifacts_manifest_indexes_security_provider_evidence() -> None:
    manifest_path = REPO_ROOT / "security" / "release_artifacts_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {entry["id"]: entry["path"] for entry in payload["artifacts"]}

    assert artifact_paths["codeql_results"] == "security/codeql-results"
    assert artifact_paths["scorecard_results"] == "security/scorecard-results.sarif"
