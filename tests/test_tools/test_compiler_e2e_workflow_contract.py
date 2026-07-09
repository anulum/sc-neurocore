# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — compiler e2e workflow contract

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _repo_root() -> Path:
    """Return the repository root containing the compiler e2e workflow."""

    return Path(__file__).resolve().parents[2]


def _load_workflow() -> dict[str, Any]:
    """Load the compiler e2e workflow through the same YAML surface CI uses."""

    workflow = _repo_root() / ".github" / "workflows" / "compiler-e2e.yml"
    return cast(dict[str, Any], yaml.safe_load(workflow.read_text(encoding="utf-8")))


def _workflow_events(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return GitHub event configuration while preserving YAML's boolean key quirk."""

    raw_events = workflow.get("on")
    if raw_events is None:
        raw_events = cast(dict[Any, Any], workflow).get(True, {})
    return cast(dict[str, Any], raw_events)


def _run_text(workflow: dict[str, Any]) -> str:
    """Return all shell snippets from the compiler e2e job."""

    steps = workflow["jobs"]["e2e"]["steps"]
    return "\n".join(step["run"] for step in steps if isinstance(step, dict) and "run" in step)


def test_compiler_e2e_workflow_runs_on_relevant_pr_paths() -> None:
    """Keep compiler and HDL generator edits wired to the e2e selector."""

    workflow = _load_workflow()
    events = _workflow_events(workflow)
    paths = events["pull_request"]["paths"]

    assert workflow["permissions"] == {}
    assert workflow["jobs"]["e2e"]["permissions"] == {"contents": "read"}
    assert "workflow_dispatch" in events
    assert "push" not in events
    assert "src/sc_neurocore/compiler/**" in paths
    assert "src/sc_neurocore/hdl_gen/**" in paths
    assert "tests/e2e/**" in paths
    assert ".github/workflows/compiler-e2e.yml" in paths


def test_compiler_e2e_workflow_uses_narrow_selector() -> None:
    """Keep this PR lane focused on the e2e corpus instead of the full suite."""

    run_text = _run_text(_load_workflow())

    assert "python tools/ci_install_dev.py" in run_text
    assert "python -m pytest tests/e2e/ -m e2e -q" in run_text
    assert "python -m pytest tests/ -q" not in run_text
    assert "--cov" not in run_text
