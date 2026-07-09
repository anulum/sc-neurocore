# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optics optional-extra CI contract tests

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _repo_root() -> Path:
    """Return the repository root containing CI and docs."""

    return Path(__file__).resolve().parents[2]


def _load_ci_workflow() -> dict[str, Any]:
    """Load the main CI workflow through the YAML parser used by tests."""

    workflow_path = _repo_root() / ".github" / "workflows" / "ci.yml"
    return cast(dict[str, Any], yaml.safe_load(workflow_path.read_text(encoding="utf-8")))


def _job_run_text(job: dict[str, Any]) -> str:
    """Return all shell snippets from a GitHub Actions job."""

    return "\n".join(
        step["run"] for step in job["steps"] if isinstance(step, dict) and "run" in step
    )


def test_optics_extra_ci_job_installs_profile_and_runs_selector() -> None:
    """Ensure the gdsfactory-gated optics path has explicit CI coverage."""

    workflow = _load_ci_workflow()
    job = cast(dict[str, Any], workflow["jobs"]["test-optics-extra"])
    run_text = _job_run_text(job)

    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["PYTHONPATH"] == "src:."
    assert 'python -m pip install -e ".[dev,optics]"' in run_text
    assert "python -m pytest tests/test_optics -q -rs" in run_text


def test_optics_extra_ci_boundary_is_documented() -> None:
    """Keep the optional-dependency matrix synchronized with the CI lane."""

    matrix_text = (_repo_root() / "docs" / "guides" / "optional_dependency_matrix.md").read_text(
        encoding="utf-8"
    )
    install_profiles = (_repo_root() / "docs" / "guides" / "install_profiles.md").read_text(
        encoding="utf-8"
    )

    assert "test-optics-extra" in matrix_text
    assert ".[dev,optics]" in matrix_text
    assert "tests/test_optics -q -rs" in matrix_text
    assert 'pip install "sc-neurocore[optics]"' in install_profiles
