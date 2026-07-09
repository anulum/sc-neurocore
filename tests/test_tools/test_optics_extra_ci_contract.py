# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optional-extra CI contract tests

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, cast

import yaml


class OptionalCiLane(NamedTuple):
    """Expected optional-extra CI matrix lane."""

    name: str
    install_extra: str
    apt_packages: str
    test_selector: str


OPTIONAL_CI_LANES = (
    OptionalCiLane(
        name="annealing",
        install_extra=".[dev,annealing]",
        apt_packages="",
        test_selector="tests/test_bridges/test_quantum_annealing_neal_parity.py",
    ),
    OptionalCiLane(
        name="onnx-protobuf",
        install_extra=".[dev]",
        apt_packages="",
        test_selector="tests/test_export/test_onnx_exporter.py",
    ),
    OptionalCiLane(
        name="mpi",
        install_extra=".[dev,mpi]",
        apt_packages="libopenmpi-dev openmpi-bin",
        test_selector="tests/test_mpi_runner_real.py",
    ),
)


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


def _optional_matrix_rows() -> dict[str, dict[str, str]]:
    """Return the optional-extra CI matrix rows keyed by lane name."""

    workflow = _load_ci_workflow()
    job = cast(dict[str, Any], workflow["jobs"]["test-optional-extras"])
    include = cast(list[dict[str, str]], job["strategy"]["matrix"]["include"])
    return {row["name"]: row for row in include}


def test_optics_extra_ci_job_installs_profile_and_runs_selector() -> None:
    """Ensure the gdsfactory-gated optics path has explicit CI coverage."""

    workflow = _load_ci_workflow()
    job = cast(dict[str, Any], workflow["jobs"]["test-optics-extra"])
    run_text = _job_run_text(job)

    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["PYTHONPATH"] == "src:."
    assert 'python -m pip install -e ".[dev,optics]"' in run_text
    assert "python -m pytest tests/test_optics -q -rs" in run_text


def test_optional_extra_ci_matrix_runs_import_skipped_selectors() -> None:
    """Ensure import-skipped optional surfaces have explicit CI lanes."""

    workflow = _load_ci_workflow()
    job = cast(dict[str, Any], workflow["jobs"]["test-optional-extras"])
    run_text = _job_run_text(job)
    rows = _optional_matrix_rows()

    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["PYTHONPATH"] == "src:."
    assert job["strategy"]["fail-fast"] is False
    assert 'python -m pip install -e "${{ matrix.install_extra }}"' in run_text
    assert "python -m pytest ${{ matrix.test_selector }} -q -rs" in run_text
    assert "sudo apt-get install -y -qq ${{ matrix.apt_packages }}" in run_text

    for lane in OPTIONAL_CI_LANES:
        assert rows[lane.name]["install_extra"] == lane.install_extra
        assert rows[lane.name]["apt_packages"] == lane.apt_packages
        assert rows[lane.name]["test_selector"] == lane.test_selector


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


def test_optional_extra_ci_matrix_is_documented() -> None:
    """Keep optional-extra CI lanes synchronized with public dependency docs."""

    matrix_text = (_repo_root() / "docs" / "guides" / "optional_dependency_matrix.md").read_text(
        encoding="utf-8"
    )

    for lane in OPTIONAL_CI_LANES:
        assert f"test-optional-extras ({lane.name})" in matrix_text
        assert lane.install_extra in matrix_text
        assert f"{lane.test_selector} -q -rs" in matrix_text
