# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tests for the CI job reader

"""The CI job reader finds each job in the workflow that declares it, and refuses ambiguity."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import ci_workflow_inventory as inventory
from tools.ci_workflow_inventory import (
    CI_WORKFLOWS,
    CiInventoryError,
    ci_job,
    ci_job_text,
    ci_job_workflow_env,
    ci_jobs,
    ci_jobs_text,
    ci_workflow_envs,
    ci_workflow_files,
)


def _root(tmp_path: Path, workflows: dict[str, str]) -> Path:
    """Write ``workflows`` (relative path to YAML text) under a fresh repository root."""
    for relative, text in workflows.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_the_live_ci_declares_the_jobs_the_contract_tests_read() -> None:
    """Every job the migrated contract tests name is found, with its own workflow text."""
    jobs = ci_jobs()
    for name in (
        "lint",
        "test",
        "frontend",
        "coverage-union",
        "engine-semver",
        "rust-documentation",
    ):
        assert name in jobs
        assert jobs[name].workflow in ci_workflow_files()
        assert f"\n  {name}:" in ci_job_text(name)
    assert ci_job("lint")["steps"]
    assert ci_job_workflow_env("test")["SCN_CI_SHARD_COUNT"]
    assert set(ci_workflow_envs()) == set(ci_workflow_files())


def test_jobs_are_read_from_the_workflow_that_declares_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With two CI workflows, each job, text and env comes from its own file."""
    monkeypatch.setattr(inventory, "CI_WORKFLOWS", ("a.yml", "b.yml"))
    root = _root(
        tmp_path,
        {
            "a.yml": "env:\n  X: '1'\njobs:\n  first:\n    steps: [{run: one}]\n",
            "b.yml": "jobs:\n  second:\n    steps: [{run: two}]\n  third:\n    steps: []\n",
        },
    )
    assert list(ci_jobs(root)) == ["first", "second", "third"]
    assert ci_job("second", root)["steps"] == [{"run": "two"}]
    assert "second:" in ci_job_text("second", root) and "first:" not in ci_job_text("second", root)
    assert ci_job_workflow_env("first", root) == {"X": "1"}
    assert ci_job_workflow_env("third", root) == {}
    assert ci_workflow_envs(root) == {root / "a.yml": {"X": "1"}, root / "b.yml": {}}
    assert ci_workflow_files(root) == (root / "a.yml", root / "b.yml")
    joined = ci_jobs_text("second", "third", "first", root=root)
    assert joined.count("second:") == 1 and joined.index("first:") < joined.index("second:")


@pytest.mark.parametrize(
    ("workflows", "message"),
    [
        ({"a.yml": "name: CI\n", "b.yml": "jobs: {}\n"}, "a.yml has no jobs mapping"),
        (
            {"a.yml": "jobs: {}\nenv: [1]\n", "b.yml": "jobs: {}\n"},
            "the workflow env is not a mapping",
        ),
        ({"a.yml": "jobs:\n  first: [1]\n", "b.yml": "jobs: {}\n"}, "job 'first' is not a mapping"),
        (
            {"a.yml": "jobs:\n  first: {}\n", "b.yml": "jobs:\n  first: {}\n"},
            "job 'first' is declared in both a.yml and b.yml",
        ),
    ],
)
def test_malformed_or_ambiguous_workflows_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflows: dict[str, str], message: str
) -> None:
    """A missing jobs mapping, a non-mapping env or job, or a duplicated id is refused."""
    monkeypatch.setattr(inventory, "CI_WORKFLOWS", ("a.yml", "b.yml"))
    root = _root(tmp_path, workflows)
    with pytest.raises(CiInventoryError, match=message):
        ci_jobs(root)


def test_an_unknown_job_is_refused_by_every_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No reader returns a default for a job that no CI workflow declares."""
    monkeypatch.setattr(inventory, "CI_WORKFLOWS", ("a.yml",))
    root = _root(tmp_path, {"a.yml": "jobs:\n  first: {}\n"})
    for reader in (ci_job, ci_job_text, ci_job_workflow_env):
        with pytest.raises(CiInventoryError, match="no CI workflow declares job 'absent'"):
            reader("absent", root)
    with pytest.raises(CiInventoryError, match=r"job\(s\) \['absent'\]"):
        ci_jobs_text("first", "absent", root=root)
    with pytest.raises(CiInventoryError, match="name at least one CI job"):
        ci_jobs_text(root=root)


def test_the_ci_workflow_list_is_ordered_and_names_existing_files() -> None:
    """Every listed CI workflow exists in the checkout, once."""
    assert len(set(CI_WORKFLOWS)) == len(CI_WORKFLOWS)
    assert all(path.is_file() for path in ci_workflow_files())
