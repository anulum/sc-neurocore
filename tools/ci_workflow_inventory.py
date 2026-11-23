# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — read CI jobs wherever their workflow lives

"""Read the jobs of the continuous-integration workflows by name.

Tests that hold a contract about a CI job read it through this module instead
of opening one workflow file, so the contract does not depend on which file
carries the job. ``CI_WORKFLOWS`` lists the workflow files that make up CI, in
order. Every job id must be declared in exactly one of them.

Examples
--------
>>> from tools.ci_workflow_inventory import ci_job
>>> steps = ci_job("lint")["steps"]  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]

CI_WORKFLOWS: tuple[str, ...] = (".github/workflows/ci.yml",)


class CiInventoryError(LookupError):
    """A CI job is absent, declared twice, or a CI workflow is malformed."""


@dataclass(frozen=True)
class CiJob:
    """One CI job and the workflow file that declares it.

    Attributes
    ----------
    name : str
        The job id under ``jobs``.
    workflow : pathlib.Path
        The workflow file that declares the job.
    body : dict
        The job's parsed mapping.
    """

    name: str
    workflow: Path
    body: dict[str, Any]


@lru_cache(maxsize=8)
def _inventory(root: Path) -> tuple[dict[str, CiJob], dict[Path, str], dict[Path, dict[str, Any]]]:
    jobs: dict[str, CiJob] = {}
    texts: dict[Path, str] = {}
    envs: dict[Path, dict[str, Any]] = {}
    for relative in CI_WORKFLOWS:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        texts[path] = text
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict) or not isinstance(loaded.get("jobs"), dict):
            raise CiInventoryError(f"{relative} has no jobs mapping")
        env = loaded.get("env", {})
        if not isinstance(env, dict):
            raise CiInventoryError(f"{relative}: the workflow env is not a mapping")
        envs[path] = cast(dict[str, Any], env)
        for name, body in cast(dict[str, Any], loaded["jobs"]).items():
            if not isinstance(body, dict):
                raise CiInventoryError(f"{relative}: job {name!r} is not a mapping")
            if name in jobs:
                first = jobs[name].workflow.relative_to(root)
                raise CiInventoryError(f"job {name!r} is declared in both {first} and {relative}")
            jobs[name] = CiJob(name=name, workflow=path, body=cast(dict[str, Any], body))
    return jobs, texts, envs


def ci_jobs(root: Path = ROOT) -> dict[str, CiJob]:
    """Return every CI job by id.

    Parameters
    ----------
    root : pathlib.Path, optional
        Repository root; the default is this checkout.

    Returns
    -------
    dict
        Job id to :class:`CiJob`, in declaration order.

    Raises
    ------
    CiInventoryError
        If a workflow has no ``jobs`` mapping, a job is not a mapping, or a
        job id is declared in two workflows.
    """
    return dict(_inventory(root.resolve())[0])


def ci_job(name: str, root: Path = ROOT) -> dict[str, Any]:
    """Return the parsed mapping of CI job ``name``.

    Parameters
    ----------
    name : str
        The job id.
    root : pathlib.Path, optional
        Repository root; the default is this checkout.

    Returns
    -------
    dict
        The job's mapping, as YAML parses it.

    Raises
    ------
    CiInventoryError
        If no CI workflow declares the job.
    """
    jobs = _inventory(root.resolve())[0]
    if name not in jobs:
        raise CiInventoryError(f"no CI workflow declares job {name!r}")
    return jobs[name].body


def ci_job_text(name: str, root: Path = ROOT) -> str:
    """Return the text of the workflow file that declares CI job ``name``.

    Parameters
    ----------
    name : str
        The job id.
    root : pathlib.Path, optional
        Repository root; the default is this checkout.

    Returns
    -------
    str
        The full text of the job's workflow file, for contracts that search
        its YAML.

    Raises
    ------
    CiInventoryError
        If no CI workflow declares the job.
    """
    jobs, texts, _envs = _inventory(root.resolve())
    if name not in jobs:
        raise CiInventoryError(f"no CI workflow declares job {name!r}")
    return texts[jobs[name].workflow]


def ci_jobs_text(*names: str, root: Path = ROOT) -> str:
    """Return the texts of the workflow files that declare the named CI jobs.

    Each workflow appears once, in ``CI_WORKFLOWS`` order, however many of the
    named jobs it declares, so a count over the result is not inflated.

    Parameters
    ----------
    *names : str
        The job ids; at least one.
    root : pathlib.Path, optional
        Repository root; the default is this checkout.

    Returns
    -------
    str
        The owning workflows' texts, joined by a newline.

    Raises
    ------
    CiInventoryError
        If no name is given, or no CI workflow declares one of the jobs.
    """
    if not names:
        raise CiInventoryError("name at least one CI job")
    jobs, texts, _envs = _inventory(root.resolve())
    missing = [name for name in names if name not in jobs]
    if missing:
        raise CiInventoryError(f"no CI workflow declares job(s) {missing}")
    owners = {jobs[name].workflow for name in names}
    return "\n".join(text for path, text in texts.items() if path in owners)


def ci_job_workflow_env(name: str, root: Path = ROOT) -> dict[str, Any]:
    """Return the workflow-level ``env`` of the workflow that declares job ``name``.

    Parameters
    ----------
    name : str
        The job id.
    root : pathlib.Path, optional
        Repository root; the default is this checkout.

    Returns
    -------
    dict
        That workflow's top-level ``env`` (empty when it declares none).

    Raises
    ------
    CiInventoryError
        If no CI workflow declares the job.
    """
    jobs, _texts, envs = _inventory(root.resolve())
    if name not in jobs:
        raise CiInventoryError(f"no CI workflow declares job {name!r}")
    return envs[jobs[name].workflow]


def ci_workflow_envs(root: Path = ROOT) -> dict[Path, dict[str, Any]]:
    """Return the workflow-level ``env`` of every CI workflow.

    A reusable workflow does not inherit its caller's ``env``, so a variable
    every CI job needs must appear in each of these mappings.

    Parameters
    ----------
    root : pathlib.Path, optional
        Repository root; the default is this checkout.

    Returns
    -------
    dict
        Workflow path to its top-level ``env`` (empty when it declares none).

    Raises
    ------
    CiInventoryError
        If a workflow's ``env`` is not a mapping.
    """
    return dict(_inventory(root.resolve())[2])


def ci_workflow_files(root: Path = ROOT) -> tuple[Path, ...]:
    """Return the workflow files that make up CI, in order.

    Parameters
    ----------
    root : pathlib.Path, optional
        Repository root; the default is this checkout.

    Returns
    -------
    tuple of pathlib.Path
        One path per entry of ``CI_WORKFLOWS``.
    """
    return tuple(root / relative for relative in CI_WORKFLOWS)
