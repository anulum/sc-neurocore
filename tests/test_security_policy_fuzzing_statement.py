# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the security policy states the fuzzing that exists

"""SECURITY.md describes the fuzzing the repository actually has.

The policy once said there was no fuzzing harness, after ``cargo-fuzz`` targets, a scheduled
fuzz workflow and Python ``hypothesis`` fuzz suites had all been added. These checks tie each
statement to its source: the target names to the two fuzz manifests, the schedule to the
workflow, the claim that the ``fuzz/`` targets are not run to every workflow, and the Python
suites to the test tree.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY = (ROOT / "SECURITY.md").read_text(encoding="utf-8")


def _targets(manifest: str) -> list[str]:
    """Return the cargo-fuzz binary names of a fuzz manifest."""
    data = tomllib.loads((ROOT / manifest).read_text(encoding="utf-8"))
    return [cast(str, entry["name"]) for entry in data["bin"]]


def _workflow_files() -> list[Path]:
    """Return every workflow file, ``.yml`` and ``.yaml``."""
    directory = ROOT / ".github/workflows"
    return sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])


def _load(path: Path) -> dict[Any, Any]:
    """Load one workflow; YAML 1.1 reads its bare ``on`` key as the boolean True."""
    return cast(dict[Any, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    """Return the trigger mapping of a loaded workflow."""
    return cast(dict[str, Any], workflow.get("on", workflow.get(True)))


def _steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    """Return every step of every job of a loaded workflow."""
    jobs = cast(dict[str, dict[str, Any]], workflow.get("jobs", {}))
    return [
        step for job in jobs.values() for step in cast(list[dict[str, Any]], job.get("steps", []))
    ]


def _limitation() -> str:
    """Return the fuzzing bullet of the Known Limitations section."""
    section = POLICY.split("## Known Limitations", 1)[1].split("\n## ", 1)[0]
    (bullet,) = [item for item in section.split("\n- ") if "Fuzzing is scheduled" in item]
    return " ".join(bullet.split())


def test_the_policy_no_longer_says_there_is_no_fuzzing() -> None:
    """The retired claim that no fuzzing harness exists is gone."""
    assert "No fuzzing harness yet" not in POLICY
    assert "dedicated fuzzing has not" not in POLICY


def test_the_limitation_names_every_scheduled_and_unscheduled_target() -> None:
    """The bullet names each engine target and each older target by its manifest name."""
    bullet = _limitation()
    for name in _targets("engine/fuzz/Cargo.toml") + _targets("fuzz/Cargo.toml"):
        assert f"`{name}`" in bullet, name
    assert "`engine/fuzz/`" in bullet and "`fuzz/`" in bullet


def test_the_stated_schedule_is_the_workflow_schedule() -> None:
    """The workflow runs every Monday and on demand, in the engine directory, as the policy says."""
    workflow = cast(
        dict[Any, Any], yaml.safe_load((ROOT / ".github/workflows/fuzz.yml").read_text())
    )
    # YAML 1.1 reads the bare key ``on`` as the boolean True.
    triggers = cast(dict[str, Any], workflow.get("on", workflow.get(True)))
    (schedule,) = cast(list[dict[str, str]], triggers["schedule"])
    assert schedule["cron"].split()[4] == "1"
    assert "workflow_dispatch" in triggers
    bullet = _limitation()
    assert "every Monday and on demand" in bullet
    assert "`.github/workflows/fuzz.yml`" in bullet
    fuzz_steps = [
        step
        for job in cast(dict[str, dict[str, Any]], workflow["jobs"]).values()
        for step in cast(list[dict[str, Any]], job.get("steps", []))
        if "fuzz run" in str(step.get("run", ""))
    ]
    assert fuzz_steps and all(step.get("working-directory") == "engine" for step in fuzz_steps)


def test_no_workflow_runs_the_older_fuzz_targets() -> None:
    """No workflow step runs cargo-fuzz from the root ``fuzz/`` directory, as the policy states."""
    older = _targets("fuzz/Cargo.toml")
    runners = []
    for path in _workflow_files():
        workflow = cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))
        for job in cast(dict[str, dict[str, Any]], workflow.get("jobs", {})).values():
            for step in cast(list[dict[str, Any]], job.get("steps", [])):
                run = str(step.get("run", ""))
                if "fuzz" in run and (
                    step.get("working-directory") in ("fuzz", "./fuzz")
                    or any(name in run for name in older)
                ):
                    runners.append(path.name)
    assert runners == []
    assert "not run by any workflow" in _limitation()


def test_the_python_fuzz_suites_the_policy_cites_exist() -> None:
    """The policy cites ``tests/test_fuzz_*.py``, and such suites exist and use hypothesis."""
    suites = sorted((ROOT / "tests").glob("test_fuzz_*.py"))
    assert suites
    assert all("hypothesis" in suite.read_text(encoding="utf-8") for suite in suites)
    assert "`tests/test_fuzz_*.py`" in _limitation()


def test_fuzzing_is_not_a_merge_gate() -> None:
    """The fuzz workflow runs only on a schedule and on demand, so it never gates a pull request."""
    triggers = _triggers(_load(ROOT / ".github/workflows/fuzz.yml"))
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    bullet = _limitation()
    assert "not a merge gate" in bullet
    assert "fails that workflow, not a pull request" in bullet


def test_no_fuzz_corpus_is_kept_between_runs() -> None:
    """No fuzz workflow step uploads or caches a corpus, and the Rust cache names none."""
    kept = []
    for step in _steps(_load(ROOT / ".github/workflows/fuzz.yml")):
        uses = str(step.get("uses", ""))
        stored = " ".join(
            str(value) for value in cast(dict[str, Any], step.get("with", {})).values()
        )
        keeps = "upload-artifact" in uses or "actions/cache" in uses or "rust-cache" in uses
        if keeps and "corpus" in stored:
            kept.append(uses)
    assert kept == []
    assert "Corpora are not kept between runs" in _limitation()


def test_the_rust_proptest_suites_the_policy_cites_run_in_ci() -> None:
    """``engine/tests/prop_*.rs`` are proptest suites tested by a push and pull-request workflow."""
    suites = sorted((ROOT / "engine/tests").glob("prop_*.rs"))
    assert suites
    assert all("proptest!" in suite.read_text(encoding="utf-8") for suite in suites)
    runners = [
        path.name
        for path in _workflow_files()
        if {"push", "pull_request"} <= set(_triggers(_load(path)))
        and any(
            "cargo test" in str(step.get("run", ""))
            and "--manifest-path engine/Cargo.toml" in str(step.get("run", ""))
            for step in _steps(_load(path))
        )
    ]
    assert runners
    bullet = _limitation()
    assert "`proptest`" in bullet and "`engine/tests/prop_*.rs`" in bullet
