# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Whole native Go MAT CI promotion contract

"""Bind the native statement gate to its actual same-job producers and maintained source policy."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import cast

import yaml

ROOT = Path(__file__).resolve().parents[1]


def mapping(value: object) -> dict[str, object]:
    """Require mappings with string keys before interpreting workflow structure."""
    assert isinstance(value, dict) and all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


def read_jobs() -> dict[str, object]:
    """Read actual workflow jobs without interpreting YAML's event key as a job field."""
    workflow: object = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert isinstance(workflow, dict)
    return mapping(workflow["jobs"])


def read_test_job() -> dict[str, object]:
    """Read the maintained behavior matrix supplying the native gate's real images."""
    return mapping(read_jobs()["test"])


def job_steps(job: dict[str, object]) -> list[dict[str, object]]:
    """Keep every real step in order, including the unchanged native producers."""
    raw = job["steps"]
    assert isinstance(raw, list)
    return [mapping(row) for row in raw]


def test_native_gate_follows_same_job_producers_and_retained_full_suite_results() -> None:
    """One existing coverage shard receives real native input and preserves ordinary CI failure."""
    job = read_test_job()
    steps = job_steps(job)
    names = [step.get("name") for step in steps]
    gate_name = "Complete native Go MAT statement gate"
    assert names.count(gate_name) == 1
    index = names.index(gate_name)
    assert names.index("Build accelerator backends (Go + Mojo)") < index
    assert names.index("Enforce Mojo ISA baseline (no AVX-512 drift)") < index
    assert names.index("Test + coverage") < index
    staging = names.index("Stage this shard's coverage data")
    preservation = names.index("Preserve this shard's coverage data")
    assert index < staging < preservation
    assert names[-2:] == [
        "Stage this shard's coverage data",
        "Preserve this shard's coverage data",
    ]
    assert steps[staging]["if"] == steps[preservation]["if"] == "matrix.coverage"
    assert all(
        str(step.get("uses", "")).startswith("actions/upload-artifact@")
        for step in steps[index + 1 :]
        if step.get("name") != "Stage this shard's coverage data"
    )
    gate = steps[index]
    assert gate["if"] == "matrix.coverage && matrix.shard-index == 8"
    assert not gate.get("continue-on-error", False)
    run = str(gate["run"])
    assert "tools/go_mat_coverage.py" in run and "subprocess.run(argv, check=True)" in run
    assert "--seconds" in run and "--jobs" in run and "--stop-seconds" in run
    assert not any(
        option in run for option in ("--deselect", "-k ", "continue-on-error", "|| true")
    )


def test_budget_uses_live_profile_and_the_actual_enclosing_timeout_expression() -> None:
    """The proposed policy fractions do not duplicate fixed host CPU or runtime limits."""
    job = read_test_job()
    steps = job_steps(job)
    gate = next(
        step for step in steps if step.get("name") == "Complete native Go MAT statement gate"
    )
    assert mapping(gate["env"])["SCN_NATIVE_GO_JOB_TIMEOUT_MINUTES"] == job["timeout-minutes"]
    run = str(gate["run"])
    assert "from tools.go_coverage_budget import derive_profile" in run
    assert "runtime_fraction=1 / 4, stop_fraction=1 / 200" in run
    assert "SCN_NATIVE_GO_JOB_STARTED_EPOCH" in run
    assert steps[0]["name"] == "Record native Go gate enclosing-job clock"
    assert steps[0]["if"] == gate["if"]
    assert "SCN_NATIVE_GO_JOB_STARTED_EPOCH=" in str(steps[0]["run"])
    assert "service-side job-start timestamp" in str(steps[0]["run"])


def test_native_evidence_is_preserved_even_when_the_gate_fails() -> None:
    """A failed native producer keeps raw artifacts and its actual command/process receipts."""
    steps = job_steps(read_test_job())
    evidence = [
        step for step in steps if step.get("name") == "Preserve native Go MAT gate evidence"
    ]
    assert len(evidence) == 1
    assert evidence[0]["if"] == (
        "always() && matrix.coverage && matrix.shard-index == 8 "
        "&& steps.native_go_mat.outputs.profile_written == 'true'"
    )
    settings = mapping(evidence[0]["with"])
    assert settings["include-hidden-files"] is True and settings["if-no-files-found"] == "error"
    for suffix in (
        "profile.json",
        "terminal.json",
        "artifacts",
        "commands",
        "process-counters",
        "*.xml",
        "*.json",
    ):
        assert suffix in str(settings["path"])


def test_all_native_gate_python_sources_are_enrolled_without_lowering_doc_floor() -> None:
    """Every introduced Python surface participates in the existing source policy."""
    policy = tomllib.loads((ROOT / "docs/docstring_policy.toml").read_text())
    enrolled = {entry["path"] for entry in policy["file"]}
    required = {str(path.relative_to(ROOT)) for path in (ROOT / "tools").glob("go*coverage*.py")}
    required.update(
        str(path.relative_to(ROOT))
        for path in (ROOT / "tests/native_go_coverage_contracts").glob("*.py")
    )
    required.update(
        {
            "tests/native_go_coverage_consumer.py",
            "tests/native_go_coverage_platform.py",
            "tests/test_go_coverage_ci_contract.py",
            "tests/test_tools/test_go_coverage_profiles.py",
        }
    )
    assert len(required) >= 25 and required.issubset(enrolled)
    assert policy["quality"]["min_docstring_chars"] >= 20
    assert policy["quality"]["expected_file_count"] == len(policy["file"])
    assert len(enrolled) == len(policy["file"])
    assert "development/native_go_coverage.md" in (ROOT / "mkdocs.yml").read_text()


def test_actual_windows_refusal_uses_its_own_job_and_remaining_timeout() -> None:
    """An independent Windows leg preserves genuine refusal evidence with a bounded child."""
    jobs = read_jobs()
    windows = mapping(jobs["native-go-windows-refusal"])
    assert windows["runs-on"] == "windows-latest"
    assert "strategy" not in windows and "needs" not in windows
    steps = job_steps(windows)
    assert steps[0]["id"] == "native_go_job_clock"
    native = next(step for step in steps if step.get("id") == "native_go_platform")
    environment = mapping(native["env"])
    ceiling = int(str(environment["SCN_NATIVE_GO_PLATFORM_JOB_TIMEOUT_MINUTES"]))
    reserve = int(str(environment["SCN_NATIVE_GO_PLATFORM_RESERVE_MINUTES"]))
    assert ceiling == windows["timeout-minutes"] and 0 < reserve < ceiling
    assert "steps.native_go_job_clock.outputs.started_ms" in str(environment)
    run = str(native["run"])
    assert "ceiling - reserve - elapsed" in run and "time.time()" in run
    assert "timeout=seconds" in run and '"--job-seconds", str(seconds)' in run
    assert "tests/native_go_coverage_platform.py" in run and "--python-coverage-config" in run
    assert '"native-go-mat-python-windows"' in run and "shd-platform-coverage" not in run
    assert "actual-enclosing-job-budget-and-terminal.json" in run
    assert not native.get("continue-on-error", False)
    assert any(
        "--require-hashes -r requirements/ci-dev.txt" in str(step.get("run", "")) for step in steps
    )
    assert any(
        mapping(step.get("with", {})).get("name") == "native-go-mat-python-windows"
        and mapping(step.get("with", {})).get("path")
        == "${{ runner.temp }}/native-go-mat-python-windows"
        for step in steps
    )
    union = mapping(jobs["coverage-union"])
    needs = union["needs"]
    assert isinstance(needs, list)
    assert {"test", "shd-non-linux-refusal", "native-go-windows-refusal"} <= set(needs)
    guard = next(
        step
        for step in job_steps(union)
        if step.get("name") == "Require successful native Go Windows refusal"
    )
    assert mapping(guard["env"])["GO_WINDOWS_RESULT"] == (
        "${{ needs.native-go-windows-refusal.result }}"
    )
    assert 'test "$GO_WINDOWS_RESULT" = success' in str(guard["run"])


def test_own_python_union_requires_both_real_platforms_and_every_statement_and_branch() -> None:
    """Counter qualification remains independent of the earlier project-wide coverage result."""
    union = mapping(read_jobs()["coverage-union"])
    steps = job_steps(union)
    names = {step.get("name"): step for step in steps}
    measured = names["Complete native Go gate Python statement and branch union"]
    run = str(measured["run"])
    assert 'frozenset({"linux", "win32"})' in run
    assert "union_config" in run and "report_contract.py" in run
    assert "--append" in run and "--fail-under=100" in run
    assert "--fail-under=0" not in run and "--omit" not in run
    for name in (
        "Retrieve native Go gate Linux Python counters",
        "Retrieve actual native Go gate Windows Python counters",
        "Complete native Go gate Python statement and branch union",
    ):
        condition = str(names[name]["if"])
        assert (
            "!cancelled()" in condition and "native_counter_tools.outcome == 'success'" in condition
        )
        assert "needs.test.result == 'success'" in condition
        assert "needs.native-go-windows-refusal.result == 'success'" in condition
