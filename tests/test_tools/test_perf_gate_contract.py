# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Performance-gate workflow contract tests

from __future__ import annotations

import ast
import shlex
from pathlib import Path
from typing import Any, cast

import yaml


def _repo_root() -> Path:
    """Return the repository root containing the perf-gated tests."""

    return Path(__file__).resolve().parents[2]


def _load_benchmark_workflow() -> dict[str, Any]:
    """Load the benchmark workflow through the same YAML surface CI uses."""

    workflow_path = _repo_root() / ".github" / "workflows" / "benchmark.yml"
    return cast(dict[str, Any], yaml.safe_load(workflow_path.read_text(encoding="utf-8")))


def _workflow_events(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return GitHub event configuration while preserving YAML's boolean key quirk."""

    raw_events = workflow.get("on")
    if raw_events is None:
        raw_events = cast(dict[Any, Any], workflow).get(True, {})
    return cast(dict[str, Any], raw_events)


def _perf_gated_test_files() -> list[str]:
    """Discover tests that opt into wall-clock assertions with SC_NEUROCORE_PERF."""

    tests_root = _repo_root() / "tests"
    files = []
    for path in sorted(tests_root.rglob("test_*.py")):
        relative = path.relative_to(_repo_root()).as_posix()
        text = path.read_text(encoding="utf-8")
        if "SC_NEUROCORE_PERF" in text and _has_pytest_skipif_decorator(text):
            files.append(relative)
    return files


def _has_pytest_skipif_decorator(source: str) -> bool:
    """Return whether source contains a real pytest skipif decorator."""

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if any(_is_pytest_skipif(decorator) for decorator in node.decorator_list):
            return True
    return False


def _is_pytest_skipif(decorator: ast.expr) -> bool:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return (
        isinstance(decorator, ast.Attribute)
        and decorator.attr == "skipif"
        and isinstance(decorator.value, ast.Attribute)
        and decorator.value.attr == "mark"
        and isinstance(decorator.value.value, ast.Name)
        and decorator.value.value.id == "pytest"
    )


def test_perf_gated_tests_are_scheduled_and_documented() -> None:
    """Ensure every perf-gated pytest file is covered by CI and public docs."""

    workflow = _load_benchmark_workflow()
    events = _workflow_events(workflow)
    perf_job = workflow["jobs"]["perf-gated-pytest"]
    run_text = "\n".join(
        step["run"] for step in perf_job["steps"] if isinstance(step, dict) and "run" in step
    )
    docs_text = (_repo_root() / "docs" / "guides" / "performance_gates.md").read_text(
        encoding="utf-8"
    )
    perf_files = _perf_gated_test_files()

    assert perf_files
    assert "schedule" in events
    assert (
        perf_job["if"]
        == "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    )
    assert perf_job["env"]["SC_NEUROCORE_PERF"] == "1"
    assert perf_job["env"]["PYTHONPATH"] == "src:."
    assert "SC_NEUROCORE_PERF=1" in docs_text

    for perf_file in perf_files:
        assert perf_file in run_text
        assert perf_file in docs_text


def test_perf_gated_selector_stays_narrow() -> None:
    """Keep the scheduled perf lane separate from full-suite local policy."""

    workflow = _load_benchmark_workflow()
    run_text = "\n".join(
        step["run"]
        for step in workflow["jobs"]["perf-gated-pytest"]["steps"]
        if isinstance(step, dict) and "run" in step
    )

    assert "python -m pytest tests" not in run_text
    assert "--cov" not in run_text
    assert "benchmarks/benchmark_suite.py --full" not in run_text


def test_perf_gated_selector_paths_exist() -> None:
    """Reject stale test paths before the scheduled workflow reaches pytest."""

    workflow = _load_benchmark_workflow()
    run_text = "\n".join(
        step["run"]
        for step in workflow["jobs"]["perf-gated-pytest"]["steps"]
        if isinstance(step, dict) and "run" in step
    )
    selected_paths = sorted(
        token
        for token in shlex.split(run_text)
        if token.startswith("tests/") and token.endswith(".py")
    )
    missing_paths = [path for path in selected_paths if not (_repo_root() / path).is_file()]

    assert selected_paths
    assert not missing_paths, f"perf-gated pytest selector has stale paths: {missing_paths}"
