# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from tests.performance_guard import (
    STRICT_THROUGHPUT_ENV,
    assert_load_tolerant_throughput,
    assert_speedup_guard,
)


def test_load_tolerant_guard_uses_capped_one_percent_smoke_floor(monkeypatch) -> None:
    monkeypatch.delenv(STRICT_THROUGHPUT_ENV, raising=False)

    assert_load_tolerant_throughput(
        label="shared host",
        observed_per_second=101.0,
        strict_minimum_per_second=50_000.0,
    )

    with pytest.raises(AssertionError, match="smoke guard failed"):
        assert_load_tolerant_throughput(
            label="shared host",
            observed_per_second=100.0,
            strict_minimum_per_second=50_000.0,
        )


def test_load_tolerant_guard_preserves_strict_threshold(monkeypatch) -> None:
    monkeypatch.setenv(STRICT_THROUGHPUT_ENV, "true")

    with pytest.raises(AssertionError, match="throughput regressed"):
        assert_load_tolerant_throughput(
            label="isolated core",
            observed_per_second=49_999.0,
            strict_minimum_per_second=50_000.0,
        )

    assert_load_tolerant_throughput(
        label="isolated core",
        observed_per_second=50_001.0,
        strict_minimum_per_second=50_000.0,
    )


@pytest.mark.parametrize("minimum", [0.0, -1.0, math.inf, math.nan])
def test_load_tolerant_guard_rejects_invalid_strict_minimum(minimum: float) -> None:
    with pytest.raises(AssertionError, match="strict throughput minimum"):
        assert_load_tolerant_throughput(
            label="invalid threshold",
            observed_per_second=1.0,
            strict_minimum_per_second=minimum,
        )


def test_speedup_guard_separates_smoke_and_strict_modes(monkeypatch) -> None:
    monkeypatch.delenv(STRICT_THROUGHPUT_ENV, raising=False)
    assert_speedup_guard(
        label="backend",
        baseline_seconds=1.0,
        candidate_seconds=2.0,
        strict_minimum_speedup=2.0,
        smoke_minimum_speedup=0.25,
    )
    with pytest.raises(AssertionError, match="speedup smoke guard failed"):
        assert_speedup_guard(
            label="backend",
            baseline_seconds=1.0,
            candidate_seconds=5.0,
            strict_minimum_speedup=2.0,
            smoke_minimum_speedup=0.25,
        )

    monkeypatch.setenv(STRICT_THROUGHPUT_ENV, "1")
    with pytest.raises(AssertionError, match="speedup regressed"):
        assert_speedup_guard(
            label="backend",
            baseline_seconds=1.0,
            candidate_seconds=0.5,
            strict_minimum_speedup=2.0,
            smoke_minimum_speedup=0.25,
        )
    assert_speedup_guard(
        label="backend",
        baseline_seconds=1.0,
        candidate_seconds=0.4,
        strict_minimum_speedup=2.0,
        smoke_minimum_speedup=0.25,
    )


@pytest.mark.parametrize("baseline,candidate", [(0.0, 1.0), (1.0, math.inf)])
def test_speedup_guard_rejects_invalid_timings(baseline: float, candidate: float) -> None:
    with pytest.raises(AssertionError, match="timing is not finite positive"):
        assert_speedup_guard(
            label="backend",
            baseline_seconds=baseline,
            candidate_seconds=candidate,
            strict_minimum_speedup=2.0,
            smoke_minimum_speedup=0.25,
        )


def test_model_performance_modules_do_not_assert_raw_clock_values() -> None:
    clock_names = {
        "elapsed",
        "rate",
        "throughput",
        "steps_per_s",
        "nsteps_per_s",
        "seconds_per_step",
        "best_seconds_per_step",
    }
    violations = []

    for path in sorted(Path(__file__).parent.glob("test_model_*performance.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for assertion in (node for node in ast.walk(tree) if isinstance(node, ast.Assert)):
            referenced_names = {
                node.id for node in ast.walk(assertion.test) if isinstance(node, ast.Name)
            }
            reads_clock_directly = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"perf_counter", "process_time", "monotonic", "time"}
                for node in ast.walk(assertion.test)
            )
            derived_name = any(
                name in clock_names
                or "throughput" in name
                or name.endswith("_per_s")
                or name.endswith("_per_second")
                for name in referenced_names
            )
            if reads_clock_directly or derived_name:
                violations.append(f"{path.name}:{assertion.lineno}")

    assert violations == [], (
        "raw model wall-clock assertions bypass the shared throughput policy: "
        + ", ".join(violations)
    )


def test_performance_scopes_do_not_assert_raw_clock_values() -> None:
    performance_tokens = {"perf", "performance", "throughput", "benchmark"}
    clock_names = {"elapsed", "rate", "throughput", "t_py", "t_rs"}
    violations = []

    for path in sorted(Path(__file__).parent.rglob("test*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        path_tokens = set(path.stem.replace("-", "_").split("_"))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            function_tokens = set(function.name.split("_"))
            if not performance_tokens & (path_tokens | function_tokens):
                continue
            for assertion in (node for node in ast.walk(function) if isinstance(node, ast.Assert)):
                referenced_names = {
                    node.id for node in ast.walk(assertion.test) if isinstance(node, ast.Name)
                }
                reads_clock_directly = any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"perf_counter", "process_time", "monotonic", "time"}
                    for node in ast.walk(assertion.test)
                )
                if reads_clock_directly or referenced_names & clock_names:
                    violations.append(
                        f"{path.relative_to(Path(__file__).parent)}:{assertion.lineno}"
                    )

    assert violations == [], (
        "raw performance wall-clock assertions bypass the shared throughput policy: "
        + ", ".join(violations)
    )
