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
