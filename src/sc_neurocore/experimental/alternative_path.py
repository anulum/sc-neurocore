# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safe harness for opt-in alternative execution paths

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import Any, Callable, Generic, Literal, Mapping, TypeVar, cast

import numpy as np

T = TypeVar("T")

ReturnedPath = Literal["baseline", "shadow-baseline", "candidate", "fallback-baseline"]


class AlternativePathMode(str, Enum):
    """Execution mode for an alternative path."""

    BASELINE = "baseline"
    SHADOW = "shadow"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class AlternativePathConfig:
    """Explicit policy for alternative-path execution.

    The default is intentionally conservative: baseline only.
    """

    enabled: bool = False
    mode: AlternativePathMode = AlternativePathMode.BASELINE
    fail_open: bool = True
    compare_outputs: bool = True
    benchmark: bool = True
    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-6


@dataclass(frozen=True)
class ComparisonStats:
    """Comparison summary between baseline and candidate outputs."""

    matched: bool
    comparable_leaf_count: int
    max_abs_diff: float | None
    max_rel_diff: float | None
    detail: str


@dataclass(frozen=True)
class AlternativePathResult(Generic[T]):
    """Structured result of a safe alternative-path execution."""

    route_name: str
    returned_path: ReturnedPath
    value: T
    baseline_value: T | None
    candidate_value: T | None
    baseline_runtime_ns: int | None
    candidate_runtime_ns: int | None
    comparison: ComparisonStats | None
    candidate_error: str | None

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for logging or benchmarking."""

        comparison = None
        if self.comparison is not None:
            comparison = {
                "matched": self.comparison.matched,
                "comparable_leaf_count": self.comparison.comparable_leaf_count,
                "max_abs_diff": self.comparison.max_abs_diff,
                "max_rel_diff": self.comparison.max_rel_diff,
                "detail": self.comparison.detail,
            }

        return {
            "route_name": self.route_name,
            "returned_path": self.returned_path,
            "baseline_runtime_ns": self.baseline_runtime_ns,
            "candidate_runtime_ns": self.candidate_runtime_ns,
            "candidate_error": self.candidate_error,
            "comparison": comparison,
        }


@dataclass(frozen=True)
class AlternativePathCase:
    """Named input case for repeated comparison and benchmarking."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlternativePathBatchSummary:
    """Aggregate report for a route evaluated over multiple named cases."""

    route_name: str
    mode: AlternativePathMode
    total_cases: int
    matched_cases: int
    candidate_failures: int
    median_baseline_runtime_ns: int | None
    median_candidate_runtime_ns: int | None
    cases: list[dict[str, Any]]

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the batch run."""

        return {
            "route_name": self.route_name,
            "mode": self.mode.value,
            "total_cases": self.total_cases,
            "matched_cases": self.matched_cases,
            "candidate_failures": self.candidate_failures,
            "median_baseline_runtime_ns": self.median_baseline_runtime_ns,
            "median_candidate_runtime_ns": self.median_candidate_runtime_ns,
            "cases": self.cases,
        }


Comparator = Callable[[Any, Any, AlternativePathConfig], ComparisonStats]


def _safe_stringify_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _numeric_comparison(
    baseline: Any,
    candidate: Any,
    config: AlternativePathConfig,
    context: str,
) -> ComparisonStats | None:
    try:
        baseline_arr = np.asarray(baseline)
        candidate_arr = np.asarray(candidate)
    except Exception:
        return None

    if baseline_arr.shape != candidate_arr.shape:
        return ComparisonStats(
            matched=False,
            comparable_leaf_count=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail=f"{context}: shape mismatch {baseline_arr.shape} != {candidate_arr.shape}",
        )

    kind_set = {baseline_arr.dtype.kind, candidate_arr.dtype.kind}
    if not kind_set.issubset({"b", "i", "u", "f"}):
        return None

    if baseline_arr.size == 0:
        return ComparisonStats(
            matched=True,
            comparable_leaf_count=1,
            max_abs_diff=0.0,
            max_rel_diff=0.0,
            detail=f"{context}: empty numeric outputs matched",
        )

    baseline_float = baseline_arr.astype(np.float64, copy=False)
    candidate_float = candidate_arr.astype(np.float64, copy=False)
    abs_diff = np.abs(baseline_float - candidate_float)
    denom = np.maximum(np.abs(baseline_float), config.absolute_tolerance)
    rel_diff = abs_diff / denom
    matched = np.allclose(
        baseline_float,
        candidate_float,
        atol=config.absolute_tolerance,
        rtol=config.relative_tolerance,
    )
    return ComparisonStats(
        matched=bool(matched),
        comparable_leaf_count=1,
        max_abs_diff=float(abs_diff.max(initial=0.0)),
        max_rel_diff=float(rel_diff.max(initial=0.0)),
        detail=(f"{context}: numeric outputs {'matched' if matched else 'diverged'}"),
    )


def _combine_stats(stats: list[ComparisonStats], context: str) -> ComparisonStats:
    if not stats:
        return ComparisonStats(
            matched=True,
            comparable_leaf_count=0,
            max_abs_diff=None,
            max_rel_diff=None,
            detail=f"{context}: no comparable leaves",
        )

    first_failure = next((s.detail for s in stats if not s.matched), None)
    abs_candidates = [s.max_abs_diff for s in stats if s.max_abs_diff is not None]
    rel_candidates = [s.max_rel_diff for s in stats if s.max_rel_diff is not None]
    return ComparisonStats(
        matched=all(s.matched for s in stats),
        comparable_leaf_count=sum(s.comparable_leaf_count for s in stats),
        max_abs_diff=max(abs_candidates) if abs_candidates else None,
        max_rel_diff=max(rel_candidates) if rel_candidates else None,
        detail=first_failure or f"{context}: all comparable leaves matched",
    )


def compare_outputs(
    baseline: Any,
    candidate: Any,
    config: AlternativePathConfig,
    context: str = "output",
) -> ComparisonStats:
    """Compare outputs from baseline and candidate implementations."""

    numeric_stats = _numeric_comparison(baseline, candidate, config, context)
    if numeric_stats is not None:
        return numeric_stats

    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        if set(baseline) != set(candidate):
            return ComparisonStats(
                matched=False,
                comparable_leaf_count=0,
                max_abs_diff=None,
                max_rel_diff=None,
                detail=f"{context}: mapping keys differ",
            )
        return _combine_stats(
            [
                compare_outputs(baseline[key], candidate[key], config, f"{context}.{key}")
                for key in baseline
            ],
            context,
        )

    if isinstance(baseline, (list, tuple)) and isinstance(candidate, (list, tuple)):
        if len(baseline) != len(candidate):
            return ComparisonStats(
                matched=False,
                comparable_leaf_count=0,
                max_abs_diff=None,
                max_rel_diff=None,
                detail=f"{context}: sequence length mismatch",
            )
        return _combine_stats(
            [
                compare_outputs(b_item, c_item, config, f"{context}[{index}]")
                for index, (b_item, c_item) in enumerate(zip(baseline, candidate))
            ],
            context,
        )

    # A bare ``int``/``float``/``bool`` pair is never routed here: ``_numeric_comparison``
    # runs first and returns a non-None result for every such pair (``np.asarray`` cannot
    # raise on a Python scalar, and the resulting dtype kind is always one of ``b``/``i``/
    # ``u``/``f``, so neither None-return branch can fire). The exact-equality fallback below
    # therefore handles the only leaves that reach this point — strings, ``None``,
    # mismatched-type pairs, and other non-numeric objects.
    matched = baseline == candidate
    return ComparisonStats(
        matched=matched,
        comparable_leaf_count=1,
        max_abs_diff=None,
        max_rel_diff=None,
        detail=f"{context}: exact {'match' if matched else 'mismatch'}",
    )


@dataclass(frozen=True)
class AlternativePathRoute(Generic[T]):
    """Named pair of stable baseline and experimental candidate implementations."""

    name: str
    baseline: Callable[..., T]
    candidate: Callable[..., T]
    summary: str
    expected_behavior: str
    comparator: Comparator = compare_outputs

    def describe(self) -> dict[str, str]:
        """Return route metadata for discovery and documentation."""

        return {
            "name": self.name,
            "summary": self.summary,
            "expected_behavior": self.expected_behavior,
        }

    def run(
        self, config: AlternativePathConfig | None = None, *args: Any, **kwargs: Any
    ) -> AlternativePathResult[T]:
        """Execute the route according to the provided policy."""

        config = config or AlternativePathConfig()

        baseline_value: T | None = None
        baseline_runtime_ns: int | None = None
        candidate_value: T | None = None
        candidate_runtime_ns: int | None = None
        comparison: ComparisonStats | None = None
        candidate_error: str | None = None

        if not config.enabled or config.mode is AlternativePathMode.BASELINE:
            baseline_value, baseline_runtime_ns = _call_timed(self.baseline, *args, **kwargs)
            return AlternativePathResult(
                route_name=self.name,
                returned_path="baseline",
                value=cast(T, baseline_value),
                baseline_value=baseline_value,
                candidate_value=None,
                baseline_runtime_ns=baseline_runtime_ns if config.benchmark else None,
                candidate_runtime_ns=None,
                comparison=None,
                candidate_error=None,
            )

        if config.mode is AlternativePathMode.SHADOW:
            baseline_value, baseline_runtime_ns = _call_timed(self.baseline, *args, **kwargs)
            try:
                candidate_value, candidate_runtime_ns = _call_timed(self.candidate, *args, **kwargs)
            except Exception as exc:
                candidate_error = _safe_stringify_error(exc)
            else:
                if config.compare_outputs:
                    comparison = self.comparator(baseline_value, candidate_value, config)
            return AlternativePathResult(
                route_name=self.name,
                returned_path="shadow-baseline",
                value=cast(T, baseline_value),
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                baseline_runtime_ns=baseline_runtime_ns if config.benchmark else None,
                candidate_runtime_ns=candidate_runtime_ns if config.benchmark else None,
                comparison=comparison,
                candidate_error=candidate_error,
            )

        try:
            candidate_value, candidate_runtime_ns = _call_timed(self.candidate, *args, **kwargs)
        except Exception as exc:
            candidate_error = _safe_stringify_error(exc)
            if not config.fail_open:
                raise
            baseline_value, baseline_runtime_ns = _call_timed(self.baseline, *args, **kwargs)
            return AlternativePathResult(
                route_name=self.name,
                returned_path="fallback-baseline",
                value=cast(T, baseline_value),
                baseline_value=baseline_value,
                candidate_value=None,
                baseline_runtime_ns=baseline_runtime_ns if config.benchmark else None,
                candidate_runtime_ns=None,
                comparison=None,
                candidate_error=candidate_error,
            )

        if config.compare_outputs or config.benchmark:
            baseline_value, baseline_runtime_ns = _call_timed(self.baseline, *args, **kwargs)
        if baseline_value is not None and config.compare_outputs:
            comparison = self.comparator(baseline_value, candidate_value, config)

        # The candidate path succeeded (no exception) — `candidate_value`
        # is guaranteed non-None even though the `T | None` narrowing
        # widened it; assert for mypy, then pass through.
        assert candidate_value is not None, (
            "candidate branch with no error must set candidate_value"
        )
        return AlternativePathResult(
            route_name=self.name,
            returned_path="candidate",
            value=candidate_value,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            baseline_runtime_ns=baseline_runtime_ns if config.benchmark else None,
            candidate_runtime_ns=candidate_runtime_ns if config.benchmark else None,
            comparison=comparison,
            candidate_error=None,
        )

    def evaluate_cases(
        self,
        cases: list[AlternativePathCase],
        config: AlternativePathConfig | None = None,
    ) -> AlternativePathBatchSummary:
        """Run the route across named cases and aggregate compare/benchmark results."""

        config = config or AlternativePathConfig()
        results = [self.run(config, *case.args, **case.kwargs) for case in cases]
        baseline_times = [
            result.baseline_runtime_ns
            for result in results
            if result.baseline_runtime_ns is not None
        ]
        candidate_times = [
            result.candidate_runtime_ns
            for result in results
            if result.candidate_runtime_ns is not None
        ]
        case_reports = []
        matched_cases = 0
        candidate_failures = 0
        for case, result in zip(cases, results):
            if result.comparison is not None and result.comparison.matched:
                matched_cases += 1
            if result.candidate_error is not None:
                candidate_failures += 1
            case_reports.append(
                {
                    "case_name": case.name,
                    "route_name": result.route_name,
                    "metadata": dict(case.metadata),
                    "returned_path": result.returned_path,
                    "baseline_runtime_ns": result.baseline_runtime_ns,
                    "candidate_runtime_ns": result.candidate_runtime_ns,
                    "candidate_error": result.candidate_error,
                    "comparison": (
                        None
                        if result.comparison is None
                        else {
                            "matched": result.comparison.matched,
                            "comparable_leaf_count": result.comparison.comparable_leaf_count,
                            "max_abs_diff": result.comparison.max_abs_diff,
                            "max_rel_diff": result.comparison.max_rel_diff,
                            "detail": result.comparison.detail,
                        }
                    ),
                }
            )
        return AlternativePathBatchSummary(
            route_name=self.name,
            mode=config.mode,
            total_cases=len(cases),
            matched_cases=matched_cases,
            candidate_failures=candidate_failures,
            median_baseline_runtime_ns=_median_runtime_ns(baseline_times),
            median_candidate_runtime_ns=_median_runtime_ns(candidate_times),
            cases=case_reports,
        )


class AlternativePathRegistry:
    """Registry for named alternative execution paths."""

    def __init__(self) -> None:
        self._routes: dict[str, AlternativePathRoute[Any]] = {}

    def register(self, route: AlternativePathRoute[Any]) -> None:
        if route.name in self._routes:
            raise ValueError(f"Alternative path '{route.name}' is already registered")
        self._routes[route.name] = route

    def get(self, name: str) -> AlternativePathRoute[Any]:
        try:
            return self._routes[name]
        except KeyError as exc:
            raise KeyError(f"Unknown alternative path: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._routes))

    def describe(self) -> list[dict[str, str]]:
        return [self._routes[name].describe() for name in self.names()]

    def run(
        self,
        name: str,
        config: AlternativePathConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> AlternativePathResult[Any]:
        return self.get(name).run(config, *args, **kwargs)

    def evaluate(
        self,
        name: str,
        cases: list[AlternativePathCase],
        config: AlternativePathConfig | None = None,
    ) -> AlternativePathBatchSummary:
        return self.get(name).evaluate_cases(cases, config)


def _call_timed(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, int]:
    start_ns = perf_counter_ns()
    value = fn(*args, **kwargs)
    elapsed_ns = perf_counter_ns() - start_ns
    return value, elapsed_ns


def _median_runtime_ns(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2
