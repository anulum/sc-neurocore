# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (evaluate_cases) from former test_alternative_path.py

from __future__ import annotations

from tests.alternative_path_support import *  # noqa: F403


def test_route_evaluate_cases_aggregates_reports():
    route = AlternativePathRoute(
        name="safe.batch",
        baseline=lambda x: np.asarray(x, dtype=np.float64) * 2.0,
        candidate=lambda x: np.asarray(x, dtype=np.float64) * 2.0,
        summary="Batch route",
        expected_behavior="All cases should match",
    )

    summary = route.evaluate_cases(
        [
            AlternativePathCase("small", args=([1.0, 2.0],)),
            AlternativePathCase("large", args=([4.0, 5.0, 6.0],)),
        ],
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )

    assert summary.route_name == "safe.batch"
    assert summary.total_cases == 2
    assert summary.matched_cases == 2
    assert summary.candidate_failures == 0
    assert len(summary.cases) == 2
    assert summary.to_report()["mode"] == "shadow"


def test_evaluate_cases_counts_candidate_failures():
    summary = _broken_shadow_route().evaluate_cases(
        [AlternativePathCase("c1"), AlternativePathCase("c2")],
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )
    assert summary.candidate_failures == 2
    assert summary.matched_cases == 0


def test_evaluate_cases_without_benchmark_yields_no_median_runtimes():
    route = AlternativePathRoute(
        name="safe.nobench",
        baseline=lambda: 1.0,
        candidate=lambda: 1.0,
        summary="Route evaluated without benchmarking",
        expected_behavior="Median runtimes collapse to None when timing is disabled",
    )
    summary = route.evaluate_cases(
        [AlternativePathCase("c1")],
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW, benchmark=False),
    )
    assert summary.median_baseline_runtime_ns is None
    assert summary.median_candidate_runtime_ns is None
