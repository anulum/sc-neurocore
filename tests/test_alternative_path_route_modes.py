# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (route_modes) from former test_alternative_path.py

from __future__ import annotations

from tests.alternative_path_support import *  # noqa: F403


def test_baseline_mode_does_not_call_candidate():
    calls = {"candidate": 0}

    def baseline(x: int) -> int:
        return x + 1

    def candidate(x: int) -> int:
        calls["candidate"] += 1
        return x + 2

    route = AlternativePathRoute(
        name="safe.add",
        baseline=baseline,
        candidate=candidate,
        summary="Candidate add path",
        expected_behavior="Returns a larger value if promoted",
    )

    result = route.run(AlternativePathConfig(), 3)

    assert result.value == 4
    assert result.returned_path == "baseline"
    assert calls["candidate"] == 0
    assert result.candidate_value is None


def test_shadow_mode_returns_baseline_and_compares_candidate():
    route = AlternativePathRoute(
        name="safe.vector",
        baseline=lambda: np.array([1.0, 2.0, 3.0]),
        candidate=lambda: np.array([1.0, 2.0, 3.0 + 1e-10]),
        summary="Vector candidate path",
        expected_behavior="Candidate should numerically agree with baseline",
    )

    result = route.run(
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )

    assert result.returned_path == "shadow-baseline"
    assert np.allclose(result.value, np.array([1.0, 2.0, 3.0]))
    assert result.candidate_value is not None
    assert result.comparison is not None
    assert result.comparison.matched


def test_candidate_mode_returns_candidate_and_records_baseline():
    route = AlternativePathRoute(
        name="safe.scalar",
        baseline=lambda x: x + 1,
        candidate=lambda x: x + 2,
        summary="Scalar candidate path",
        expected_behavior="Candidate intentionally differs",
    )

    result = route.run(
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.CANDIDATE),
        4,
    )

    assert result.returned_path == "candidate"
    assert result.value == 6
    assert result.baseline_value == 5
    assert result.comparison is not None
    assert not result.comparison.matched


def test_candidate_mode_fail_open_falls_back_to_baseline():
    route = AlternativePathRoute(
        name="safe.fail-open",
        baseline=lambda: "stable",
        candidate=lambda: (_ for _ in ()).throw(RuntimeError("candidate broke")),
        summary="Fail-open route",
        expected_behavior="Falls back to baseline on candidate error",
    )

    result = route.run(
        AlternativePathConfig(
            enabled=True,
            mode=AlternativePathMode.CANDIDATE,
            fail_open=True,
        )
    )

    assert result.returned_path == "fallback-baseline"
    assert result.value == "stable"
    assert result.candidate_error is not None
    assert "candidate broke" in result.candidate_error


def test_candidate_mode_fail_closed_raises():
    route = AlternativePathRoute(
        name="safe.fail-closed",
        baseline=lambda: "stable",
        candidate=lambda: (_ for _ in ()).throw(RuntimeError("stop")),
        summary="Fail-closed route",
        expected_behavior="Propagates candidate failures",
    )

    with pytest.raises(RuntimeError, match="stop"):
        route.run(
            AlternativePathConfig(
                enabled=True,
                mode=AlternativePathMode.CANDIDATE,
                fail_open=False,
            )
        )


def test_shadow_mode_records_candidate_error_without_comparison():
    result = _broken_shadow_route().run(
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW)
    )
    assert result.returned_path == "shadow-baseline"
    assert result.value == 1.0
    assert result.candidate_value is None
    assert result.comparison is None
    assert result.candidate_error is not None
    assert "shadow boom" in result.candidate_error
