# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for safe alternative-path routing

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.experimental import solver_routes
from sc_neurocore.experimental import (
    AlternativePathCase,
    AlternativePathConfig,
    AlternativePathMode,
    AlternativePathRegistry,
    AlternativePathRoute,
    build_builtin_registry,
    build_demo_registry,
    default_report_path,
    make_delayed_recall_shared_state_route,
    make_harmonic_symplectic_route,
    make_heat_cosine_mode_route,
    make_kuramoto_noiseless_symplectic_lift_route,
    make_lif_subthreshold_exact_route,
    write_batch_report,
)
from sc_neurocore.experimental.builtins import builtin_cases_for_route


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


def test_registry_register_get_and_run():
    registry = AlternativePathRegistry()
    route = AlternativePathRoute(
        name="safe.registry",
        baseline=lambda x: x * 2,
        candidate=lambda x: x * 3,
        summary="Registry route",
        expected_behavior="Candidate can be enabled by name",
    )
    registry.register(route)

    assert registry.names() == ("safe.registry",)
    assert registry.get("safe.registry") is route
    result = registry.run("safe.registry", AlternativePathConfig(), 5)
    assert result.value == 10


def test_registry_rejects_duplicate_names():
    registry = AlternativePathRegistry()
    route = AlternativePathRoute(
        name="safe.dup",
        baseline=lambda: 1,
        candidate=lambda: 2,
        summary="Duplicate route",
        expected_behavior="Registry must reject duplicate names",
    )
    registry.register(route)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(route)


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


def test_registry_describe_returns_metadata():
    registry = build_demo_registry()

    descriptions = registry.describe()

    assert len(descriptions) == 1
    assert descriptions[0]["name"] == "demo.affine-sigmoid"
    assert "vectorised NumPy sigmoid candidate" in descriptions[0]["summary"]


def test_demo_registry_evaluate_matches_on_demo_route():
    registry = build_demo_registry()

    summary = registry.evaluate(
        "demo.affine-sigmoid",
        [
            AlternativePathCase("v1", args=([0.0, 1.0, -1.0],)),
            AlternativePathCase("v2", args=([2.0, -2.0],), kwargs={"bias": 0.25}),
        ],
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )

    assert summary.total_cases == 2
    assert summary.matched_cases == 2
    assert summary.candidate_failures == 0
    assert summary.median_baseline_runtime_ns is not None
    assert summary.median_candidate_runtime_ns is not None


def test_heat_cosine_mode_route_matches_exact_candidate_within_sampling_tolerance():
    route = make_heat_cosine_mode_route()

    result = route.run(
        AlternativePathConfig(
            enabled=True,
            mode=AlternativePathMode.SHADOW,
            absolute_tolerance=5e-2,
            relative_tolerance=1e-1,
        ),
        0.2,
        0.01,
        mode_index=1,
        length=1.0,
        diffusivity=0.5,
        num_walkers=40_000,
        dt=1e-4,
        seed=11,
    )

    assert result.returned_path == "shadow-baseline"
    assert result.comparison is not None
    assert result.comparison.matched


def test_builtin_registry_exposes_real_physics_route():
    registry = build_builtin_registry()

    descriptions = registry.describe()
    names = {item["name"] for item in descriptions}

    assert "memory.delayed-recall.shared-state" in names
    assert "physics.heat.cosine-mode" in names
    assert "physics.oscillator.harmonic-symplectic" in names
    assert "physics.kuramoto.noiseless-symplectic-lift" in names
    assert "solver.lif.subthreshold-exact" in names


def test_builtin_kuramoto_cases_include_higher_coupling_evidence():
    cases = builtin_cases_for_route("physics.kuramoto.noiseless-symplectic-lift")

    couplings = [case.kwargs["coupling"] for case in cases]

    assert max(couplings) >= 1.0
    assert any(
        case.metadata
        == {
            "oscillator_count": 4,
            "horizon": 0.008,
            "coupling": 1.0,
            "dt": 2.5e-4,
            "regime": "higher_coupling_noiseless",
        }
        for case in cases
    )


def test_delayed_recall_shared_state_route_beats_local_baseline():
    route = make_delayed_recall_shared_state_route()

    result = route.run(
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
        16,
    )

    assert result.returned_path == "shadow-baseline"
    assert result.baseline_value is not None
    assert result.candidate_value is not None
    assert result.comparison is not None
    assert result.comparison.matched
    assert result.candidate_value["mean_accuracy"] > result.baseline_value["mean_accuracy"]
    assert result.candidate_value["mean_accuracy"] >= 0.6


def test_kuramoto_noiseless_symplectic_lift_route_stays_close_on_short_horizon():
    route = make_kuramoto_noiseless_symplectic_lift_route()

    result = route.run(
        AlternativePathConfig(
            enabled=True,
            mode=AlternativePathMode.SHADOW,
            absolute_tolerance=5e-2,
            relative_tolerance=1e-1,
        ),
        np.array([0.1, 1.2, 2.4], dtype=np.float64),
        0.01,
        omegas=np.array([0.8, 1.0, 1.1], dtype=np.float64),
        coupling=0.18,
        dt=5e-4,
    )

    assert result.returned_path == "shadow-baseline"
    assert result.comparison is not None
    assert result.comparison.matched


def test_kuramoto_noiseless_symplectic_lift_route_matches_at_higher_coupling():
    route = make_kuramoto_noiseless_symplectic_lift_route()

    result = route.run(
        AlternativePathConfig(
            enabled=True,
            mode=AlternativePathMode.SHADOW,
            absolute_tolerance=8e-2,
            relative_tolerance=2e-1,
        ),
        np.array([0.05, 0.9, 1.7, 2.6], dtype=np.float64),
        0.008,
        omegas=np.array([0.88, 0.96, 1.04, 1.12], dtype=np.float64),
        coupling=1.0,
        dt=2.5e-4,
    )

    assert result.returned_path == "shadow-baseline"
    assert result.comparison is not None
    assert result.comparison.matched
    assert result.comparison.max_abs_diff is not None
    assert result.comparison.max_abs_diff < 1e-3
    assert result.candidate_value is not None
    assert np.isfinite(result.candidate_value["order_parameter"])
    assert np.isfinite(result.candidate_value["interaction_energy_drift"])


def test_kuramoto_noiseless_symplectic_lift_report_carries_regime_metadata():
    route = make_kuramoto_noiseless_symplectic_lift_route()
    cases = builtin_cases_for_route("physics.kuramoto.noiseless-symplectic-lift")

    summary = route.evaluate_cases(
        cases,
        AlternativePathConfig(
            enabled=True,
            mode=AlternativePathMode.SHADOW,
            absolute_tolerance=8e-2,
            relative_tolerance=2e-1,
        ),
    )

    higher_coupling_case = next(
        case
        for case in summary.to_report()["cases"]
        if case["case_name"] == "quartet_higher_coupling_short"
    )
    assert higher_coupling_case["metadata"] == {
        "oscillator_count": 4,
        "horizon": 0.008,
        "coupling": 1.0,
        "dt": 2.5e-4,
        "regime": "higher_coupling_noiseless",
    }


@pytest.mark.parametrize(
    ("initial_phases", "horizon", "omegas", "coupling", "dt", "match"),
    [
        (np.array([]), 0.01, np.array([]), 0.18, 5e-4, "initial_phases"),
        (np.array([0.1, np.nan, 2.4]), 0.01, np.array([0.8, 1.0, 1.1]), 0.18, 5e-4, "finite"),
        (np.array([0.1, 1.2, 2.4]), 0.01, np.array([0.8, 1.0]), 0.18, 5e-4, "omegas"),
        (np.array([0.1, 1.2, 2.4]), 0.01, np.array([0.8, np.inf, 1.1]), 0.18, 5e-4, "finite"),
        (np.array([0.1, 1.2, 2.4]), 0.0, np.array([0.8, 1.0, 1.1]), 0.18, 5e-4, "horizon"),
        (np.array([0.1, 1.2, 2.4]), True, np.array([0.8, 1.0, 1.1]), 0.18, 5e-4, "horizon"),
        (np.array([0.1, 1.2, 2.4]), 0.01, np.array([0.8, 1.0, 1.1]), -0.1, 5e-4, "coupling"),
        (np.array([0.1, 1.2, 2.4]), 0.01, np.array([0.8, 1.0, 1.1]), True, 5e-4, "coupling"),
        (np.array([0.1, 1.2, 2.4]), 0.01, np.array([0.8, 1.0, 1.1]), 0.18, 0.0, "dt"),
        (np.array([0.1, 1.2, 2.4]), 0.01, np.array([0.8, 1.0, 1.1]), 0.18, True, "dt"),
    ],
)
def test_kuramoto_noiseless_symplectic_lift_route_rejects_invalid_inputs(
    initial_phases,
    horizon,
    omegas,
    coupling,
    dt,
    match,
):
    route = make_kuramoto_noiseless_symplectic_lift_route()

    with pytest.raises(ValueError, match=match):
        route.run(
            AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
            initial_phases,
            horizon,
            omegas=omegas,
            coupling=coupling,
            dt=dt,
        )


def test_harmonic_symplectic_route_matches_rk4_with_low_energy_drift():
    route = make_harmonic_symplectic_route()

    result = route.run(
        AlternativePathConfig(
            enabled=True,
            mode=AlternativePathMode.SHADOW,
            absolute_tolerance=5e-3,
            relative_tolerance=5e-2,
        ),
        1.0,
        0.0,
        0.5 * np.pi,
        dt=5e-3,
    )

    assert result.returned_path == "shadow-baseline"
    assert result.comparison is not None
    assert result.comparison.matched
    assert result.candidate_value is not None
    energy_drift = result.candidate_value["relative_energy_drift"]
    assert isinstance(energy_drift, float)
    assert energy_drift < 1e-3


@pytest.mark.parametrize(
    ("q0", "p0", "horizon", "kwargs", "match"),
    [
        (0.0, 0.0, 1.0, {"dt": 1e-2}, "energy"),
        (True, 0.0, 1.0, {"dt": 1e-2}, "q0"),
        (float("nan"), 0.0, 1.0, {"dt": 1e-2}, "q0"),
        (1.0, 0.0, 0.0, {"dt": 1e-2}, "horizon"),
        (1.0, 0.0, 1.0, {"dt": 0.0}, "dt"),
    ],
)
def test_harmonic_symplectic_route_rejects_invalid_hamiltonian_inputs(
    q0, p0, horizon, kwargs, match
):
    route = make_harmonic_symplectic_route()

    with pytest.raises(ValueError, match=match):
        route.run(
            AlternativePathConfig(
                enabled=True,
                mode=AlternativePathMode.SHADOW,
            ),
            q0,
            p0,
            horizon,
            **kwargs,
        )


def test_write_batch_report_writes_json_file(tmp_path):
    registry = build_demo_registry()
    summary = registry.evaluate(
        "demo.affine-sigmoid",
        [AlternativePathCase("small", args=([0.0, 1.0, -1.0],))],
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )
    out = tmp_path / "report.json"

    written = write_batch_report(summary, out)

    assert written == out
    text = out.read_text()
    assert '"route_name": "demo.affine-sigmoid"' in text


def test_default_report_path_is_under_benchmarks_results():
    path = default_report_path("physics.heat.cosine-mode")

    assert str(path) == "benchmarks/results/experimental_physics_heat_cosine_mode.json"


def test_default_report_path_handles_second_physics_route():
    path = default_report_path("physics.oscillator.harmonic-symplectic")

    assert (
        str(path) == "benchmarks/results/experimental_physics_oscillator_harmonic_symplectic.json"
    )


def test_lif_subthreshold_exact_route_matches_rk4_and_stays_below_threshold():
    route = make_lif_subthreshold_exact_route()

    result = route.run(
        AlternativePathConfig(
            enabled=True,
            mode=AlternativePathMode.SHADOW,
            absolute_tolerance=1e-4,
            relative_tolerance=1e-4,
        ),
        -65.0,
        10.0,
        20.0,
        tau=20.0,
        v_rest=-65.0,
        v_thresh=-50.0,
        r_m=1.0,
        dt=1e-2,
    )

    assert result.returned_path == "shadow-baseline"
    assert result.comparison is not None
    assert result.comparison.matched
    assert result.candidate_value is not None
    assert result.candidate_value["subthreshold"] is True
    assert result.candidate_value["predicted_spike_time"] is None


@pytest.mark.parametrize(
    ("v0", "current", "horizon", "kwargs", "match"),
    [
        (-65.0, 10.0, 0.0, {}, "horizon"),
        (-65.0, 10.0, True, {}, "horizon"),
        (-50.0, 10.0, 20.0, {}, "v0"),
        (-65.0, 20.0, 20.0, {}, "suprathreshold"),
        (-65.0, float("nan"), 20.0, {}, "current"),
        (-65.0, 10.0, 20.0, {"tau": 0.0}, "tau"),
        (-65.0, 10.0, 20.0, {"v_rest": -50.0}, "v_rest"),
        (-65.0, 10.0, 20.0, {"r_m": True}, "r_m"),
        (-65.0, 10.0, 20.0, {"dt": 0.0}, "dt"),
    ],
)
def test_lif_subthreshold_exact_route_rejects_invalid_domain(v0, current, horizon, kwargs, match):
    route = make_lif_subthreshold_exact_route()

    with pytest.raises(ValueError, match=match):
        route.run(
            AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
            v0,
            current,
            horizon,
            tau=kwargs.get("tau", 20.0),
            v_rest=kwargs.get("v_rest", -65.0),
            v_thresh=kwargs.get("v_thresh", -50.0),
            r_m=kwargs.get("r_m", 1.0),
            dt=kwargs.get("dt", 1e-2),
        )


def test_lif_subthreshold_exact_route_rejects_nonfinite_steady_state_voltage():
    route = make_lif_subthreshold_exact_route()

    with pytest.raises(ValueError, match="steady-state voltage"):
        route.run(
            AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
            -1e308,
            1e308,
            20.0,
            tau=20.0,
            v_rest=-1e308,
            v_thresh=-1e307,
            r_m=10.0,
            dt=1e-2,
        )


def test_lif_rk4_baseline_rejects_nonfinite_numerical_candidate():
    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="baseline voltage"):
        solver_routes._lif_rk4_baseline(
            -60.0,
            0.0,
            1.0,
            tau=1e-308,
            v_rest=-65.0,
            v_thresh=-50.0,
            r_m=1.0,
            dt=1.0,
        )


def test_default_report_path_handles_solver_route():
    path = default_report_path("solver.lif.subthreshold-exact")

    assert str(path) == "benchmarks/results/experimental_solver_lif_subthreshold_exact.json"


def test_result_report_is_json_friendly():
    route = AlternativePathRoute(
        name="safe.report",
        baseline=lambda: {"x": np.array([1.0, 2.0])},
        candidate=lambda: {"x": np.array([1.0, 2.0])},
        summary="Report route",
        expected_behavior="Serializable report should not leak raw arrays",
    )

    result = route.run(
        AlternativePathConfig(enabled=True, mode=AlternativePathMode.SHADOW),
    )
    report = result.to_report()

    assert report["route_name"] == "safe.report"
    assert report["returned_path"] == "shadow-baseline"
    assert report["comparison"]["matched"] is True
