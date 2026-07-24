# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (physics_routes) from former test_alternative_path.py

from __future__ import annotations

from tests.alternative_path_support import *  # noqa: F403

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


