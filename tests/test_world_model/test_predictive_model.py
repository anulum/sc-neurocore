# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the LGSSM Kalman / RTS / EM predictive model

"""Tests for `sc_neurocore.world_model.predictive_model`.

The module was rewritten 2026-04-17 from a deterministic linear
matmul placeholder to a proper Linear Gaussian State-Space Model
(Kalman filter + RTS smoother + EM learner). These tests exercise
the math invariants per Bishop 2006 §13.3 and Murphy 2023 §29.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.world_model.predictive_model import (
    EMLearner,
    KalmanFilter,
    LinearGaussianSSM,
    PredictiveWorldModel,
    RTSSmoother,
)


# ───────────────────────── helper ─────────────────────────


def _scalar_random_walk(seed: int = 7) -> LinearGaussianSSM:
    """1-D random walk with noisy observation: x_t = x_{t-1} + w, y_t = x_t + v."""
    return LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.zeros((1, 0)),
        C=np.array([[1.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.1]]),
        R=np.array([[1.0]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )


def _simulate(model: LinearGaussianSSM, T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample (states, observations) from the SSM."""
    rng = np.random.default_rng(seed)
    d = model.state_dim
    p = model.obs_dim
    states = np.zeros((T, d))
    obs = np.zeros((T, p))
    x = rng.multivariate_normal(model.mu_0, model.Sigma_0)
    for t in range(T):
        states[t] = x
        obs[t] = model.C @ x + rng.multivariate_normal(np.zeros(p), model.R)
        x = model.A @ x + rng.multivariate_normal(np.zeros(d), model.Q)
    return states, obs


# ───────────────────────── LGSSM dataclass ─────────────────────────


def test_lgssm_validates_shapes() -> None:
    """Mismatched-shape parameters raise ValueError.

    Construct A=(3,3) but C=(2,2). state_dim derived from A is 3,
    so validation must detect that C should be (?, 3) but is (2, 2).
    """
    with pytest.raises(ValueError, match="C must be"):
        LinearGaussianSSM(
            A=np.eye(3),
            B=np.zeros((3, 0)),
            C=np.eye(2),
            D=np.zeros((2, 0)),
            Q=np.eye(3),
            R=np.eye(2),
            mu_0=np.zeros(3),
            Sigma_0=np.eye(3),
        )


def test_lgssm_rejects_non_symmetric_covariance() -> None:
    Q = np.array([[1.0, 0.5], [-0.5, 1.0]])  # not symmetric
    with pytest.raises(ValueError, match="must be symmetric"):
        LinearGaussianSSM(
            A=np.eye(2),
            B=np.zeros((2, 0)),
            C=np.eye(2),
            D=np.zeros((2, 0)),
            Q=Q,
            R=np.eye(2),
            mu_0=np.zeros(2),
            Sigma_0=np.eye(2),
        )


def test_lgssm_random_is_stable() -> None:
    """LGSSM.random() must produce a stable A (spectral radius < 1)."""
    model = LinearGaussianSSM.random(state_dim=4, obs_dim=3, control_dim=0, seed=42)
    eigs = np.linalg.eigvals(model.A)
    assert np.max(np.abs(eigs)) < 1.0, "random A is not stable"


# ───────────────────────── Kalman filter ─────────────────────────


def test_kalman_filter_returns_correct_shapes() -> None:
    model = _scalar_random_walk()
    T = 50
    obs = np.random.default_rng(1).standard_normal((T, 1))
    fr = KalmanFilter(model).filter(obs)
    assert fr.means.shape == (T, 1)
    assert fr.covariances.shape == (T, 1, 1)
    assert fr.pred_means.shape == (T, 1)
    assert fr.pred_covariances.shape == (T, 1, 1)
    assert isinstance(fr.log_likelihood, float)


def test_kalman_filter_log_likelihood_finite() -> None:
    model = _scalar_random_walk()
    T = 50
    obs = np.random.default_rng(2).standard_normal((T, 1))
    fr = KalmanFilter(model).filter(obs)
    assert np.isfinite(fr.log_likelihood)


def test_kalman_filter_low_noise_tracks_observation() -> None:
    """When R → 0, the filter mean should hug the observation."""
    model = LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.zeros((1, 0)),
        C=np.array([[1.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.01]]),
        R=np.array([[1e-6]]),  # near-perfect observation
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    obs = np.array([[1.0], [2.0], [3.0], [4.0]])
    fr = KalmanFilter(model).filter(obs)
    np.testing.assert_allclose(fr.means.flatten(), obs.flatten(), atol=1e-3)


def test_kalman_filter_high_noise_relies_on_prior() -> None:
    """When R → ∞, the filter mean should ignore observations."""
    model = LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.zeros((1, 0)),
        C=np.array([[1.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.01]]),
        R=np.array([[1e10]]),  # near-useless observation
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[0.01]]),
    )
    # Observations are large outliers
    obs = np.array([[100.0], [-100.0], [50.0], [-50.0]])
    fr = KalmanFilter(model).filter(obs)
    # The filter mean stays close to mu_0 = 0 because the observation
    # is so noisy that the Kalman gain is tiny.
    assert np.all(np.abs(fr.means) < 1.0)


def test_kalman_filter_with_controls() -> None:
    """Filter must accept control input when control_dim > 0."""
    model = LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.zeros((1, 1)),
        Q=np.array([[0.1]]),
        R=np.array([[0.1]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    obs = np.zeros((10, 1))
    controls = np.ones((10, 1))
    fr = KalmanFilter(model).filter(obs, controls=controls)
    # Without control: state would stay near 0; WITH unit control
    # each step adds B·u = 1, so the state drifts upwards.
    assert fr.pred_means[5, 0] > 0.0


def test_kalman_filter_rejects_wrong_obs_dim() -> None:
    model = _scalar_random_walk()
    bad_obs = np.zeros((10, 3))  # obs_dim = 1, not 3
    with pytest.raises(ValueError, match="obs dim"):
        KalmanFilter(model).filter(bad_obs)


def test_kalman_filter_covariance_psd() -> None:
    """Filtered covariance must remain PSD across all timesteps."""
    model = LinearGaussianSSM.random(state_dim=4, obs_dim=2, seed=11)
    obs = np.random.default_rng(11).standard_normal((30, 2))
    fr = KalmanFilter(model).filter(obs)
    for t in range(30):
        eigs = np.linalg.eigvalsh(fr.covariances[t])
        assert np.all(eigs > -1e-9), f"non-PSD at t={t}: eigs={eigs}"


# ───────────────────────── RTS smoother ─────────────────────────


def test_rts_smoother_last_step_equals_filter_last_step() -> None:
    """Smoother and filter must agree at t=T-1 (no future to incorporate)."""
    model = _scalar_random_walk()
    obs = np.random.default_rng(3).standard_normal((20, 1))
    fr = KalmanFilter(model).filter(obs)
    sr = RTSSmoother(model).smooth(fr)
    np.testing.assert_allclose(sr.means[-1], fr.means[-1])
    np.testing.assert_allclose(sr.covariances[-1], fr.covariances[-1])


def test_rts_smoother_reduces_uncertainty() -> None:
    """Smoothed covariance ≤ filtered covariance (smoothing uses more info)."""
    model = _scalar_random_walk()
    obs = np.random.default_rng(4).standard_normal((30, 1))
    fr = KalmanFilter(model).filter(obs)
    sr = RTSSmoother(model).smooth(fr)
    # For 1-D, compare scalar variances
    for t in range(29):
        assert sr.covariances[t, 0, 0] <= fr.covariances[t, 0, 0] + 1e-9, (
            f"smoothed var > filtered var at t={t}"
        )


def test_rts_smoother_cross_covariances_shape() -> None:
    model = LinearGaussianSSM.random(state_dim=3, obs_dim=2, seed=5)
    obs = np.random.default_rng(5).standard_normal((25, 2))
    fr = KalmanFilter(model).filter(obs)
    sr = RTSSmoother(model).smooth(fr)
    assert sr.cross_covariances.shape == (24, 3, 3)


# ───────────────────────── EM learner ─────────────────────────


def test_em_log_likelihood_monotone_non_decreasing() -> None:
    """EM likelihood must be monotone non-decreasing across iterations.

    This is the foundational EM theorem (Dempster et al. 1977). A
    decrease would indicate a bug in the M-step.
    """
    rng = np.random.default_rng(42)
    true_model = _scalar_random_walk()
    _, obs = _simulate(true_model, T=200, seed=42)

    init = LinearGaussianSSM(
        A=np.array([[0.5]]),  # wrong A
        B=np.zeros((1, 0)),
        C=np.array([[2.0]]),  # wrong C
        D=np.zeros((1, 0)),
        Q=np.array([[1.0]]),  # wrong Q
        R=np.array([[5.0]]),  # wrong R
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    learner = EMLearner(max_iter=10, tol=1e-9)
    learner.fit(obs, init)
    history = learner.log_likelihood_history
    assert len(history) > 1
    # Allow tiny round-off (1e-6 absolute) — EM is monotone in exact
    # arithmetic but float rounding can shave a fraction off.
    diffs = np.diff(history)
    assert np.all(diffs > -1e-6), f"EM log-lik not monotone: history={history}"


def test_em_improves_held_out_log_likelihood() -> None:
    """EM-fit model should achieve higher log-likelihood on held-out data
    than the (poorly initialised) starting point.

    Note: directly comparing the learned C to the true C is brittle
    because LGSSM has a well-known sign + scale ambiguity (Bishop
    2006 §13.3.4): the pair (A, C) and (αA, C/α) are observationally
    equivalent for any α > 0. The proper recovery test is on the
    OBSERVATION-LIKELIHOOD, which IS identifiable.
    """
    true_model = LinearGaussianSSM(
        A=np.array([[0.95]]),
        B=np.zeros((1, 0)),
        C=np.array([[2.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.05]]),
        R=np.array([[0.1]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    _, obs_train = _simulate(true_model, T=400, seed=7)
    _, obs_test = _simulate(true_model, T=200, seed=99)

    init = LinearGaussianSSM(
        A=np.array([[0.95]]),
        B=np.zeros((1, 0)),
        C=np.array([[0.5]]),  # wrong
        D=np.zeros((1, 0)),
        Q=np.array([[0.05]]),
        R=np.array([[0.1]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    init_ll = KalmanFilter(init).filter(obs_test).log_likelihood
    learned = EMLearner(max_iter=50, tol=1e-6).fit(obs_train, init)
    learned_ll = KalmanFilter(learned).filter(obs_test).log_likelihood

    assert learned_ll > init_ll, (
        f"EM did not improve test log-likelihood: init={init_ll:.2f}, learned={learned_ll:.2f}"
    )


# ───────────────────────── PredictiveWorldModel wrapper ───────────


def test_legacy_predict_next_state_shape() -> None:
    """Legacy API shape preserved (regression for existing tests)."""
    m = PredictiveWorldModel(state_dim=4, action_dim=2)
    pred = m.predict_next_state(np.zeros(4), np.zeros(2))
    assert pred.shape == (4,)


def test_legacy_forecast_returns_n_steps() -> None:
    m = PredictiveWorldModel(state_dim=4, action_dim=2)
    actions = [np.zeros(2), np.ones(2), np.full(2, 0.5)]
    traj = m.forecast(np.zeros(4), actions)
    assert len(traj) == 3
    for x in traj:
        assert x.shape == (4,)


def test_predict_with_cov_grows_uncertainty() -> None:
    """E[Σ_{t+1}] = A Σ_t A^T + Q ⇒ covariance must grow under stable A."""
    m = PredictiveWorldModel(state_dim=2, action_dim=1, seed=1)
    Sigma_0 = np.eye(2) * 0.01
    _, Sigma_1 = m.predict_next_state_with_cov(
        np.zeros(2),
        Sigma_0,
        np.zeros(1),
    )
    # trace(Sigma_1) > trace(Sigma_0) under non-zero Q
    assert np.trace(Sigma_1) > np.trace(Sigma_0)


def test_forecast_with_cov_returns_pairs() -> None:
    m = PredictiveWorldModel(state_dim=3, action_dim=1, seed=2)
    actions = [np.zeros(1), np.ones(1)]
    traj = m.forecast_with_cov(np.zeros(3), np.eye(3), actions)
    assert len(traj) == 2
    for mu, Sigma in traj:
        assert mu.shape == (3,)
        assert Sigma.shape == (3, 3)


def test_reset_restores_prior() -> None:
    m = PredictiveWorldModel(state_dim=2, action_dim=0, seed=3)
    # Construct an action of correct shape (control_dim=0 means empty)
    m._mu = np.array([99.0, 99.0])
    m._Sigma = np.eye(2) * 100.0
    m.reset()
    np.testing.assert_array_equal(m._mu, m.model.mu_0)
    np.testing.assert_array_equal(m._Sigma, m.model.Sigma_0)


# ───────────────────────── Rust ↔ Python parity ─────────────────────────

# These tests skip if the Rust LGSSM backend is not present in
# the engine wheel (e.g. a fresh checkout without
# `cd bridge && maturin develop --release`).

from sc_neurocore.world_model.predictive_model import _HAS_RUST_LGSSM


@pytest.mark.skipif(
    not _HAS_RUST_LGSSM,
    reason="Rust LGSSM backend not built (run `cd bridge && maturin develop --release`)",
)
def test_rust_parity_means_match_python_to_float64() -> None:
    """Rust filtered means must match Python to within float64 round-off.

    This is the parity contract — any divergence indicates a bug
    in the Rust implementation (sign flip, wrong matrix order,
    indexing error, etc.).
    """
    rng = np.random.default_rng(101)
    model = LinearGaussianSSM.random(state_dim=4, obs_dim=3, control_dim=0, seed=101)
    obs = rng.standard_normal((50, 3))

    py_result = KalmanFilter(model).filter(obs, backend="python")
    ru_result = KalmanFilter(model).filter(obs, backend="rust")

    np.testing.assert_allclose(ru_result.means, py_result.means, atol=1e-9)


@pytest.mark.skipif(not _HAS_RUST_LGSSM, reason="Rust LGSSM backend not built")
def test_rust_parity_covariances_match_python() -> None:
    rng = np.random.default_rng(102)
    model = LinearGaussianSSM.random(state_dim=3, obs_dim=2, control_dim=0, seed=102)
    obs = rng.standard_normal((30, 2))

    py_result = KalmanFilter(model).filter(obs, backend="python")
    ru_result = KalmanFilter(model).filter(obs, backend="rust")

    np.testing.assert_allclose(
        ru_result.covariances,
        py_result.covariances,
        atol=1e-9,
    )


@pytest.mark.skipif(not _HAS_RUST_LGSSM, reason="Rust LGSSM backend not built")
def test_rust_parity_log_likelihood_matches_python() -> None:
    rng = np.random.default_rng(103)
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=2, control_dim=0, seed=103)
    obs = rng.standard_normal((100, 2))

    py_ll = KalmanFilter(model).filter(obs, backend="python").log_likelihood
    ru_ll = KalmanFilter(model).filter(obs, backend="rust").log_likelihood

    assert abs(ru_ll - py_ll) < 1e-9, f"log-likelihood mismatch: python={py_ll}, rust={ru_ll}"


@pytest.mark.skipif(not _HAS_RUST_LGSSM, reason="Rust LGSSM backend not built")
def test_rust_backend_explicit_request_works() -> None:
    """`backend='rust'` must dispatch to the Rust path successfully."""
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=1, seed=1)
    obs = np.zeros((10, 1))
    fr = KalmanFilter(model).filter(obs, backend="rust")
    assert fr.means.shape == (10, 2)


def test_rust_backend_unavailable_raises_when_explicitly_requested() -> None:
    """If backend='rust' is requested but unavailable, raise RuntimeError."""
    if _HAS_RUST_LGSSM:
        pytest.skip("Rust backend IS available — cannot test the unavailable path")
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=1, seed=1)
    obs = np.zeros((5, 1))
    with pytest.raises(RuntimeError, match="Rust LGSSM backend"):
        KalmanFilter(model).filter(obs, backend="rust")


def test_invalid_backend_raises() -> None:
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=1, seed=1)
    obs = np.zeros((5, 1))
    with pytest.raises(ValueError, match="backend must be"):
        KalmanFilter(model).filter(obs, backend="cuda")


# ───────────────────────── Julia ↔ Python parity ─────────────────────────

# These tests trigger Julia startup (~5 s on first call). They
# skip cleanly when juliacall is not installed or the .jl module
# is missing.

import importlib.util as _il_util


def _julia_available() -> bool:
    """Avoid importing juliacall at module-load (5 s startup)."""
    if _il_util.find_spec("juliacall") is None:
        return False
    import os as _os

    jl_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
        "src",
        "sc_neurocore",
        "accel",
        "julia",
        "world_model",
        "predictive_model.jl",
    )
    return _os.path.isfile(jl_path)


@pytest.mark.skipif(
    not _julia_available(),
    reason="juliacall not installed or accel/julia/world_model/predictive_model.jl missing",
)
def test_julia_parity_means_match_python() -> None:
    """Julia filtered means must match Python to atol=1e-9.

    Same parity contract as Rust: `JuliaLGSSMKalmanFilter ==
    PythonLGSSMKalmanFilter` to within float64 round-off.
    """
    rng = np.random.default_rng(201)
    model = LinearGaussianSSM.random(state_dim=4, obs_dim=3, control_dim=0, seed=201)
    obs = rng.standard_normal((50, 3))

    py = KalmanFilter(model).filter(obs, backend="python")
    ju = KalmanFilter(model).filter(obs, backend="julia")

    np.testing.assert_allclose(ju.means, py.means, atol=1e-9)


@pytest.mark.skipif(not _julia_available(), reason="Julia LGSSM backend unavailable")
def test_julia_parity_covariances_match_python() -> None:
    rng = np.random.default_rng(202)
    model = LinearGaussianSSM.random(state_dim=3, obs_dim=2, control_dim=0, seed=202)
    obs = rng.standard_normal((30, 2))

    py = KalmanFilter(model).filter(obs, backend="python")
    ju = KalmanFilter(model).filter(obs, backend="julia")

    np.testing.assert_allclose(ju.covariances, py.covariances, atol=1e-9)


@pytest.mark.skipif(not _julia_available(), reason="Julia LGSSM backend unavailable")
def test_julia_parity_log_likelihood_matches_python() -> None:
    rng = np.random.default_rng(203)
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=2, control_dim=0, seed=203)
    obs = rng.standard_normal((100, 2))

    py_ll = KalmanFilter(model).filter(obs, backend="python").log_likelihood
    ju_ll = KalmanFilter(model).filter(obs, backend="julia").log_likelihood

    assert abs(ju_ll - py_ll) < 1e-9, f"log-likelihood mismatch: python={py_ll}, julia={ju_ll}"


@pytest.mark.skipif(not _julia_available(), reason="Julia LGSSM backend unavailable")
def test_three_backend_parity_when_all_available() -> None:
    """Python = Rust = Julia results must agree to atol=1e-9."""
    rng = np.random.default_rng(204)
    model = LinearGaussianSSM.random(state_dim=3, obs_dim=2, control_dim=0, seed=204)
    obs = rng.standard_normal((40, 2))

    py = KalmanFilter(model).filter(obs, backend="python")
    ju = KalmanFilter(model).filter(obs, backend="julia")

    np.testing.assert_allclose(ju.log_likelihood, py.log_likelihood, atol=1e-9)
    np.testing.assert_allclose(ju.means, py.means, atol=1e-9)

    if _HAS_RUST_LGSSM:
        ru = KalmanFilter(model).filter(obs, backend="rust")
        np.testing.assert_allclose(ru.log_likelihood, py.log_likelihood, atol=1e-9)
        np.testing.assert_allclose(ru.means, py.means, atol=1e-9)
        np.testing.assert_allclose(ru.means, ju.means, atol=1e-9)


def test_julia_backend_unavailable_raises_when_explicitly_requested() -> None:
    """If backend='julia' is requested but unavailable, raise RuntimeError."""
    if _julia_available():
        pytest.skip("Julia backend IS available — cannot test the unavailable path")
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=1, seed=1)
    obs = np.zeros((5, 1))
    with pytest.raises(RuntimeError, match="Julia LGSSM backend"):
        KalmanFilter(model).filter(obs, backend="julia")


# ───────────────────────── Go ↔ Python parity ─────────────────────────


def _go_available() -> bool:
    """Check liblgssm.so is present without forcing the load."""
    import os as _os

    so_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
        "src",
        "sc_neurocore",
        "accel",
        "go",
        "lgssm",
        "liblgssm.so",
    )
    return _os.path.isfile(so_path)


@pytest.mark.skipif(
    not _go_available(),
    reason="Go shared lib (accel/go/lgssm/liblgssm.so) not built — "
    "run `cd src/sc_neurocore/accel/go/lgssm && go build "
    "-buildmode=c-shared -o liblgssm.so lgssm.go`",
)
def test_go_parity_means_match_python() -> None:
    """Go filtered means must match Python to atol=1e-9."""
    rng = np.random.default_rng(301)
    model = LinearGaussianSSM.random(state_dim=4, obs_dim=3, control_dim=0, seed=301)
    obs = rng.standard_normal((50, 3))

    py = KalmanFilter(model).filter(obs, backend="python")
    go = KalmanFilter(model).filter(obs, backend="go")

    np.testing.assert_allclose(go.means, py.means, atol=1e-9)


@pytest.mark.skipif(not _go_available(), reason="Go LGSSM backend unavailable")
def test_go_parity_covariances_match_python() -> None:
    rng = np.random.default_rng(302)
    model = LinearGaussianSSM.random(state_dim=3, obs_dim=2, control_dim=0, seed=302)
    obs = rng.standard_normal((30, 2))

    py = KalmanFilter(model).filter(obs, backend="python")
    go = KalmanFilter(model).filter(obs, backend="go")

    np.testing.assert_allclose(go.covariances, py.covariances, atol=1e-9)


@pytest.mark.skipif(not _go_available(), reason="Go LGSSM backend unavailable")
def test_go_parity_log_likelihood_matches_python() -> None:
    rng = np.random.default_rng(303)
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=2, control_dim=0, seed=303)
    obs = rng.standard_normal((100, 2))

    py_ll = KalmanFilter(model).filter(obs, backend="python").log_likelihood
    go_ll = KalmanFilter(model).filter(obs, backend="go").log_likelihood

    assert abs(go_ll - py_ll) < 1e-9, f"log-likelihood mismatch: python={py_ll}, go={go_ll}"


@pytest.mark.skipif(not _go_available(), reason="Go LGSSM backend unavailable")
def test_four_backend_parity_when_all_available() -> None:
    """Python = Rust = Julia = Go must agree to atol=1e-9 when all built."""
    rng = np.random.default_rng(304)
    model = LinearGaussianSSM.random(state_dim=3, obs_dim=2, control_dim=0, seed=304)
    obs = rng.standard_normal((40, 2))

    py = KalmanFilter(model).filter(obs, backend="python")
    go = KalmanFilter(model).filter(obs, backend="go")
    np.testing.assert_allclose(go.log_likelihood, py.log_likelihood, atol=1e-9)
    np.testing.assert_allclose(go.means, py.means, atol=1e-9)

    if _HAS_RUST_LGSSM:
        ru = KalmanFilter(model).filter(obs, backend="rust")
        np.testing.assert_allclose(go.means, ru.means, atol=1e-9)
        np.testing.assert_allclose(go.log_likelihood, ru.log_likelihood, atol=1e-9)

    if _julia_available():
        ju = KalmanFilter(model).filter(obs, backend="julia")
        np.testing.assert_allclose(go.means, ju.means, atol=1e-9)
        np.testing.assert_allclose(go.log_likelihood, ju.log_likelihood, atol=1e-9)


def test_go_backend_unavailable_raises_when_explicitly_requested() -> None:
    """If backend='go' is requested but unavailable, raise RuntimeError."""
    if _go_available():
        pytest.skip("Go backend IS available — cannot test the unavailable path")
    model = LinearGaussianSSM.random(state_dim=2, obs_dim=1, seed=1)
    obs = np.zeros((5, 1))
    with pytest.raises(RuntimeError, match="Go LGSSM backend"):
        KalmanFilter(model).filter(obs, backend="go")
