# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Structured Kalman-filter tests

"""Inference, validation, numerical, and installed-backend Kalman contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.world_model import _lgssm_backends as backends
from sc_neurocore.world_model._lgssm_backends import ExplicitBackendName
from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import KalmanFilter, LinearGaussianSSM
from _predictive_model_test_support import (
    controlled_scalar_model,
    scalar_random_walk,
    simulate_model,
)


def test_filter_returns_finite_moments_and_likelihood() -> None:
    model = LinearGaussianSSM.random(3, 2, seed=11)
    observations = np.random.default_rng(11).normal(size=(30, 2))
    result = KalmanFilter(model).filter(observations, backend="python")

    assert result.means.shape == (30, 3)
    assert result.covariances.shape == (30, 3, 3)
    assert result.pred_means.shape == (30, 3)
    assert result.pred_covariances.shape == (30, 3, 3)
    assert np.isfinite(result.log_likelihood)


def test_scalar_first_update_matches_closed_form_gaussian_conditioning() -> None:
    model = scalar_random_walk()
    result = KalmanFilter(model).filter(np.array([[2.0]]), backend="python")

    np.testing.assert_allclose(result.means, [[1.0]])
    np.testing.assert_allclose(result.covariances, [[[0.5]]])
    np.testing.assert_allclose(result.pred_means, [[0.0]])
    np.testing.assert_allclose(result.pred_covariances, [[[1.0]]])
    expected = -0.5 * (np.log(4.0 * np.pi) + 2.0)
    assert result.log_likelihood == pytest.approx(expected)


def test_low_measurement_noise_tracks_observations() -> None:
    model = LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.zeros((1, 0)),
        C=np.array([[1.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.01]]),
        R=np.array([[1e-6]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    observations = np.arange(1.0, 5.0)[:, None]
    result = KalmanFilter(model).filter(observations, backend="python")

    np.testing.assert_allclose(result.means[:, 0], observations[:, 0], atol=1e-3)


def test_high_measurement_noise_preserves_prior_scale() -> None:
    model = LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.zeros((1, 0)),
        C=np.array([[1.0]]),
        D=np.zeros((1, 0)),
        Q=np.array([[0.01]]),
        R=np.array([[1e10]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[0.01]]),
    )
    observations = np.array([[100.0], [-100.0], [50.0], [-50.0]])
    result = KalmanFilter(model).filter(observations, backend="python")

    assert np.all(np.abs(result.means) < 1.0)


def test_control_terms_affect_observation_and_transition_predictions() -> None:
    model = LinearGaussianSSM(
        A=np.array([[1.0]]),
        B=np.array([[2.0]]),
        C=np.array([[1.0]]),
        D=np.array([[3.0]]),
        Q=np.array([[0.1]]),
        R=np.array([[0.1]]),
        mu_0=np.array([0.0]),
        Sigma_0=np.array([[1.0]]),
    )
    controls = np.ones((3, 1))
    observations = np.full((3, 1), 3.0)
    result = KalmanFilter(model).filter(
        observations,
        controls,
        backend="python",
    )

    assert result.means[0, 0] == pytest.approx(0.0)
    assert result.pred_means[1, 0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (np.zeros(3), "2-dimensional"),
        (np.zeros((0, 1)), "at least one time step"),
        (np.zeros((3, 2)), "shape"),
        (np.array([[np.nan]]), "finite"),
    ],
)
def test_filter_rejects_invalid_observation_sequences(
    observations: FloatArray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        KalmanFilter(scalar_random_walk()).filter(observations, backend="python")


def test_filter_rejects_missing_malformed_and_non_finite_controls() -> None:
    model = controlled_scalar_model()
    observations = np.zeros((3, 1))
    with pytest.raises(ValueError, match="controls must have shape"):
        KalmanFilter(model).filter(observations, backend="python")
    with pytest.raises(ValueError, match="controls must have shape"):
        KalmanFilter(model).filter(observations, np.zeros((2, 1)), backend="python")
    with pytest.raises(ValueError, match="finite"):
        KalmanFilter(model).filter(
            observations,
            np.full((3, 1), np.inf),
            backend="python",
        )


def test_no_control_model_rejects_non_empty_control_matrix() -> None:
    with pytest.raises(ValueError, match="controls must have shape"):
        KalmanFilter(scalar_random_walk()).filter(
            np.zeros((2, 1)),
            np.zeros((2, 1)),
            backend="python",
        )


def test_filter_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        KalmanFilter(scalar_random_walk()).filter(np.zeros((2, 1)), backend="cuda")


def test_filter_fails_closed_if_model_is_corrupted_after_validation() -> None:
    model = scalar_random_walk()
    model.R = np.array([[-2.0]])
    with pytest.raises(np.linalg.LinAlgError, match="innovation covariance"):
        KalmanFilter(model).filter(np.zeros((1, 1)), backend="python")


def test_filtered_and_predicted_covariances_remain_symmetric_psd() -> None:
    model = LinearGaussianSSM.random(4, 3, seed=22)
    observations = np.random.default_rng(22).normal(size=(80, 3))
    result = KalmanFilter(model).filter(observations, backend="python")

    for covariance in np.concatenate(
        (result.covariances, result.pred_covariances),
        axis=0,
    ):
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        assert float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-12


def test_each_installed_native_backend_matches_controlled_python_filter() -> None:
    model = controlled_scalar_model()
    rng = np.random.default_rng(510)
    controls = rng.normal(size=(40, 1))
    _, observations = simulate_model(
        model,
        time_steps=40,
        seed=511,
        controls=controls,
    )
    reference = KalmanFilter(model).filter(
        observations,
        controls,
        backend="python",
    )

    exercised: list[str] = []
    for backend in ("mojo", "rust", "julia", "go"):
        available, _ = backends.probe_backend(backend)
        if not available:
            continue
        candidate = KalmanFilter(model).filter(observations, controls, backend=backend)
        exercised.append(backend)
        np.testing.assert_allclose(candidate.means, reference.means, atol=1e-9)
        np.testing.assert_allclose(candidate.covariances, reference.covariances, atol=1e-9)
        assert candidate.log_likelihood == pytest.approx(
            reference.log_likelihood,
            abs=1e-7,
        )

    assert exercised, "focused environment must provide at least one native LGSSM backend"


def test_all_backends_handle_observation_dimension_wider_than_state() -> None:
    model = LinearGaussianSSM(
        A=np.array([[0.8]]),
        B=np.array([[0.3]]),
        C=np.array([[1.0], [0.5], [-0.25]]),
        D=np.array([[0.2], [-0.1], [0.4]]),
        Q=np.array([[0.04]]),
        R=np.array(
            [
                [0.20, 0.02, 0.01],
                [0.02, 0.25, -0.01],
                [0.01, -0.01, 0.30],
            ],
        ),
        mu_0=np.array([0.1]),
        Sigma_0=np.array([[0.4]]),
    )
    controls = np.random.default_rng(601).normal(size=(24, 1))
    _, observations = simulate_model(
        model,
        time_steps=24,
        seed=602,
        controls=controls,
    )
    reference = KalmanFilter(model).filter(observations, controls, backend="python")
    native_backends: tuple[ExplicitBackendName, ...] = (
        "mojo",
        "rust",
        "julia",
        "go",
    )
    unavailable = {
        backend: reason
        for backend in native_backends
        for available, reason in [backends.probe_backend(backend)]
        if not available
    }
    assert unavailable == {}, unavailable

    for backend in native_backends:
        candidate = KalmanFilter(model).filter(observations, controls, backend=backend)
        np.testing.assert_allclose(candidate.means, reference.means, atol=1e-9)
        np.testing.assert_allclose(candidate.covariances, reference.covariances, atol=1e-9)
        np.testing.assert_allclose(candidate.pred_means, reference.pred_means, atol=1e-9)
        np.testing.assert_allclose(
            candidate.pred_covariances,
            reference.pred_covariances,
            atol=1e-9,
        )
        assert candidate.log_likelihood == pytest.approx(
            reference.log_likelihood,
            abs=1e-7,
        )
