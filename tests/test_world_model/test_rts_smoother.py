# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rauch-Tung-Striebel smoother tests

"""Backward-recursion and exact batch-conditioning tests for RTS smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.world_model._lgssm_types import FilterResult, FloatArray
from sc_neurocore.world_model.predictive_model import (
    KalmanFilter,
    LinearGaussianSSM,
    RTSSmoother,
)
from _predictive_model_test_support import scalar_random_walk


def test_last_smoothed_state_equals_last_filtered_state() -> None:
    model = scalar_random_walk()
    observations = np.random.default_rng(3).normal(size=(20, 1))
    filtered = KalmanFilter(model).filter(observations, backend="python")
    smoothed = RTSSmoother(model).smooth(filtered)

    np.testing.assert_allclose(smoothed.means[-1], filtered.means[-1])
    np.testing.assert_allclose(smoothed.covariances[-1], filtered.covariances[-1])


def test_smoothing_reduces_scalar_posterior_uncertainty() -> None:
    model = scalar_random_walk()
    observations = np.random.default_rng(4).normal(size=(30, 1))
    filtered = KalmanFilter(model).filter(observations, backend="python")
    smoothed = RTSSmoother(model).smooth(filtered)

    assert np.all(smoothed.covariances[:-1, 0, 0] <= filtered.covariances[:-1, 0, 0])


def _controlled_multivariate_model() -> LinearGaussianSSM:
    return LinearGaussianSSM(
        A=np.array([[0.82, 0.18], [-0.11, 0.73]]),
        B=np.array([[0.7], [-0.4]]),
        C=np.array([[1.0, 0.35], [-0.25, 0.9]]),
        D=np.array([[0.2], [0.5]]),
        Q=np.array([[0.07, 0.015], [0.015, 0.05]]),
        R=np.array([[0.09, 0.01], [0.01, 0.08]]),
        mu_0=np.array([0.1, -0.2]),
        Sigma_0=np.array([[0.5, 0.08], [0.08, 0.4]]),
    )


def _joint_state_moments(
    model: LinearGaussianSSM,
    controls: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    time_steps = controls.shape[0]
    state_dim = model.state_dim
    state_mean = np.zeros((time_steps, state_dim), dtype=np.float64)
    state_mean[0] = model.mu_0
    transform = np.zeros(
        (time_steps * state_dim, time_steps * state_dim),
        dtype=np.float64,
    )
    transform[:state_dim, :state_dim] = np.eye(state_dim)
    for time_index in range(1, time_steps):
        previous = transform[(time_index - 1) * state_dim : time_index * state_dim,]
        current_slice = slice(time_index * state_dim, (time_index + 1) * state_dim)
        transform[current_slice] = model.A @ previous
        transform[current_slice, current_slice] = np.eye(state_dim)
        state_mean[time_index] = (
            model.A @ state_mean[time_index - 1] + model.B @ controls[time_index - 1]
        )

    independent_covariance = np.zeros_like(transform)
    independent_covariance[:state_dim, :state_dim] = model.Sigma_0
    for time_index in range(1, time_steps):
        block = slice(time_index * state_dim, (time_index + 1) * state_dim)
        independent_covariance[block, block] = model.Q
    state_covariance = transform @ independent_covariance @ transform.T
    return state_mean, state_covariance


def _condition_full_state_sequence(
    model: LinearGaussianSSM,
    controls: FloatArray,
    observations: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    time_steps = observations.shape[0]
    state_mean, state_covariance = _joint_state_moments(model, controls)
    observation_operator = np.kron(np.eye(time_steps), model.C)
    observation_mean = (state_mean @ model.C.T + controls @ model.D.T).reshape(-1)
    observation_covariance = (
        observation_operator @ state_covariance @ observation_operator.T
        + np.kron(np.eye(time_steps), model.R)
    )
    state_observation_covariance = state_covariance @ observation_operator.T
    innovation = observations.reshape(-1) - observation_mean
    posterior_mean = state_mean.reshape(-1) + state_observation_covariance @ np.linalg.solve(
        observation_covariance,
        innovation,
    )
    posterior_covariance = state_covariance - state_observation_covariance @ np.linalg.solve(
        observation_covariance,
        state_observation_covariance.T,
    )
    return posterior_mean.reshape(time_steps, model.state_dim), posterior_covariance


def test_multivariate_smoother_matches_exact_batch_conditioning() -> None:
    model = _controlled_multivariate_model()
    controls = np.array([[0.3], [-0.4], [0.8], [0.1]])
    observations = np.array([[0.2, -0.1], [0.7, -0.3], [-0.2, 0.9], [0.4, 0.1]])
    filtered = KalmanFilter(model).filter(
        observations,
        controls,
        backend="python",
    )
    smoothed = RTSSmoother(model).smooth(filtered)
    expected_means, expected_covariance = _condition_full_state_sequence(
        model,
        controls,
        observations,
    )

    np.testing.assert_allclose(smoothed.means, expected_means, atol=1e-11)
    for time_index in range(observations.shape[0]):
        block = slice(time_index * model.state_dim, (time_index + 1) * model.state_dim)
        np.testing.assert_allclose(
            smoothed.covariances[time_index],
            expected_covariance[block, block],
            atol=1e-11,
        )
    for time_index in range(observations.shape[0] - 1):
        current = slice(time_index * model.state_dim, (time_index + 1) * model.state_dim)
        following = slice(
            (time_index + 1) * model.state_dim,
            (time_index + 2) * model.state_dim,
        )
        np.testing.assert_allclose(
            smoothed.cross_covariances[time_index],
            expected_covariance[current, following],
            atol=1e-11,
        )


def test_single_step_smoothing_has_empty_lag_axis() -> None:
    model = scalar_random_walk()
    filtered = KalmanFilter(model).filter(np.array([[0.5]]), backend="python")
    smoothed = RTSSmoother(model).smooth(filtered)
    assert smoothed.cross_covariances.shape == (0, 1, 1)


def test_smoother_rejects_result_from_different_state_dimension() -> None:
    filtered = KalmanFilter(scalar_random_walk()).filter(
        np.array([[0.5], [0.6]]),
        backend="python",
    )
    with pytest.raises(ValueError, match="does not match model state dimension"):
        RTSSmoother(LinearGaussianSSM.random(2, 1)).smooth(filtered)


def test_smoother_fails_closed_on_singular_predicted_covariance() -> None:
    filtered = FilterResult(
        means=np.zeros((2, 1)),
        covariances=np.ones((2, 1, 1)),
        pred_means=np.zeros((2, 1)),
        pred_covariances=np.array([[[1.0]], [[0.0]]]),
        log_likelihood=-1.0,
    )
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        RTSSmoother(scalar_random_walk()).smooth(filtered)
