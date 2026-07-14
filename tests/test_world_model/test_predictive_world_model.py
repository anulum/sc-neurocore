# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive world-model compatibility tests

"""Planning-facing state and covariance forecast contracts."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import PredictiveWorldModel


@pytest.mark.parametrize(
    ("state_dim", "action_dim", "message"),
    [
        (0, 1, "state_dim must be positive"),
        (2, -1, "action_dim must be non-negative"),
        (True, 1, "state_dim must be an integer"),
    ],
)
def test_world_model_rejects_invalid_dimensions(
    state_dim: int,
    action_dim: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PredictiveWorldModel(state_dim, action_dim)


def test_mean_prediction_obeys_state_and_control_dynamics() -> None:
    world_model = PredictiveWorldModel(3, 2, seed=71)
    state = np.array([0.2, -0.1, 0.5])
    action = np.array([0.4, -0.3])

    predicted = world_model.predict_next_state(state, action)

    np.testing.assert_allclose(
        predicted,
        world_model.model.A @ state + world_model.model.B @ action,
    )
    assert predicted.shape == (3,)


def test_scalar_action_is_accepted_only_for_one_control_dimension() -> None:
    world_model = PredictiveWorldModel(2, 1, seed=1)
    scalar_result = world_model.predict_next_state(np.array([1.0, -1.0]), np.array(0.5))
    vector_result = world_model.predict_next_state(
        np.array([1.0, -1.0]),
        np.array([0.5]),
    )
    np.testing.assert_allclose(scalar_result, vector_result)

    with pytest.raises(ValueError, match="action must have shape"):
        PredictiveWorldModel(2, 2).predict_next_state(np.zeros(2), np.array(0.5))


@pytest.mark.parametrize(
    ("state", "action", "message"),
    [
        (np.zeros(3), np.zeros(1), "current_state must have shape"),
        (np.zeros(2), np.zeros(2), "action must have shape"),
        (np.array([np.nan, 0.0]), np.zeros(1), "finite"),
        (np.zeros(2), np.array([np.inf]), "finite"),
    ],
)
def test_mean_prediction_rejects_invalid_inputs(
    state: FloatArray,
    action: FloatArray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PredictiveWorldModel(2, 1).predict_next_state(state, action)


def test_mean_prediction_rejects_non_numeric_vectors() -> None:
    invalid = cast(FloatArray, np.array(["not", "numeric"]))
    with pytest.raises(ValueError, match="real numeric"):
        PredictiveWorldModel(2, 1).predict_next_state(invalid, np.zeros(1))


def test_mean_prediction_rejects_vectors_outside_float64_range() -> None:
    invalid = cast(FloatArray, [10**10_000, 0])
    with pytest.raises(ValueError, match="real numeric"):
        PredictiveWorldModel(2, 1).predict_next_state(invalid, np.zeros(1))


def test_covariance_prediction_matches_linear_gaussian_recursion() -> None:
    world_model = PredictiveWorldModel(2, 1, seed=2)
    state = np.array([0.1, -0.2])
    covariance = np.array([[0.3, 0.04], [0.04, 0.2]])
    mean, predicted_covariance = world_model.predict_next_state_with_cov(
        state,
        covariance,
        np.array([0.5]),
    )

    np.testing.assert_allclose(mean, world_model.predict_next_state(state, np.array([0.5])))
    np.testing.assert_allclose(
        predicted_covariance,
        world_model.model.A @ covariance @ world_model.model.A.T + world_model.model.Q,
    )
    np.testing.assert_allclose(predicted_covariance, predicted_covariance.T)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.eye(3), "current_cov must have shape"),
        (np.array([[1.0, 0.2], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite"),
    ],
)
def test_covariance_prediction_rejects_invalid_covariance(
    covariance: FloatArray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PredictiveWorldModel(2, 1).predict_next_state_with_cov(
            np.zeros(2),
            covariance,
            np.zeros(1),
        )


def test_forecast_returns_independent_state_arrays() -> None:
    world_model = PredictiveWorldModel(3, 1, seed=3)
    trajectory = world_model.forecast(
        np.zeros(3),
        [np.zeros(1), np.ones(1), np.full(1, 0.5)],
    )

    assert len(trajectory) == 3
    assert all(state.shape == (3,) for state in trajectory)
    original_second = trajectory[1].copy()
    trajectory[0][0] = 99.0
    np.testing.assert_array_equal(trajectory[1], original_second)


def test_covariance_forecast_returns_independent_pairs() -> None:
    world_model = PredictiveWorldModel(2, 0, seed=4)
    trajectory = world_model.forecast_with_cov(
        np.zeros(2),
        np.eye(2),
        [np.zeros(0), np.zeros(0)],
    )

    assert len(trajectory) == 2
    assert trajectory[0][0].shape == (2,)
    assert trajectory[0][1].shape == (2, 2)
    original_second = trajectory[1][1].copy()
    trajectory[0][1][0, 0] = 99.0
    np.testing.assert_array_equal(trajectory[1][1], original_second)


def test_reset_restores_independent_prior_copies() -> None:
    world_model = PredictiveWorldModel(2, 0, seed=5)
    world_model._mu[:] = 99.0
    world_model._Sigma[:] = 100.0
    world_model.reset()

    np.testing.assert_array_equal(world_model._mu, world_model.model.mu_0)
    np.testing.assert_array_equal(world_model._Sigma, world_model.model.Sigma_0)
    assert not np.shares_memory(world_model._mu, world_model.model.mu_0)
    assert not np.shares_memory(world_model._Sigma, world_model.model.Sigma_0)
