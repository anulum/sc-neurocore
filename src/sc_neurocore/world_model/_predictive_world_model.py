# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive world-model compatibility API

"""Validated state-transition forecasts for the historical world-model API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._lgssm_types import (
    FloatArray,
    LinearGaussianSSM,
    _normalise_state_covariance,
    _normalise_vector,
    _require_dimension,
    _symmetrise,
)


@dataclass
class PredictiveWorldModel:
    """Forecast latent-state means and covariances through an LGSSM.

    Parameters
    ----------
    state_dim : int
        Positive latent-state dimension.
    action_dim : int
        Non-negative action dimension.
    seed : int, default=42
        Seed used to initialise the stable random LGSSM.

    Notes
    -----
    This class preserves the historical planning-facing API. Use
    :class:`LinearGaussianSSM`, :class:`KalmanFilter`, and
    :class:`RTSSmoother` when observations are available.

    """

    state_dim: int
    action_dim: int
    seed: int = 42

    def __post_init__(self) -> None:
        """Initialise a validated stable state-transition model."""
        self.state_dim = _require_dimension(
            self.state_dim,
            name="state_dim",
            allow_zero=False,
        )
        self.action_dim = _require_dimension(
            self.action_dim,
            name="action_dim",
            allow_zero=True,
        )
        self.model = LinearGaussianSSM.random(
            state_dim=self.state_dim,
            obs_dim=self.state_dim,
            control_dim=self.action_dim,
            seed=self.seed,
        )
        self._mu = self.model.mu_0.copy()
        self._Sigma = self.model.Sigma_0.copy()

    def reset(self) -> None:
        """Reset the stored belief moments to the model prior."""
        self._mu = self.model.mu_0.copy()
        self._Sigma = self.model.Sigma_0.copy()

    def predict_next_state(
        self,
        current_state: npt.ArrayLike,
        action: npt.ArrayLike,
    ) -> FloatArray:
        """Predict the next latent-state mean.

        Parameters
        ----------
        current_state : array-like, shape (state_dim,)
            Current state estimate.
        action : array-like, shape (action_dim,)
            Current control input. A scalar is accepted when ``action_dim=1``.

        Returns
        -------
        numpy.ndarray, shape (state_dim,)
            Conditional mean ``A x_t + B u_t``.

        Raises
        ------
        ValueError
            If an input has an incompatible shape or non-finite value.

        """
        state = _normalise_vector(
            current_state,
            name="current_state",
            length=self.state_dim,
        )
        control = _normalise_vector(
            action,
            name="action",
            length=self.action_dim,
            allow_scalar=True,
        )
        return np.asarray(self.model.A @ state + self.model.B @ control)

    def predict_next_state_with_cov(
        self,
        current_state: npt.ArrayLike,
        current_cov: npt.ArrayLike,
        action: npt.ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Predict the next latent-state mean and covariance.

        Parameters
        ----------
        current_state : array-like, shape (state_dim,)
            Current state estimate.
        current_cov : array-like, shape (state_dim, state_dim)
            Symmetric positive-semidefinite current covariance.
        action : array-like, shape (action_dim,)
            Current control input.

        Returns
        -------
        tuple of numpy.ndarray
            Mean ``A x_t + B u_t`` and covariance ``A P_t A^T + Q``.

        """
        state_covariance = _normalise_state_covariance(
            current_cov,
            state_dim=self.state_dim,
            name="current_cov",
        )
        mean = self.predict_next_state(current_state, action)
        covariance = _symmetrise(self.model.A @ state_covariance @ self.model.A.T + self.model.Q)
        return mean, covariance

    def forecast(
        self,
        initial_state: npt.ArrayLike,
        actions: list[npt.ArrayLike],
    ) -> list[FloatArray]:
        """Forecast a deterministic mean trajectory.

        Parameters
        ----------
        initial_state : array-like, shape (state_dim,)
            State before the first action.
        actions : list of array-like
            Ordered actions, one per returned state.

        Returns
        -------
        list of numpy.ndarray
            Independent state arrays after each action.

        """
        state = _normalise_vector(
            initial_state,
            name="initial_state",
            length=self.state_dim,
        )
        trajectory: list[FloatArray] = []
        for action in actions:
            state = self.predict_next_state(state, action)
            trajectory.append(state.copy())
        return trajectory

    def forecast_with_cov(
        self,
        initial_state: npt.ArrayLike,
        initial_cov: npt.ArrayLike,
        actions: list[npt.ArrayLike],
    ) -> list[tuple[FloatArray, FloatArray]]:
        """Forecast a mean and covariance trajectory.

        Parameters
        ----------
        initial_state : array-like, shape (state_dim,)
            State before the first action.
        initial_cov : array-like, shape (state_dim, state_dim)
            Initial symmetric positive-semidefinite covariance.
        actions : list of array-like
            Ordered actions, one per returned state.

        Returns
        -------
        list of tuple of numpy.ndarray
            Independent ``(mean, covariance)`` pairs after each action.

        """
        state = _normalise_vector(
            initial_state,
            name="initial_state",
            length=self.state_dim,
        )
        covariance = _normalise_state_covariance(
            initial_cov,
            state_dim=self.state_dim,
            name="initial_cov",
        )
        trajectory: list[tuple[FloatArray, FloatArray]] = []
        for action in actions:
            state, covariance = self.predict_next_state_with_cov(
                state,
                covariance,
                action,
            )
            trajectory.append((state.copy(), covariance.copy()))
        return trajectory
