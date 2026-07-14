# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Structured forward Kalman filtering

"""Forward inference for validated linear Gaussian state-space models."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._lgssm_backends import filter_native, resolve_backend
from ._lgssm_types import (
    FilterResult,
    FloatArray,
    LinearGaussianSSM,
    _normalise_controls,
    _normalise_observations,
    _symmetrise,
)


def _solve_cholesky(lower: FloatArray, right_hand_side: FloatArray) -> FloatArray:
    intermediate = np.linalg.solve(lower, right_hand_side)
    return np.asarray(np.linalg.solve(lower.T, intermediate), dtype=np.float64)


class KalmanFilter:
    """Forward Kalman filter for a linear Gaussian state-space model.

    Parameters
    ----------
    model : LinearGaussianSSM
        Validated model parameters shared by the Python and native paths.

    """

    def __init__(self, model: LinearGaussianSSM) -> None:
        self.model = model

    def filter(
        self,
        observations: npt.ArrayLike,
        controls: npt.ArrayLike | None = None,
        backend: str = "auto",
    ) -> FilterResult:
        """Filter an observation sequence.

        Parameters
        ----------
        observations : array-like, shape (T, p)
            Finite observations ordered by time.
        controls : array-like, shape (T, m), optional
            Finite controls. They are required when ``m > 0`` and may be
            omitted only for a model with no control input.
        backend : {"auto", "mojo", "go", "rust", "julia", "python"}
            Execution backend. ``auto`` follows the availability- and
            initialisation-aware order Mojo, Go, Rust, Julia, then Python.

        Returns
        -------
        FilterResult
            Filtered and predicted moments plus the sequence log-likelihood.

        Raises
        ------
        ValueError
            If sequence shapes, values, or the backend name are invalid.
        RuntimeError
            If an explicitly requested native backend is unavailable.
        numpy.linalg.LinAlgError
            If an innovation covariance is not positive definite.

        """
        observation_array = _normalise_observations(
            observations,
            obs_dim=self.model.obs_dim,
        )
        control_array = _normalise_controls(
            controls,
            time_steps=observation_array.shape[0],
            control_dim=self.model.control_dim,
        )
        selected_backend = resolve_backend(backend)
        if selected_backend == "python":
            return self._filter_python(observation_array, control_array)
        return filter_native(selected_backend, self.model, observation_array, control_array)

    def _filter_python(
        self,
        observations: FloatArray,
        controls: FloatArray,
    ) -> FilterResult:
        time_steps, obs_dim = observations.shape
        state_dim = self.model.state_dim
        A, B, C, D = self.model.A, self.model.B, self.model.C, self.model.D
        Q, R = self.model.Q, self.model.R

        means = np.zeros((time_steps, state_dim), dtype=np.float64)
        covariances = np.zeros((time_steps, state_dim, state_dim), dtype=np.float64)
        pred_means = np.zeros((time_steps, state_dim), dtype=np.float64)
        pred_covariances = np.zeros((time_steps, state_dim, state_dim), dtype=np.float64)

        predicted_mean = self.model.mu_0.copy()
        predicted_covariance = self.model.Sigma_0.copy()
        identity = np.eye(state_dim, dtype=np.float64)
        gaussian_constant = obs_dim * np.log(2.0 * np.pi)
        log_likelihood = 0.0

        for time_index in range(time_steps):
            pred_means[time_index] = predicted_mean
            pred_covariances[time_index] = predicted_covariance

            control = controls[time_index]
            innovation = observations[time_index] - C @ predicted_mean - D @ control
            innovation_covariance = _symmetrise(C @ predicted_covariance @ C.T + R)
            try:
                lower = np.asarray(
                    np.linalg.cholesky(innovation_covariance),
                    dtype=np.float64,
                )
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "innovation covariance must be positive definite"
                ) from exc
            solved_innovation = _solve_cholesky(lower, innovation)
            log_determinant = 2.0 * float(np.sum(np.log(np.diag(lower))))
            log_likelihood -= 0.5 * (
                gaussian_constant + log_determinant + float(innovation @ solved_innovation)
            )

            covariance_observation = predicted_covariance @ C.T
            gain = _solve_cholesky(lower, covariance_observation.T).T
            filtered_mean = predicted_mean + gain @ innovation
            residual_operator = identity - gain @ C
            filtered_covariance = _symmetrise(
                residual_operator @ predicted_covariance @ residual_operator.T + gain @ R @ gain.T
            )

            means[time_index] = filtered_mean
            covariances[time_index] = filtered_covariance
            predicted_mean = A @ filtered_mean + B @ control
            predicted_covariance = _symmetrise(A @ filtered_covariance @ A.T + Q)

        return FilterResult(
            means=means,
            covariances=covariances,
            pred_means=pred_means,
            pred_covariances=pred_covariances,
            log_likelihood=log_likelihood,
        )
