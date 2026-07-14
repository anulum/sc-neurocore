# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Controlled LGSSM expectation-maximisation

"""Expectation-maximisation for linear Gaussian state-space parameters."""

from __future__ import annotations

from math import isfinite
from numbers import Integral

import numpy as np
import numpy.typing as npt

from ._lgssm_filter import KalmanFilter
from ._lgssm_smoothing import RTSSmoother
from ._lgssm_types import (
    FloatArray,
    LinearGaussianSSM,
    SmoothResult,
    _normalise_controls,
    _normalise_observations,
    _right_solve_positive_definite,
    _stabilise_covariance,
)

_LIKELIHOOD_ROUNDOFF_MULTIPLIER = 256.0


def _likelihood_tolerance(previous: float) -> float:
    return float(
        _LIKELIHOOD_ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * max(1.0, abs(previous))
    )


def _expected_state_products(smoothed: SmoothResult) -> tuple[FloatArray, FloatArray]:
    means = smoothed.means
    second_moments = smoothed.covariances + np.einsum(
        "ti,tj->tij",
        means,
        means,
    )
    # SmoothResult stores Cov[x_t, x_{t+1}]. The transition M-step needs
    # E[x_{t+1} x_t^T], hence the explicit transpose of each lag block.
    next_current = smoothed.cross_covariances.transpose(0, 2, 1) + np.einsum(
        "ti,tj->tij",
        means[1:],
        means[:-1],
    )
    return np.asarray(second_moments), np.asarray(next_current)


def _transition_update(
    model: LinearGaussianSSM,
    controls: FloatArray,
    smoothed: SmoothResult,
) -> tuple[FloatArray, FloatArray]:
    means = smoothed.means
    second_moments, next_current = _expected_state_products(smoothed)
    offsets = controls[:-1] @ model.B.T
    target_current = next_current - np.einsum(
        "ti,tj->tij",
        offsets,
        means[:-1],
    )
    target_second = (
        second_moments[1:]
        - np.einsum("ti,tj->tij", offsets, means[1:])
        - np.einsum("ti,tj->tij", means[1:], offsets)
        + np.einsum("ti,tj->tij", offsets, offsets)
    )

    current_second_sum = np.sum(second_moments[:-1], axis=0)
    transition = _right_solve_positive_definite(
        current_second_sum,
        np.sum(target_current, axis=0),
    )

    process_accumulator = np.zeros_like(model.Q)
    for time_index in range(target_current.shape[0]):
        process_accumulator += (
            target_second[time_index]
            - target_current[time_index] @ transition.T
            - transition @ target_current[time_index].T
            + transition @ second_moments[time_index] @ transition.T
        )
    process_covariance = _stabilise_covariance(
        process_accumulator / target_current.shape[0],
        positive_definite=False,
    )
    return transition, process_covariance


def _observation_update(
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
    smoothed: SmoothResult,
) -> tuple[FloatArray, FloatArray]:
    adjusted_observations = observations - controls @ model.D.T
    means = smoothed.means
    second_moments, _ = _expected_state_products(smoothed)
    observation = _right_solve_positive_definite(
        np.sum(second_moments, axis=0),
        adjusted_observations.T @ means,
    )
    residuals = adjusted_observations - means @ observation.T
    observation_covariance = residuals.T @ residuals
    for covariance in smoothed.covariances:
        observation_covariance += observation @ covariance @ observation.T
    observation_covariance = _stabilise_covariance(
        observation_covariance / observations.shape[0],
        positive_definite=True,
    )
    return observation, observation_covariance


def _maximisation_step(
    model: LinearGaussianSSM,
    observations: FloatArray,
    controls: FloatArray,
    smoothed: SmoothResult,
) -> LinearGaussianSSM:
    transition, process_covariance = _transition_update(model, controls, smoothed)
    observation, observation_covariance = _observation_update(
        model,
        observations,
        controls,
        smoothed,
    )
    return LinearGaussianSSM(
        A=transition,
        B=model.B,
        C=observation,
        D=model.D,
        Q=process_covariance,
        R=observation_covariance,
        mu_0=smoothed.means[0],
        Sigma_0=_stabilise_covariance(
            smoothed.covariances[0],
            positive_definite=True,
        ),
    )


class EMLearner:
    """Estimate selected LGSSM parameters by expectation-maximisation.

    Parameters
    ----------
    max_iter : int, default=50
        Positive maximum number of E/M iterations.
    tol : float, default=1e-4
        Non-negative absolute log-likelihood convergence threshold.

    Notes
    -----
    The M-step updates ``A``, ``C``, ``Q``, ``R``, ``mu_0``, and
    ``Sigma_0``. ``B`` and ``D`` are treated as known, but their control
    contributions are subtracted from the transition and observation
    sufficient statistics as required by Shumway and Stoffer (1982).

    Raises
    ------
    ValueError
        If ``max_iter`` or ``tol`` is outside its documented domain.

    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-4) -> None:
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
            raise ValueError("max_iter must be a positive integer")
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        self.log_likelihood_history: list[float] = []

    def fit(
        self,
        observations: npt.ArrayLike,
        initial_model: LinearGaussianSSM,
        controls: npt.ArrayLike | None = None,
        backend: str = "python",
    ) -> LinearGaussianSSM:
        """Estimate model parameters from one observation sequence.

        Parameters
        ----------
        observations : array-like, shape (T, p)
            Finite observation sequence with ``T >= 2``.
        initial_model : LinearGaussianSSM
            Starting parameters. The returned model preserves its ``B`` and
            ``D`` values exactly.
        controls : array-like, shape (T, m), optional
            Controls paired with the observations.
        backend : {"auto", "mojo", "go", "rust", "julia", "python"}, default="python"
            Forward-filter backend used in each E-step. The deterministic
            default keeps the entire learning benchmark on the Python path.

        Returns
        -------
        LinearGaussianSSM
            Last accepted M-step model.

        Raises
        ------
        ValueError
            If data shapes or values are invalid or fewer than two samples
            are supplied.
        RuntimeError
            If a likelihood decrease exceeds accumulated float64 round-off.

        """
        observation_array = _normalise_observations(
            observations,
            obs_dim=initial_model.obs_dim,
        )
        if observation_array.shape[0] < 2:
            raise ValueError("EM requires at least two observation time steps")
        control_array = _normalise_controls(
            controls,
            time_steps=observation_array.shape[0],
            control_dim=initial_model.control_dim,
        )

        model = initial_model
        filter_result = KalmanFilter(model).filter(
            observation_array,
            controls=control_array,
            backend=backend,
        )
        previous_likelihood = filter_result.log_likelihood
        self.log_likelihood_history = [previous_likelihood]

        for _ in range(self.max_iter):
            smoothed = RTSSmoother(model).smooth(filter_result)
            candidate = _maximisation_step(
                model,
                observation_array,
                control_array,
                smoothed,
            )
            candidate_result = KalmanFilter(candidate).filter(
                observation_array,
                controls=control_array,
                backend=backend,
            )
            candidate_likelihood = candidate_result.log_likelihood
            self.log_likelihood_history.append(candidate_likelihood)
            decrease = previous_likelihood - candidate_likelihood
            if decrease > _likelihood_tolerance(previous_likelihood):
                raise RuntimeError(
                    "EM log-likelihood decreased beyond float64 round-off: "
                    f"{previous_likelihood} -> {candidate_likelihood}"
                )

            model = candidate
            if abs(candidate_likelihood - previous_likelihood) <= self.tol:
                break
            filter_result = candidate_result
            previous_likelihood = candidate_likelihood

        return model
