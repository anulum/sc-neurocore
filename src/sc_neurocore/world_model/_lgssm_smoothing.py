# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rauch-Tung-Striebel smoothing

"""Backward smoothing for linear Gaussian state-space filter results."""

from __future__ import annotations

import numpy as np

from ._lgssm_types import (
    FilterResult,
    LinearGaussianSSM,
    SmoothResult,
    _solve_positive_definite,
    _symmetrise,
)


class RTSSmoother:
    """Rauch-Tung-Striebel backward smoother.

    Parameters
    ----------
    model : LinearGaussianSSM
        Model used to produce the corresponding forward-filter result.

    Notes
    -----
    The recursion follows Rauch, Tung, and Striebel (1965). The returned
    lag-one covariance is oriented as ``Cov[x_t, x_{t+1} | y]``.

    """

    def __init__(self, model: LinearGaussianSSM) -> None:
        self.model = model

    def smooth(self, filter_result: FilterResult) -> SmoothResult:
        """Smooth every state in a validated forward-filter result.

        Parameters
        ----------
        filter_result : FilterResult
            Forward moments for at least one time step.

        Returns
        -------
        SmoothResult
            Full-sequence posterior means, covariances, and lag-one
            cross-covariances.

        Raises
        ------
        ValueError
            If the filter-result state dimension differs from the model.
        numpy.linalg.LinAlgError
            If a predicted covariance required by the recursion is singular.

        """
        time_steps, state_dim = filter_result.means.shape
        if state_dim != self.model.state_dim:
            raise ValueError(
                "filter_result state dimension "
                f"{state_dim} does not match model state dimension {self.model.state_dim}"
            )

        smoothed_means = filter_result.means.copy()
        smoothed_covariances = filter_result.covariances.copy()
        cross_covariances = np.zeros(
            (time_steps - 1, state_dim, state_dim),
            dtype=np.float64,
        )

        for time_index in range(time_steps - 2, -1, -1):
            predicted_next = filter_result.pred_covariances[time_index + 1]
            covariance_transition = filter_result.covariances[time_index] @ self.model.A.T
            smoothing_gain = _solve_positive_definite(
                predicted_next,
                covariance_transition.T,
            ).T
            smoothed_means[time_index] = filter_result.means[time_index] + smoothing_gain @ (
                smoothed_means[time_index + 1] - filter_result.pred_means[time_index + 1]
            )
            smoothed_covariances[time_index] = _symmetrise(
                filter_result.covariances[time_index]
                + smoothing_gain
                @ (smoothed_covariances[time_index + 1] - predicted_next)
                @ smoothing_gain.T
            )
            cross_covariances[time_index] = smoothing_gain @ smoothed_covariances[time_index + 1]

        return SmoothResult(
            means=smoothed_means,
            covariances=smoothed_covariances,
            cross_covariances=cross_covariances,
        )
