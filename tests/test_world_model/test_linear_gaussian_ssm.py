# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian state-space data-contract tests

"""Shape, value, covariance, and result contracts for LGSSM data types."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import (
    FilterResult,
    LinearGaussianSSM,
    SmoothResult,
)


def _model(
    *,
    A: FloatArray | None = None,
    B: FloatArray | None = None,
    C: FloatArray | None = None,
    D: FloatArray | None = None,
    Q: FloatArray | None = None,
    R: FloatArray | None = None,
    mu_0: FloatArray | None = None,
    Sigma_0: FloatArray | None = None,
) -> LinearGaussianSSM:
    return LinearGaussianSSM(
        A=np.eye(2) if A is None else A,
        B=np.zeros((2, 1)) if B is None else B,
        C=np.eye(2) if C is None else C,
        D=np.zeros((2, 1)) if D is None else D,
        Q=np.eye(2) if Q is None else Q,
        R=np.eye(2) if R is None else R,
        mu_0=np.zeros(2) if mu_0 is None else mu_0,
        Sigma_0=np.eye(2) if Sigma_0 is None else Sigma_0,
    )


def test_model_copies_parameters_to_finite_contiguous_float64() -> None:
    transition = np.eye(2, dtype=np.float32)
    model = _model(A=cast(FloatArray, transition))
    transition[0, 0] = 9.0

    assert model.A.dtype == np.float64
    assert model.A.flags.c_contiguous
    assert model.A[0, 0] == 1.0
    assert (model.state_dim, model.obs_dim, model.control_dim) == (2, 2, 1)


@pytest.mark.parametrize(
    ("factory", "malformed", "message"),
    [
        (lambda value: _model(A=value), np.zeros((2, 3)), "A must be a non-empty square"),
        (lambda value: _model(B=value), np.zeros((3, 1)), "B must have 2 rows"),
        (lambda value: _model(C=value), np.zeros((2, 3)), "C must have shape"),
        (lambda value: _model(D=value), np.zeros((2, 2)), "D must have shape"),
        (lambda value: _model(Q=value), np.eye(3), "Q must have shape"),
        (lambda value: _model(R=value), np.eye(3), "R must have shape"),
        (lambda value: _model(mu_0=value), np.zeros(3), "mu_0 must have shape"),
        (lambda value: _model(Sigma_0=value), np.eye(3), "Sigma_0 must have shape"),
    ],
)
def test_model_rejects_each_malformed_parameter_shape(
    factory: Callable[[FloatArray], LinearGaussianSSM],
    malformed: FloatArray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory(malformed)


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (lambda value: _model(A=value), np.full((2, 2), np.nan)),
        (lambda value: _model(B=value), np.full((2, 1), np.nan)),
        (lambda value: _model(C=value), np.full((2, 2), np.nan)),
        (lambda value: _model(D=value), np.full((2, 1), np.nan)),
        (lambda value: _model(Q=value), np.full((2, 2), np.nan)),
        (lambda value: _model(R=value), np.full((2, 2), np.nan)),
        (lambda value: _model(mu_0=value), np.full(2, np.nan)),
        (lambda value: _model(Sigma_0=value), np.full((2, 2), np.nan)),
    ],
)
def test_model_rejects_non_finite_parameters(
    factory: Callable[[FloatArray], LinearGaussianSSM],
    value: FloatArray,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        factory(value)


def test_model_rejects_non_numeric_parameters() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        LinearGaussianSSM(
            A=cast(FloatArray, np.array([["bad"]])),
            B=np.zeros((1, 0)),
            C=np.ones((1, 1)),
            D=np.zeros((1, 0)),
            Q=np.ones((1, 1)),
            R=np.ones((1, 1)),
            mu_0=np.zeros(1),
            Sigma_0=np.ones((1, 1)),
        )


def test_model_rejects_values_outside_float64_range() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        LinearGaussianSSM(
            A=cast(FloatArray, [[10**10_000]]),
            B=np.zeros((1, 0)),
            C=np.ones((1, 1)),
            D=np.zeros((1, 0)),
            Q=np.ones((1, 1)),
            R=np.ones((1, 1)),
            mu_0=np.zeros(1),
            Sigma_0=np.ones((1, 1)),
        )


def test_model_rejects_non_symmetric_covariance() -> None:
    with pytest.raises(ValueError, match="Q must be symmetric"):
        _model(Q=np.array([[1.0, 0.4], [0.0, 1.0]]))


def test_model_rejects_indefinite_covariance_with_positive_diagonal() -> None:
    with pytest.raises(ValueError, match="Q must be positive semidefinite"):
        _model(Q=np.array([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: _model(R=value),
        lambda value: _model(Sigma_0=value),
    ],
)
def test_model_requires_positive_definite_measurement_and_prior_covariance(
    factory: Callable[[FloatArray], LinearGaussianSSM],
) -> None:
    with pytest.raises(ValueError, match="positive definite"):
        factory(np.diag([1.0, 0.0]))


def test_model_accepts_singular_process_covariance() -> None:
    model = _model(Q=np.zeros((2, 2)))
    np.testing.assert_array_equal(model.Q, np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("state_dim", "obs_dim", "control_dim", "message"),
    [
        (0, 1, 0, "state_dim must be positive"),
        (1, 0, 0, "obs_dim must be positive"),
        (1, 1, -1, "control_dim must be non-negative"),
        (True, 1, 0, "state_dim must be an integer"),
    ],
)
def test_random_rejects_invalid_dimensions(
    state_dim: int,
    obs_dim: int,
    control_dim: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LinearGaussianSSM.random(state_dim, obs_dim, control_dim)


def test_random_model_is_reproducible_and_stable() -> None:
    first = LinearGaussianSSM.random(4, 3, 2, seed=841)
    second = LinearGaussianSSM.random(4, 3, 2, seed=841)

    np.testing.assert_array_equal(first.A, second.A)
    assert float(np.max(np.abs(np.linalg.eigvals(first.A)))) < 1.0
    assert first.B.shape == (4, 2)
    assert first.D.shape == (3, 2)


def _filter_result() -> FilterResult:
    return FilterResult(
        means=np.zeros((2, 2)),
        covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        pred_means=np.zeros((2, 2)),
        pred_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        log_likelihood=-2.0,
    )


def test_filter_result_copies_and_validates_moments() -> None:
    result = _filter_result()
    assert result.means.dtype == np.float64
    assert result.means.flags.c_contiguous
    assert result.log_likelihood == -2.0


def test_filter_result_rejects_inconsistent_and_indefinite_covariances() -> None:
    with pytest.raises(ValueError, match="pred_means must have shape"):
        FilterResult(
            means=np.zeros((2, 2)),
            covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            pred_means=np.zeros((3, 2)),
            pred_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            log_likelihood=-1.0,
        )
    result = _filter_result()
    with pytest.raises(ValueError, match="positive semidefinite"):
        FilterResult(
            means=result.means,
            covariances=np.repeat(np.diag([1.0, -1.0])[None, :, :], 2, axis=0),
            pred_means=result.pred_means,
            pred_covariances=result.pred_covariances,
            log_likelihood=result.log_likelihood,
        )

    with pytest.raises(ValueError, match="covariances must have shape"):
        FilterResult(
            means=np.zeros((2, 2)),
            covariances=np.ones((2, 1, 1)),
            pred_means=np.zeros((2, 2)),
            pred_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            log_likelihood=-1.0,
        )

    with pytest.raises(ValueError, match="covariances must be symmetric"):
        FilterResult(
            means=np.zeros((2, 2)),
            covariances=np.repeat(
                np.array([[[1.0, 0.3], [0.0, 1.0]]]),
                2,
                axis=0,
            ),
            pred_means=np.zeros((2, 2)),
            pred_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            log_likelihood=-1.0,
        )


def test_filter_result_rejects_empty_state_or_time_axis() -> None:
    with pytest.raises(ValueError, match="non-zero time and state dimensions"):
        FilterResult(
            means=np.zeros((0, 1)),
            covariances=np.zeros((0, 1, 1)),
            pred_means=np.zeros((0, 1)),
            pred_covariances=np.zeros((0, 1, 1)),
            log_likelihood=0.0,
        )


def test_filter_result_rejects_non_finite_likelihood() -> None:
    result = _filter_result()
    with pytest.raises(ValueError, match="log_likelihood must be finite"):
        FilterResult(
            means=result.means,
            covariances=result.covariances,
            pred_means=result.pred_means,
            pred_covariances=result.pred_covariances,
            log_likelihood=np.inf,
        )


def test_smooth_result_validates_cross_covariance_shape() -> None:
    with pytest.raises(ValueError, match="cross_covariances must have shape"):
        SmoothResult(
            means=np.zeros((3, 2)),
            covariances=np.repeat(np.eye(2)[None, :, :], 3, axis=0),
            cross_covariances=np.zeros((3, 2, 2)),
        )


def test_smooth_result_accepts_single_step_sequence() -> None:
    result = SmoothResult(
        means=np.zeros((1, 2)),
        covariances=np.eye(2)[None, :, :],
        cross_covariances=np.zeros((0, 2, 2)),
    )
    assert result.cross_covariances.shape == (0, 2, 2)


def test_smooth_result_rejects_empty_and_mismatched_covariance_axes() -> None:
    with pytest.raises(ValueError, match="non-zero time and state dimensions"):
        SmoothResult(
            means=np.zeros((0, 1)),
            covariances=np.zeros((0, 1, 1)),
            cross_covariances=np.zeros((0, 1, 1)),
        )
    with pytest.raises(ValueError, match="covariances must have shape"):
        SmoothResult(
            means=np.zeros((2, 2)),
            covariances=np.ones((2, 1, 1)),
            cross_covariances=np.zeros((1, 2, 2)),
        )
