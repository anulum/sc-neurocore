# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian state-space parameter contracts

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import LinearGaussianSSM
from tests.test_world_model.linear_gaussian_ssm_support import model


def test_model_copies_parameters_to_finite_contiguous_float64() -> None:
    transition = np.eye(2, dtype=np.float32)
    instance = model(A=cast(FloatArray, transition))
    transition[0, 0] = 9.0

    assert instance.A.dtype == np.float64
    assert instance.A.flags.c_contiguous
    assert instance.A[0, 0] == 1.0
    assert (instance.state_dim, instance.obs_dim, instance.control_dim) == (2, 2, 1)


@pytest.mark.parametrize(
    ("factory", "malformed", "message"),
    [
        (lambda value: model(A=value), np.zeros((2, 3)), "A must be a non-empty square"),
        (lambda value: model(B=value), np.zeros((3, 1)), "B must have 2 rows"),
        (lambda value: model(C=value), np.zeros((2, 3)), "C must have shape"),
        (lambda value: model(D=value), np.zeros((2, 2)), "D must have shape"),
        (lambda value: model(Q=value), np.eye(3), "Q must have shape"),
        (lambda value: model(R=value), np.eye(3), "R must have shape"),
        (lambda value: model(mu_0=value), np.zeros(3), "mu_0 must have shape"),
        (lambda value: model(Sigma_0=value), np.eye(3), "Sigma_0 must have shape"),
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
        (lambda value: model(A=value), np.full((2, 2), np.nan)),
        (lambda value: model(B=value), np.full((2, 1), np.nan)),
        (lambda value: model(C=value), np.full((2, 2), np.nan)),
        (lambda value: model(D=value), np.full((2, 1), np.nan)),
        (lambda value: model(Q=value), np.full((2, 2), np.nan)),
        (lambda value: model(R=value), np.full((2, 2), np.nan)),
        (lambda value: model(mu_0=value), np.full(2, np.nan)),
        (lambda value: model(Sigma_0=value), np.full((2, 2), np.nan)),
    ],
)
def test_model_rejects_non_finite_parameters(
    factory: Callable[[FloatArray], LinearGaussianSSM], value: FloatArray
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
        model(Q=np.array([[1.0, 0.4], [0.0, 1.0]]))


def test_model_rejects_indefinite_covariance_with_positive_diagonal() -> None:
    with pytest.raises(ValueError, match="Q must be positive semidefinite"):
        model(Q=np.array([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize(
    "factory", [lambda value: model(R=value), lambda value: model(Sigma_0=value)]
)
def test_model_requires_positive_definite_measurement_and_prior_covariance(
    factory: Callable[[FloatArray], LinearGaussianSSM],
) -> None:
    with pytest.raises(ValueError, match="positive definite"):
        factory(np.diag([1.0, 0.0]))


def test_model_accepts_singular_process_covariance() -> None:
    instance = model(Q=np.zeros((2, 2)))
    np.testing.assert_array_equal(instance.Q, np.zeros((2, 2)))
