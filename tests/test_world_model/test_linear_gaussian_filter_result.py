# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian filtering result contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.world_model.predictive_model import FilterResult
from tests.test_world_model.linear_gaussian_ssm_support import filter_result


def test_filter_result_copies_and_validates_moments() -> None:
    result = filter_result()
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
    result = filter_result()
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
            covariances=np.repeat(np.array([[[1.0, 0.3], [0.0, 1.0]]]), 2, axis=0),
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
    result = filter_result()
    with pytest.raises(ValueError, match="log_likelihood must be finite"):
        FilterResult(
            means=result.means,
            covariances=result.covariances,
            pred_means=result.pred_means,
            pred_covariances=result.pred_covariances,
            log_likelihood=np.inf,
        )
