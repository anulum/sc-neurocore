# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian smoothing result contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.world_model.predictive_model import SmoothResult


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
