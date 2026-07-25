# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Linear Gaussian state-space contract fixtures

from __future__ import annotations

import numpy as np

from sc_neurocore.world_model._lgssm_types import FloatArray
from sc_neurocore.world_model.predictive_model import FilterResult, LinearGaussianSSM


def model(
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
    """Build the two-state reference model used by contract tests."""

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


def filter_result() -> FilterResult:
    """Build a valid two-step filtering result."""

    return FilterResult(
        means=np.zeros((2, 2)),
        covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        pred_means=np.zeros((2, 2)),
        pred_covariances=np.repeat(np.eye(2)[None, :, :], 2, axis=0),
        log_likelihood=-2.0,
    )
