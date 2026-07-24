# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLogLikelihood from former test_gpfa.py

"""Focused suite: TestLogLikelihood from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403


class TestLogLikelihood:
    def test_finite_for_valid_model(self) -> None:
        Y = np.asarray(_synthetic_trains(5, 200), dtype=np.float64)[:, :20]
        c0, d0, r0, tau = gpfa_pca_init(Y, 2, 20.0)
        k_all = [_gp_kernel(Y.shape[1], float(tau[j])) for j in range(2)]
        ll = _gpfa_log_likelihood(Y, c0, d0, r0, k_all)
        assert np.isfinite(ll)

    def test_rejects_non_psd_covariance(self) -> None:
        # A 1x1 marginal covariance with a large negative noise term is negative,
        # so slogdet reports a non-positive sign and the guard fires.
        Y = np.ones((1, 1), dtype=np.float64)
        C = np.array([[1.0]])
        d = np.zeros(1)
        R = np.diag([-100.0])
        k_all = [_gp_kernel(1, 40.0)]
        with pytest.raises(np.linalg.LinAlgError):
            _gpfa_log_likelihood(Y, C, d, R, k_all)
