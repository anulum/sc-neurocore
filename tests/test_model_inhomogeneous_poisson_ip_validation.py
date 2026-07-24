# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIPValidation from former test_model_inhomogeneous_poisson.py

"""Focused suite: TestIPValidation from former test_model_inhomogeneous_poisson.py."""

from __future__ import annotations

from tests.model_inhomogeneous_poisson_support import *  # noqa: F403


class TestIPValidation:
    @pytest.mark.parametrize("dt_ms", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt_ms: float):
        with pytest.raises(ValueError, match="dt_ms"):
            InhomogeneousPoissonNeuron(dt_ms=dt_ms)

    @pytest.mark.parametrize("rate_hz", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_rate_before_sampling(self, rate_hz: float):
        n = InhomogeneousPoissonNeuron()
        with pytest.raises(ValueError, match="rate_hz"):
            n.step(rate_hz)
