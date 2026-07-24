# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaValidation from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaValidation from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403


class TestSigmaDeltaValidation:
    @pytest.mark.parametrize("sigma", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_accumulator(self, sigma: float):
        with pytest.raises(ValueError, match="sigma"):
            SigmaDeltaNeuron(sigma=sigma)

    @pytest.mark.parametrize("v_threshold", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_threshold(self, v_threshold: float):
        with pytest.raises(ValueError, match="v_threshold"):
            SigmaDeltaNeuron(v_threshold=v_threshold)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_accumulator_mutation(self, current: float):
        n = SigmaDeltaNeuron(sigma=0.25)
        before = n.sigma
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.sigma == before
