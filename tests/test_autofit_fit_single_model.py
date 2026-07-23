# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFitSingleModel from former test_autofit.py

"""Focused suite: TestFitSingleModel from former test_autofit.py."""

from __future__ import annotations

from tests.autofit_support import *  # noqa: F403

class TestFitSingleModel:
    def test_fit_returns_fitted_model(self):
        class GoodNeuron:
            def __init__(self):
                self.v = 0.0

            def step(self, current):
                self.v = current * 0.5

        v_target = np.random.randn(100) * 0.1
        current = np.ones(100)
        result = _fit_single_model(GoodNeuron, "good", v_target, current, dt=0.1, threshold=0.0)
        assert isinstance(result, FittedModel)
        assert result.model_name == "good"
        assert result.rmse >= 0
        assert len(result.simulated_voltage) == 100
