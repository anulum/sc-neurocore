# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidRatePipeline from former test_model_sigmoid_rate.py

"""Focused suite: TestSigmoidRatePipeline from former test_model_sigmoid_rate.py."""

from __future__ import annotations

from tests.model_sigmoid_rate_support import *  # noqa: F403


class TestSigmoidRatePipeline:
    def test_population_creates(self):
        assert Population(SigmoidRateNeuron, n=10, label="sr").n == 10

    def test_returns_float_not_spike(self):
        """Rate model — returns float. Network.step_all limited."""
        n = SigmoidRateNeuron()
        assert isinstance(n.step(5.0), (float, np.floating))
