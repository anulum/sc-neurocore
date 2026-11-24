# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidRateParameters from former test_model_sigmoid_rate.py

"""Focused suite: TestSigmoidRateParameters from former test_model_sigmoid_rate.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.sigmoid_rate import (
    SigmoidRateNeuron,
)


class TestSigmoidRateParameters:
    """Parameter behaviour of the SigmoidRateNeuron."""

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.5])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = SigmoidRateNeuron(dt=dt)
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.r)

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = SigmoidRateNeuron()
            trace = [n.step(3.0) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
