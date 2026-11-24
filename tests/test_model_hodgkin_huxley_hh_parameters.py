# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHHParameters from former test_model_hodgkin_huxley.py

"""Focused suite: TestHHParameters from former test_model_hodgkin_huxley.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.hodgkin_huxley import (
    HodgkinHuxleyNeuron,
)


class TestHHParameters:
    """Parameter behaviour of the HodgkinHuxleyNeuron."""

    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float) -> None:
        """The state stays finite across time steps."""
        n = HodgkinHuxleyNeuron(dt=dt)
        for _ in range(2000):
            n.step(10.0)
        assert np.isfinite(n.v)

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = HodgkinHuxleyNeuron()
            trace = [(n.step(10.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
