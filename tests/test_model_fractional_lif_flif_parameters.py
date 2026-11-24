# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFLIFParameters from former test_model_fractional_lif.py

"""Focused suite: TestFLIFParameters from former test_model_fractional_lif.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.fractional_lif import (
    FractionalLIFNeuron,
)


class TestFLIFParameters:
    """Parameter behaviour of the FractionalLIFNeuron."""

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
    def test_alpha_variations(self, alpha: float) -> None:
        """The state stays finite across fractional orders."""
        n = FractionalLIFNeuron(alpha=alpha)
        for _ in range(5000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = FractionalLIFNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
