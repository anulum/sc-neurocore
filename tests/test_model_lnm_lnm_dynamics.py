# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLNMDynamics from former test_model_lnm.py

"""Focused suite: TestLNMDynamics from former test_model_lnm.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.lnm import (
    LearnableNeuronModel,
)
from tests.model_lnm_support import (
    _run,
)


class TestLNMDynamics:
    """Dynamics of the LearnableNeuronModel."""

    def test_fires(self) -> None:
        """Sustained drive makes the neuron fire."""
        assert len(_run(LearnableNeuronModel(), 5.0, 5000)) >= 50

    def test_subthreshold(self) -> None:
        """Weak drive gives no spikes."""
        assert len(_run(LearnableNeuronModel(), 0.05, 5000)) == 0

    def test_rate_monotonic(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = [len(_run(LearnableNeuronModel(), I, 5000)) for I in [1.0, 5.0, 10.0]]
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 1.0, 5.0, 10.0])
    def test_fi_sweep(self, current: float) -> None:
        """The state stays finite across a current sweep."""
        n = LearnableNeuronModel()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.v)
