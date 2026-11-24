# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPSNCustomKernel from former test_model_psn.py

"""Focused suite: TestPSNCustomKernel from former test_model_psn.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.sc_resetting_psn import (
    SCResettingParallelSpikingNeuron,
)


class TestPSNCustomKernel:
    """A custom kernel changes the parallel spiking neuron's scoring."""

    def test_custom_kernel_affects_scoring(self) -> None:
        """Non-uniform kernel weights recent inputs differently."""
        n = SCResettingParallelSpikingNeuron(kernel_size=4, v_threshold=1.0)
        n.kernel = np.array([0.0, 0.0, 0.0, 1.0])  # only last entry
        # Only the value at position 3 matters
        n.step(0.0)
        n.step(0.0)
        n.step(0.0)
        s = n.step(1.0)  # pos 3 gets 1.0, score = 1.0
        assert s == 1
