# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeConvolve from former test_spike_dsp.py

"""Focused suite: TestSpikeConvolve from former test_spike_dsp.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.spike_dsp import (
    spike_convolve,
)


class TestSpikeConvolve:
    """Convolution of spike trains."""

    def test_basic(self) -> None:
        """Convolution returns a trace of the input length."""
        spikes = np.zeros(30, dtype=np.int8)
        spikes[10:15] = 1
        out = spike_convolve(spikes, np.array([0.5, 0.3, 0.2]), threshold=0.3)
        assert out.shape == (30,)
