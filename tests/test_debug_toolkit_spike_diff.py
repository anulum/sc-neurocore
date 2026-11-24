# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeDiff from former test_debug_toolkit.py

"""Focused suite: TestSpikeDiff from former test_debug_toolkit.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.debug.analyzer import (
    spike_diff,
)
from tests.debug_toolkit_support import (
    _make_trace,
)


class TestSpikeDiff:
    """Comparing the spikes of two traces."""

    def test_identical(self) -> None:
        """Identical traces have no mismatch."""
        spikes = np.zeros((10, 5), dtype=np.int8)
        t1 = _make_trace(spikes=spikes.copy())
        t2 = _make_trace(spikes=spikes.copy())
        d = spike_diff(t1, t2)
        assert d["total_mismatches"] == 0
        assert d["mismatch_rate"] == 0.0
        assert d["first_divergence"] is None

    def test_with_mismatches(self) -> None:
        """Two mismatches give their rate and first divergence."""
        s1 = np.zeros((10, 5), dtype=np.int8)
        s2 = np.zeros((10, 5), dtype=np.int8)
        s1[0, 0] = 1
        s1[5, 3] = 1
        t1 = _make_trace(spikes=s1)
        t2 = _make_trace(spikes=s2)
        d = spike_diff(t1, t2)
        assert d["total_mismatches"] == 2
        assert d["mismatch_rate"] == pytest.approx(2.0 / 50)
        assert d["first_divergence"] is not None
        assert d["per_neuron_mismatches"][0] == 1
        assert d["per_neuron_mismatches"][3] == 1
