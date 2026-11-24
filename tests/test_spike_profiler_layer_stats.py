# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerStats from former test_spike_profiler.py

"""Focused suite: TestLayerStats from former test_spike_profiler.py."""

from __future__ import annotations

from sc_neurocore.profiling.spike_profiler import (
    LayerStats,
)


class TestLayerStats:
    """The fields of the profiler's layer statistics."""

    def test_fields(self) -> None:
        """Fresh layer statistics start at zero."""
        s = LayerStats(name="test", n_neurons=10, n_steps=5)
        assert s.dead_neuron_count == 0
        assert s.estimated_syn_ops == 0
