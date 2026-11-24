# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMEdgeCases from former test_model_spike_response.py

"""Focused suite: TestSRMEdgeCases from former test_model_spike_response.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.spike_response import (
    SpikeResponseNeuron,
)


class TestSRMEdgeCases:
    """Zero and negative input, spike timing state and determinism of the SpikeResponseNeuron."""

    def test_zero_input_silent(self) -> None:
        """Zero input gives no spikes."""
        n = SpikeResponseNeuron()
        assert all(n.step(0.0) == 0 for _ in range(1000))

    def test_negative_input(self) -> None:
        """A negative input long after a spike gives a negative voltage."""
        n = SpikeResponseNeuron()
        n.time_since_spike = 1000.0
        n.step(-10.0)
        assert n.v < 0

    def test_time_since_spike_increments(self) -> None:
        """Time since spike increments."""
        n = SpikeResponseNeuron()
        tss0 = n.time_since_spike
        n.step(0.0)
        assert n.time_since_spike == tss0 + n.dt

    def test_spike_resets_tss_to_zero(self) -> None:
        """Spike resets tss to zero."""
        n = SpikeResponseNeuron()
        n.step(10.0)
        assert n.time_since_spike == 0.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = SpikeResponseNeuron()
            trace = [(n.step(10.0), n.v, n.time_since_spike) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
