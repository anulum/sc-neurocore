# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEventFilter from former test_wave4.py

"""Focused suite: TestEventFilter from former test_wave4.py."""

from __future__ import annotations

from sc_neurocore.debug.hil_client import (
    EventFilter,
    filter_events,
    SpikeEvent,
)


class TestEventFilter:
    """Filtering spike events."""

    def test_layer_filter(self) -> None:
        """Only events from the chosen layer match."""
        f = EventFilter(layer_id="L1")
        assert f.match(SpikeEvent(layer_id="L1"))
        assert not f.match(SpikeEvent(layer_id="L2"))

    def test_neuron_range(self) -> None:
        """Only events in the neuron range match."""
        f = EventFilter(has_neuron=True, min_neuron=10, max_neuron=20)
        assert f.match(SpikeEvent(neuron_id=15))
        assert not f.match(SpikeEvent(neuron_id=25))

    def test_filter_events(self) -> None:
        """Filtering a list keeps the matching event."""
        events = [SpikeEvent(layer_id="L0"), SpikeEvent(layer_id="L1")]
        result = filter_events(events, EventFilter(layer_id="L1"))
        assert len(result) == 1
