# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAERToSCConverter from former test_encoding.py

"""Focused suite: TestAERToSCConverter from former test_encoding.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.bioware.bioware import (
    AEREvent,
    AERToSCConverter,
)


class TestAERToSCConverter:
    """Converting address events to bitstreams."""

    def test_convert(self) -> None:
        """Each neuron gets a bitstream of 128 bits."""
        events = [
            AEREvent(neuron_id=0, timestamp=100),
            AEREvent(neuron_id=0, timestamp=200),
            AEREvent(neuron_id=1, timestamp=150),
        ]
        conv = AERToSCConverter(bitstream_length=128)
        bitstreams = conv.convert(events)
        assert 0 in bitstreams
        assert 1 in bitstreams
        assert len(bitstreams[0]) == 128

    def test_density_proportional(self) -> None:
        """A busier neuron gets a denser bitstream."""
        events = [AEREvent(neuron_id=0, timestamp=i) for i in range(10)]
        events += [AEREvent(neuron_id=1, timestamp=i) for i in range(5)]
        conv = AERToSCConverter(bitstream_length=1024)
        bs = conv.convert(events)
        d0 = float(np.sum(bs[0])) / len(bs[0])
        d1 = float(np.sum(bs[1])) / len(bs[1])
        assert d0 > d1

    def test_empty_events(self) -> None:
        """No events give no bitstreams."""
        conv = AERToSCConverter()
        bs = conv.convert([])
        assert len(bs) == 0

    def test_lfsr_encode_zero_seed_is_reset(self) -> None:
        """A zero seed still encodes a full uint8 stream."""
        # A zero LFSR register is a fixed point; with lfsr_seed=0 and neuron 0
        # the derived seed is 0 and must be bumped to 1 before stepping.
        conv = AERToSCConverter(bitstream_length=64, num_neurons=4, lfsr_seed=0)
        bits = conv._lfsr_encode(0.5, neuron_id=0)
        assert bits.shape == (64,)
        assert bits.dtype == np.uint8
