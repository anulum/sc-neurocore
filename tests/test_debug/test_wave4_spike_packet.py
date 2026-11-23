# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikePacket from former test_wave4.py

"""Focused suite: TestSpikePacket from former test_wave4.py."""

from __future__ import annotations

from sc_neurocore.bridges.aer_router import (
    PACKET_SIZE,
    SpikePacket,
)


class TestSpikePacket:
    """Encoding spike packets."""

    def test_encode_decode(self) -> None:
        """A packet decodes back with its ids."""
        p = SpikePacket(source_id=100, target_id=200, timestamp=12345, spike_len=64, sequence=42)
        data = p.encode()
        assert len(data) == PACKET_SIZE
        p2 = SpikePacket.decode(data)
        assert p2.source_id == 100
        assert p2.target_id == 200
        assert p2.sequence == 42
