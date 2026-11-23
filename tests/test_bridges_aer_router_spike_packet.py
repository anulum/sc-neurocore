# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikePacket from former test_bridges_aer_router.py

"""Focused suite: TestSpikePacket from former test_bridges_aer_router.py."""

from __future__ import annotations

import pytest
import struct
from sc_neurocore.bridges.aer_router import (
    PACKET_SIZE,
    SpikePacket,
)


class TestSpikePacket:
    """Packet encode / decode round-trip and edge cases."""

    def test_encode_decode_basic(self) -> None:
        """A packet round-trips with its fields."""
        pkt = SpikePacket(source_id=42, target_id=99, timestamp=1000, spike_len=4, sequence=1)
        raw = pkt.encode()
        assert len(raw) == PACKET_SIZE
        restored = SpikePacket.decode(raw)
        assert restored.source_id == 42
        assert restored.target_id == 99
        assert restored.timestamp == 1000
        assert restored.spike_len == 4
        assert restored.sequence == 1

    def test_encode_decode_zero_fields(self) -> None:
        """Zero fields round-trip."""
        pkt = SpikePacket()
        restored = SpikePacket.decode(pkt.encode())
        assert restored.source_id == 0
        assert restored.target_id == 0
        assert restored.timestamp == 0
        assert restored.spike_len == 0
        assert restored.sequence == 0

    def test_encode_decode_negative_sequence(self) -> None:
        """A negative sequence number round-trips."""
        pkt = SpikePacket(source_id=1, target_id=2, timestamp=5, spike_len=1, sequence=-42)
        restored = SpikePacket.decode(pkt.encode())
        assert restored.sequence == -42

    def test_encode_decode_large_timestamp(self) -> None:
        """A large timestamp round-trips."""
        ts = 2**48
        pkt = SpikePacket(source_id=10, target_id=20, timestamp=ts, spike_len=8, sequence=100)
        restored = SpikePacket.decode(pkt.encode())
        assert restored.timestamp == ts

    def test_encode_is_big_endian(self) -> None:
        """Packets are encoded big-endian."""
        pkt = SpikePacket(source_id=1, target_id=0, timestamp=0, spike_len=0, sequence=0)
        raw = pkt.encode()
        assert raw[3] == 1  # big-endian u32: least significant byte last

    @pytest.mark.parametrize(
        "src,tgt,seq",
        [
            (0, 0, 0),
            (1, 2, 3),
            (2**32 - 1, 2**32 - 1, 2**63 - 1),
            (100, 200, -100),
        ],
    )
    def test_fuzz_encode_decode(self, src: int, tgt: int, seq: int) -> None:
        """Boundary values round-trip."""
        pkt = SpikePacket(source_id=src, target_id=tgt, timestamp=0, spike_len=0, sequence=seq)
        restored = SpikePacket.decode(pkt.encode())
        assert restored.source_id == src
        assert restored.target_id == tgt
        assert restored.sequence == seq

    def test_decode_ignores_trailing_data(self) -> None:
        """Trailing bytes are ignored on decode."""
        pkt = SpikePacket(source_id=5, target_id=10, timestamp=99, spike_len=1, sequence=7)
        raw = pkt.encode() + b"\xff" * 16
        restored = SpikePacket.decode(raw)
        assert restored.source_id == 5
        assert restored.sequence == 7

    def test_decode_short_data_raises(self) -> None:
        """Short data cannot be decoded."""
        with pytest.raises(struct.error):
            SpikePacket.decode(b"\x00" * 4)

    def test_packet_size_constant(self) -> None:
        """A packet is 28 bytes."""
        assert PACKET_SIZE == 28
