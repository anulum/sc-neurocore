# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Local UDP address-event integration

"""Exercise the public UDP raster adapter using real datagrams."""

from __future__ import annotations

import socket
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing

import numpy as np
import pytest

from sc_neurocore.comm import (
    AERReceiver,
    AERSender,
    AERTransfer,
    ReceivedAERPacket,
    receive_aer_packet,
    send_aer_packet,
)
from sc_neurocore.comm.aer_udp import EVENT_FMT, HEADER_FMT, MAGIC, MAX_EVENTS_PER_PACKET
from sc_neurocore.spike_codec import AERSpikeCodec
from tests.engine_requirement import require_engine


def _raw(seq: int, events: tuple[tuple[int, int, int], ...], *, reserved: int = 0) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, seq, len(events), reserved) + b"".join(
        struct.pack(EVENT_FMT, *event) for event in events
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_udp_inverts_dense_container_into_active_events_and_reconstructs(backend: str) -> None:
    """Translate AERI over multiple real datagrams, preserving absolute tick mapping."""
    if backend == "rust":
        require_engine()
    codec = AERSpikeCodec(timestamp_bits=4, neuron_bits=16, backend=backend)
    raster = np.ones((16, 30), dtype=np.int8)
    raster[1::6, 2::3] = 0
    packet, stats = codec.compress(raster)
    assert packet[:4] == b"AERI"
    with (
        closing(AERReceiver(port=0, timeout=1.0)) as receiver,
        closing(AERSender(port=receiver.bound_port)) as sender,
        ThreadPoolExecutor(max_workers=1) as worker,
    ):
        incoming: list[Future[ReceivedAERPacket]] = []

        def announce(description: AERTransfer) -> None:
            assert sender.next_sequence == description.first_sequence
            incoming.append(worker.submit(receive_aer_packet, receiver, description, codec=codec))

        transfer = send_aer_packet(
            sender, packet, codec=codec, timestamp_origin=501, ticks_per_sample=7, announce=announce
        )
        assert (
            transfer.packet_count
            == (stats.n_spikes + MAX_EVENTS_PER_PACKET - 1) // MAX_EVENTS_PER_PACKET
        )
        assert sender.next_sequence == transfer.packet_count
        result = incoming[0].result(timeout=2.0)
    assert result.reordered_datagrams == 0 and result.packet == packet
    np.testing.assert_array_equal(codec.decompress(result.packet), raster)


def test_udp_reorders_sequence_wrap_and_coalesces_duplicate_raster_events() -> None:
    """Place out-of-order datagrams and equal coordinates into binary positions."""
    transfer = AERTransfer(3, 3, 65535, 2, 100, 2)
    with (
        closing(AERReceiver(port=0, timeout=1.0)) as receiver,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
    ):
        target = ("127.0.0.1", receiver.bound_port)
        sender.sendto(_raw(0, ((104, 0, 0), (104, 1, 0))), target)
        sender.sendto(_raw(65535, ((100, 2, 0), (104, 0, 0))), target)
        received = receive_aer_packet(receiver, transfer)
    assert received.reordered_datagrams == 1
    np.testing.assert_array_equal(
        AERSpikeCodec().decompress(received.packet), [[0, 0, 1], [0, 0, 0], [1, 1, 0]]
    )


def test_empty_transfer_and_clock_or_channel_overflow() -> None:
    """Send no datagram for silence and reject unrepresentable UDP coordinates."""
    codec = AERSpikeCodec(backend="python")
    with (
        closing(AERReceiver(port=0, timeout=0.03)) as receiver,
        closing(AERSender(port=receiver.bound_port)) as sender,
    ):
        packet, _ = codec.compress(np.zeros((3, 4), dtype=np.int8))
        transfer = send_aer_packet(sender, packet)
        assert transfer.packet_count == 0
        assert receive_aer_packet(receiver, transfer).packet == packet
        assert receiver.receive_packet() is None
        with pytest.raises(ValueError, match="timestamps must fit"):
            send_aer_packet(sender, packet, timestamp_origin=0xFFFFFFFF)
        oversized, _ = codec.compress(np.empty((0, 65537), dtype=np.int8))
        with pytest.raises(ValueError, match="dimensions exceed"):
            send_aer_packet(sender, oversized)
    with pytest.raises(ValueError, match="sequence window"):
        AERTransfer(3, 4, 0, 32768)
    with pytest.raises(ValueError, match="dimensions exceed"):
        AERTransfer(-1, 4, 0, 0)


@pytest.mark.parametrize(
    "packets,message",
    [
        ([b"\x00"], "truncated AER UDP header"),
        ([_raw(7, ((100, 0, 0),), reserved=1)], "invalid AER UDP header"),
        ([_raw(7, ((100, 0, 0),))[:-1]], "invalid AER UDP event body length"),
        ([_raw(7, ((100, 0, 0),)) + b"\x00"], "invalid AER UDP event body length"),
        ([_raw(8, ((100, 0, 0),)), _raw(8, ((100, 0, 0),))], "duplicate"),
        ([_raw(9, ((100, 0, 0),))], "sequence is outside"),
        ([_raw(7, ())], "empty datagram"),
        ([_raw(7, ((101, 0, 0),))], "outside the described binary raster"),
        ([_raw(7, ((100, 3, 0),))], "outside the described binary raster"),
        ([_raw(7, ((100, 1, 1),))], "outside the described binary raster"),
    ],
)
def test_udp_rejects_malformed_or_inconsistent_transfers(
    packets: list[bytes], message: str
) -> None:
    """Report damaged or unexpected datagrams through the strict adapter boundary."""
    with (
        closing(AERReceiver(port=0, timeout=0.1)) as receiver,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
    ):
        for body in packets:
            sender.sendto(body, ("127.0.0.1", receiver.bound_port))
        with pytest.raises(ValueError, match=message):
            receive_aer_packet(receiver, AERTransfer(3, 3, 7, 2, 100, 2))


def test_udp_reports_loss_and_rejects_mixed_senders() -> None:
    """Refuse incomplete transfer windows and events from two socket sources."""
    transfer = AERTransfer(2, 2, 0, 2)
    with (
        closing(AERReceiver(port=0, timeout=0.03)) as receiver,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as foreign,
    ):
        target = ("127.0.0.1", receiver.bound_port)
        sender.sendto(_raw(1, ((0, 1, 0),)), target)
        with pytest.raises(TimeoutError, match="received 1 of 2"):
            receive_aer_packet(receiver, transfer)
        sender.sendto(_raw(0, ((0, 1, 0),)), target)
        foreign.sendto(_raw(1, ((0, 1, 0),)), target)
        with pytest.raises(ValueError, match="different sources"):
            receive_aer_packet(receiver, transfer)
