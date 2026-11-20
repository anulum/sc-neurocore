# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Address-event raster integration with UDP

"""Translate ordered binary raster packets to absolute UDP events and back."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import index

import numpy as np

from sc_neurocore.spike_codec import AERSpikeCodec
from sc_neurocore.spike_codec.aer_packet import _Packet

from .aer_udp import MAX_EVENTS_PER_PACKET, AEREvent, AERReceiver, AERSender


def _clock(rows: int, origin: int, stride: int) -> tuple[int, int]:
    start, step = index(origin), index(stride)
    if start < 0 or step < 1 or start + max(rows - 1, 0) * step > 0xFFFFFFFF:
        raise ValueError("AER transfer timestamps must fit uint32 without wrap")
    return start, step


@dataclass(frozen=True, slots=True)
class AERTransfer:
    """Out-of-band raster geometry, clock and sequence window for one transfer.

    Rows count samples, channels count binary raster columns, and timestamps
    use unsigned transport ticks. Each row advances by ticks_per_sample
    from timestamp_origin. The UDP datagrams carry only active events and
    their sequence numbers. Share this descriptor through the controlling
    application before receiving the packet batch.
    """

    rows: int
    channels: int
    first_sequence: int
    packet_count: int
    timestamp_origin: int = 0
    ticks_per_sample: int = 1

    def __post_init__(self) -> None:
        """Validate dimensions, clock and the unambiguous sequence range."""
        rows, channels, first, count = map(
            index, (self.rows, self.channels, self.first_sequence, self.packet_count)
        )
        if not 0 <= rows <= 0xFFFFFFFF or not 0 <= channels <= 0x10000:
            raise ValueError("AER transfer dimensions exceed UDP field widths")
        if not 0 <= first <= 0xFFFF or not 0 <= count < 0x8000:
            raise ValueError("AER transfer requires a sequence window smaller than half its range")
        _clock(rows, self.timestamp_origin, self.ticks_per_sample)


@dataclass(frozen=True, slots=True)
class ReceivedAERPacket:
    """Reconstructed raster packet and the number of reordered datagrams."""

    packet: bytes
    reordered_datagrams: int


def send_aer_packet(
    sender: AERSender,
    packet: bytes,
    *,
    codec: AERSpikeCodec | None = None,
    timestamp_origin: int = 0,
    ticks_per_sample: int = 1,
    announce: Callable[[AERTransfer], None] | None = None,
) -> AERTransfer:
    """Send the active positions from an AERX or AERI packet over the real UDP sender.

    Parameters
    ----------
    sender : AERSender
        UDP sender reserved for this transfer until the call returns.
    packet : bytes
        Complete compressed raster packet.
    codec : AERSpikeCodec or None
        Decoder to use. None selects the automatic backend.
    timestamp_origin, ticks_per_sample : int
        Map raster row zero and each subsequent row to absolute UDP ticks.
    announce : callable or None
        Optional application callback to publish the transfer descriptor
        before the first data datagram is sent.

    Returns
    -------
    AERTransfer
        Batch shape, sequence window and clock for the receiving application.

    Notes
    -----
    Each UDP event carries a zero data field. The adapter buffers at most
    one datagram of events. A socket error may follow partial UDP delivery.
    """
    header = _Packet.parse(packet)
    if header.channels > 0x10000:
        raise ValueError("AER transfer dimensions exceed UDP field widths")
    start, step = _clock(header.rows, timestamp_origin, ticks_per_sample)
    selected = codec if codec is not None else AERSpikeCodec()
    raster = selected.decompress(packet)
    rows, channels = raster.shape
    spikes = int(np.count_nonzero(raster))
    size = MAX_EVENTS_PER_PACKET
    count = (spikes + size - 1) // size
    transfer = AERTransfer(rows, channels, sender.next_sequence, count, start, step)
    if announce is not None:
        announce(transfer)
    batch: list[AEREvent] = []
    for row, samples in enumerate(raster):
        timestamp = start + row * step
        for channel in np.flatnonzero(samples):
            batch.append(AEREvent(timestamp, int(channel)))
            if len(batch) == size:
                sender.send(batch)
                batch.clear()
    if batch:
        sender.send(batch)
    return transfer


def receive_aer_packet(
    receiver: AERReceiver, transfer: AERTransfer, *, codec: AERSpikeCodec | None = None
) -> ReceivedAERPacket:
    """Receive a complete described UDP batch and encode its binary raster.

    Parameters
    ----------
    receiver : AERReceiver
        UDP receiver exclusively assigned to this transfer.
    transfer : AERTransfer
        Shape, clock and sequence window supplied by the controlling application.
    codec : AERSpikeCodec or None
        Encoder to use. Match the sending codec settings for the same wire bytes.

    Returns
    -------
    ReceivedAERPacket
        Encoded raster and count of distinct datagrams that arrived ahead of a gap.

    Raises
    ------
    TimeoutError
        If the sequence window was incomplete when a receive timed out.
    ValueError
        If a datagram is malformed, empty, duplicated, outside the window or
        contains data outside the described binary raster and clock.

    Notes
    -----
    Processing retains one raster, one datagram and a bounded sequence
    bitmap. Reordered datagrams are placed in their raster positions.
    Repeated occurrences of the same spike remain one binary position.
    """
    rows, channels = index(transfer.rows), index(transfer.channels)
    total, first = index(transfer.packet_count), index(transfer.first_sequence)
    origin, stride = _clock(rows, transfer.timestamp_origin, transfer.ticks_per_sample)
    selected = codec if codec is not None else AERSpikeCodec()
    raster = np.zeros((rows, channels), dtype=np.int8)
    seen = bytearray(total)
    next_slot = 0
    reordered = 0
    source: tuple[str, int] | None = None
    for received in range(total):
        datagram = receiver.receive_packet()
        if datagram is None:
            raise TimeoutError(f"incomplete AER transfer: received {received} of {total} datagrams")
        if source is not None and datagram.source != source:
            raise ValueError("AER UDP datagrams arrived from different sources")
        source = datagram.source
        slot = (datagram.sequence - first) & 0xFFFF
        if slot >= total:
            raise ValueError("AER UDP sequence is outside the transfer window")
        if seen[slot]:
            raise ValueError("duplicate AER UDP datagram")
        if not datagram.events:
            raise ValueError("AER transfer received an empty datagram")
        reordered += int(slot != next_slot)
        seen[slot] = 1
        while next_slot < total and seen[next_slot]:
            next_slot += 1
        for event in datagram.events:
            row, remainder = divmod(event.timestamp - origin, stride)
            if not 0 <= row < rows or remainder or event.neuron_id >= channels or event.data:
                raise ValueError("AER UDP event is outside the described binary raster or clock")
            raster[row, event.neuron_id] = 1
    packet, _ = selected.compress(raster)
    return ReceivedAERPacket(packet, reordered)
