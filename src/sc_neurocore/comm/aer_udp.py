# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER-over-UDP spike communication protocol

"""AER-over-UDP: open protocol for inter-FPGA spike routing.

Pack AER (Address Event Representation) spike events into UDP packets
for communication between FPGA boards or between FPGA and host.

Packet format::

  Header: magic(16) | seq(16) | n_events(16) | reserved(16) = 8 bytes
  Events: timestamp(32) | neuron_id(16) | data(16) = 8 bytes each
  Max events per packet: 180 (fits in 1500-byte Ethernet MTU)

No equivalent open standard exists for multi-FPGA SNN communication.
SpiNNaker uses a proprietary protocol. BrainScaleS uses wafer routing.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

MAGIC = 0xAE01
HEADER_FMT = "!HHHH"  # magic, seq, n_events, reserved
EVENT_FMT = "!IHH"  # timestamp, neuron_id, data
HEADER_SIZE = struct.calcsize(HEADER_FMT)
EVENT_SIZE = struct.calcsize(EVENT_FMT)
MAX_EVENTS_PER_PACKET = (1500 - HEADER_SIZE) // EVENT_SIZE


@dataclass
class AEREvent:
    """Single AER spike event."""

    timestamp: int
    neuron_id: int
    data: int = 0


class AERSender:
    """Send AER spike events over UDP.

    Parameters
    ----------
    host : str
        Destination IP address.
    port : int
        Destination UDP port.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, events: list[AEREvent]) -> int:
        """Send a batch of AER events. Returns number of packets sent."""
        packets_sent = 0
        for i in range(0, len(events), MAX_EVENTS_PER_PACKET):
            batch = events[i : i + MAX_EVENTS_PER_PACKET]
            header = struct.pack(HEADER_FMT, MAGIC, self._seq & 0xFFFF, len(batch), 0)
            body = b"".join(
                struct.pack(
                    EVENT_FMT, e.timestamp & 0xFFFFFFFF, e.neuron_id & 0xFFFF, e.data & 0xFFFF
                )
                for e in batch
            )
            self._sock.sendto(header + body, (self.host, self.port))
            self._seq += 1
            packets_sent += 1
        return packets_sent

    def send_spikes(self, spike_vector: np.ndarray[Any, Any], timestamp: int) -> int:
        """Convert a binary spike vector to AER events and send."""
        events = [
            AEREvent(timestamp=timestamp, neuron_id=int(i)) for i in np.nonzero(spike_vector)[0]
        ]
        if events:
            return self.send(events)
        return 0

    def close(self) -> None:
        self._sock.close()


class AERReceiver:
    """Receive AER spike events over UDP.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Listen port.
    timeout : float
        Socket timeout in seconds.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, timeout: float = 1.0):
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)
        self._sock.bind((host, port))

    def receive(self) -> list[AEREvent]:
        """Receive one packet of AER events. Returns empty list on timeout."""
        try:
            data, addr = self._sock.recvfrom(2048)
        except TimeoutError:
            return []

        if len(data) < HEADER_SIZE:
            return []

        magic, seq, n_events, _ = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        if magic != MAGIC:
            return []

        events = []
        offset = HEADER_SIZE
        for _ in range(n_events):
            if offset + EVENT_SIZE > len(data):
                break
            ts, nid, d = struct.unpack(EVENT_FMT, data[offset : offset + EVENT_SIZE])
            events.append(AEREvent(timestamp=ts, neuron_id=nid, data=d))
            offset += EVENT_SIZE

        return events

    def receive_as_vector(self, n_neurons: int) -> tuple[np.ndarray[Any, Any], int]:
        """Receive and convert to binary spike vector.

        Returns (spike_vector, timestamp) or (zeros, -1) on timeout.
        """
        events = self.receive()
        vector = np.zeros(n_neurons, dtype=np.int8)
        ts = -1
        for e in events:
            if 0 <= e.neuron_id < n_neurons:
                vector[e.neuron_id] = 1
            ts = e.timestamp
        return vector, ts

    def close(self) -> None:
        self._sock.close()
