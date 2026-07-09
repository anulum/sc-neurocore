# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.comm (AER-over-UDP)

from __future__ import annotations

import struct

import numpy as np

from sc_neurocore.comm.aer_udp import (
    AEREvent,
    AERSender,
    AERReceiver,
    MAGIC,
    HEADER_FMT,
    EVENT_FMT,
    HEADER_SIZE,
    EVENT_SIZE,
    MAX_EVENTS_PER_PACKET,
)


class TestAEREvent:
    def test_default_data(self):
        e = AEREvent(timestamp=100, neuron_id=5)
        assert e.data == 0

    def test_fields(self):
        e = AEREvent(timestamp=42, neuron_id=7, data=255)
        assert e.timestamp == 42
        assert e.neuron_id == 7
        assert e.data == 255


class TestProtocolConstants:
    def test_magic(self):
        assert MAGIC == 0xAE01

    def test_header_size(self):
        assert HEADER_SIZE == 8

    def test_event_size(self):
        assert EVENT_SIZE == 8

    def test_max_events_fits_mtu(self):
        assert MAX_EVENTS_PER_PACKET * EVENT_SIZE + HEADER_SIZE <= 1500


class TestAERSenderReceiver:
    def test_send_receive_roundtrip(self):
        port = 19876
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=2.0)
        sender = AERSender(host="127.0.0.1", port=port)

        events = [
            AEREvent(timestamp=100, neuron_id=1, data=10),
            AEREvent(timestamp=100, neuron_id=5, data=20),
            AEREvent(timestamp=100, neuron_id=10, data=30),
        ]

        try:
            sent = sender.send(events)
            assert sent == 1

            received = receiver.receive()
            assert len(received) == 3
            assert received[0].timestamp == 100
            assert received[0].neuron_id == 1
            assert received[0].data == 10
            assert received[2].neuron_id == 10
        finally:
            sender.close()
            receiver.close()

    def test_send_spikes_vector(self):
        port = 19877
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=2.0)
        sender = AERSender(host="127.0.0.1", port=port)

        try:
            spike_vector = np.array([0, 1, 0, 1, 0, 0, 1, 0])
            sent = sender.send_spikes(spike_vector, timestamp=42)
            assert sent == 1

            received = receiver.receive()
            assert len(received) == 3
            neuron_ids = {e.neuron_id for e in received}
            assert neuron_ids == {1, 3, 6}
            assert all(e.timestamp == 42 for e in received)
        finally:
            sender.close()
            receiver.close()

    def test_send_empty_spikes(self):
        sender = AERSender(host="127.0.0.1", port=19878)
        try:
            sent = sender.send_spikes(np.zeros(10), timestamp=0)
            assert sent == 0
        finally:
            sender.close()

    def test_receive_timeout(self):
        receiver = AERReceiver(host="127.0.0.1", port=19879, timeout=0.1)
        try:
            events = receiver.receive()
            assert events == []
        finally:
            receiver.close()

    def test_receive_as_vector(self):
        port = 19880
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=2.0)
        sender = AERSender(host="127.0.0.1", port=port)

        try:
            events = [AEREvent(timestamp=7, neuron_id=2), AEREvent(timestamp=7, neuron_id=4)]
            sender.send(events)
            vec, ts = receiver.receive_as_vector(n_neurons=8)
            assert vec[2] == 1
            assert vec[4] == 1
            assert vec[0] == 0
            assert ts == 7
        finally:
            sender.close()
            receiver.close()

    def test_receive_as_vector_timeout(self):
        receiver = AERReceiver(host="127.0.0.1", port=19881, timeout=0.1)
        try:
            vec, ts = receiver.receive_as_vector(n_neurons=10)
            assert np.all(vec == 0)
            assert ts == -1
        finally:
            receiver.close()

    def test_receive_bad_magic(self):
        import socket

        port = 19882
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=1.0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            bad_header = struct.pack(HEADER_FMT, 0xDEAD, 0, 1, 0)
            bad_event = struct.pack(EVENT_FMT, 100, 5, 0)
            sock.sendto(bad_header + bad_event, ("127.0.0.1", port))
            events = receiver.receive()
            assert events == []
        finally:
            sock.close()
            receiver.close()

    def test_receive_truncated_packet(self):
        import socket

        port = 19883
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=1.0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(b"\x00\x01\x02", ("127.0.0.1", port))
            events = receiver.receive()
            assert events == []
        finally:
            sock.close()
            receiver.close()

    def test_receive_truncated_events(self):
        import socket

        port = 19886
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=1.0)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Header claims 5 events but only 1 event in data
            header = struct.pack(HEADER_FMT, MAGIC, 0, 5, 0)
            one_event = struct.pack(EVENT_FMT, 100, 1, 0)
            sock.sendto(header + one_event, ("127.0.0.1", port))
            events = receiver.receive()
            assert len(events) == 1
        finally:
            sock.close()
            receiver.close()

    def test_receive_as_vector_out_of_range_neuron(self):
        port = 19884
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=2.0)
        sender = AERSender(host="127.0.0.1", port=port)
        try:
            events = [AEREvent(timestamp=1, neuron_id=999)]
            sender.send(events)
            vec, ts = receiver.receive_as_vector(n_neurons=5)
            assert np.all(vec == 0)
        finally:
            sender.close()
            receiver.close()

    def test_large_batch_multiple_packets(self):
        port = 19885
        receiver = AERReceiver(host="127.0.0.1", port=port, timeout=2.0)
        sender = AERSender(host="127.0.0.1", port=port)
        try:
            n = MAX_EVENTS_PER_PACKET + 10
            events = [AEREvent(timestamp=i, neuron_id=i % 100) for i in range(n)]
            sent = sender.send(events)
            assert sent == 2

            first_batch = receiver.receive()
            assert len(first_batch) == MAX_EVENTS_PER_PACKET
            second_batch = receiver.receive()
            assert len(second_batch) == 10
        finally:
            sender.close()
            receiver.close()
