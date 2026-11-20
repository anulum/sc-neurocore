<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 50: AER-over-UDP Communication

Route addressed spike events between software processes over UDP.
The examples below use the SC-NeuroCore packet format and its localhost
raster adapter.

## What is AER

Address-Event Representation (Boahen 2000) is the standard
communication protocol for neuromorphic systems. When a neuron fires,
its address (ID) is broadcast as a packet. Only active neurons generate
traffic — idle neurons consume zero bandwidth.

```
Traditional bus: clock every cycle → N neurons × N timesteps data
AER bus: fire event only → proportional to spike count
```

Packet size depends on the event rate, encoded fields and transport headers.

## Sending Events

```python
from sc_neurocore.comm import AERSender, AEREvent
import numpy as np

sender = AERSender(host="127.0.0.1", port=9000)

# Send individual events
events = [
    AEREvent(timestamp=100, neuron_id=5),
    AEREvent(timestamp=100, neuron_id=12),
    AEREvent(timestamp=200, neuron_id=7),
]
sender.send(events)
print(f"Sent {len(events)} events")

# Or from a binary spike vector (auto-converts to AER)
spikes = np.zeros(100, dtype=np.int8)
spikes[[5, 12, 47, 83]] = 1  # 4 active neurons
sender.send_spikes(spikes, timestamp=300)
print(f"Sent {spikes.sum()} spikes as AER")

sender.close()
```

## Receiving Events

```python
from sc_neurocore.comm import AERReceiver

receiver = AERReceiver(port=9000, timeout=1.0)

events = receiver.receive()
for e in events:
    print(f"t={e.timestamp:>6}, neuron={e.neuron_id:>4}")

# Continuous receive loop (for real-time applications)
receiver = AERReceiver(port=9000, timeout=0.1)
while True:
    events = receiver.receive()
    if events:
        # Process events...
        print(f"Received {len(events)} events")

receiver.close()
```

## Sending a compressed raster

```python
from contextlib import closing
import numpy as np

from sc_neurocore.comm import AERReceiver, AERSender, receive_aer_packet, send_aer_packet
from sc_neurocore.spike_codec import AERSpikeCodec

codec = AERSpikeCodec()
spikes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
packet, _ = codec.compress(spikes)
with closing(AERReceiver(port=0)) as receiver:
    with closing(AERSender(port=receiver.bound_port)) as sender:
        transfer = send_aer_packet(sender, packet, timestamp_origin=100, ticks_per_sample=5)
        result = receive_aer_packet(receiver, transfer)
np.testing.assert_array_equal(codec.decompress(result.packet), spikes)
```

In a distributed application, pass an `announce` callback to
`send_aer_packet` to share the transfer descriptor through the control
channel before data is sent. Start the receiver while the sender is
active. The socket receive queue is finite. The adapter accepts
reordering within the expected sequence window and reports loss on timeout.
It refuses malformed or duplicate datagrams and does not retransmit them.
The verified round trip uses software on localhost.

## Packet Format

```
AER-over-UDP packet (up to 1500 bytes of UDP payload):

Header (8 bytes):
  [0:2]  Magic: 0xAE01 (identifies AER packet)
  [2:4]  Sequence number (for drop detection)
  [4:6]  Event count (0–186)
  [6:8]  Reserved (zero)

Per Event (8 bytes):
  [0:4]  Timestamp (uint32 ticks, units selected by the application)
  [4:6]  Neuron ID (uint16, 0–65,535)
  [6:8]  Data (16-bit: payload, polarity, layer ID, etc.)

Max events per packet: floor((1500 - 8) / 8) = 186
```

All datagram fields use big-endian byte order. The raster adapter uses a
zero data field and requires the complete timestamp range to fit uint32.
The payload limit excludes IP and UDP headers.

## Multi-FPGA Cluster

Connect multiple FPGAs via standard Ethernet switches:

```
FPGA_A (layer 1) → Ethernet switch → FPGA_B (layer 2) → FPGA_C (readout)
```

Each FPGA uses `sc_aer_encoder.v` (output) and `sc_aer_router.v` (input).
The clocked encoder emits one lowest-index pending address per cycle and records
the service-cycle timestamp. Already pending spike bits coalesce, with clearing
the served bit taking precedence over a simultaneous spike on that bit. A queue
between the encoder and transport owns downstream flow control.

```python
# FPGA A: encode output spikes as AER-over-UDP
sender_a = AERSender(host="192.168.1.10", port=9000)  # FPGA B address

# FPGA B: receive, process, forward
receiver_b = AERReceiver(port=9000)
sender_b = AERSender(host="192.168.1.11", port=9000)  # FPGA C address

# FPGA C: receive and classify
receiver_c = AERReceiver(port=9000)
```

Bandwidth at 1% firing rate, 10K neurons per FPGA:
```
Events/second: 10,000 × 0.01 × 1000 Hz = 100,000 events/s
Bandwidth: 100,000 × 8 bytes = 800 KB/s (0.6% of Gigabit Ethernet)
```

The raw Gigabit Ethernet line-rate ceiling for eight-byte events is
15.625 million events/second before framing and protocol overhead — enough for
a multi-million neuron distributed system.

## Hardware-in-the-Loop

Run part of the network in simulation, part on FPGA:

```python
# Python simulation (layers 1-3)
from sc_neurocore.studio.network import simulate_ei_network

result = simulate_ei_network(n_exc=800, n_inh=200, duration=100.0)

# Send output spikes to FPGA (layer 4)
sender = AERSender(host="192.168.1.100", port=9000)
for t, nid in zip(result["spike_times"], result["spike_neurons"]):
    sender.send([AEREvent(timestamp=int(t * 1000), neuron_id=nid)])

# Receive FPGA output
receiver = AERReceiver(port=9001)
fpga_spikes = receiver.receive()
```

## Use Cases

| Application | Setup | Why AER |
|-------------|-------|---------|
| Multi-FPGA SNN | 2-8 FPGAs via switch | Standard networking, no custom interconnect |
| FPGA ↔ host | FPGA + PC | Live visualisation in Studio |
| HIL simulation | Python + FPGA | Verify hardware against software golden model |
| DVS → FPGA | Camera + FPGA | DVS cameras natively output AER |
| Distributed neuromorphic | Multiple chips | SpiNNaker, Loihi use AER internally |

## References

- Boahen (2000). "Point-to-point connectivity between neuromorphic
  chips using address events." IEEE TCAS-II 47(5):416-434.
- Serrano-Gotarredona et al. (2009). "A Neuromorphic Cortical-Layer
  Microchip for Spike-Based Event Processing Vision Systems." IEEE
  TCAS-I 56(3):601-612.
- Deiss et al. (1998). "Address-event asynchronous local broadcast
  protocol." Analog Integrated Circuits 13(1):67-81.
