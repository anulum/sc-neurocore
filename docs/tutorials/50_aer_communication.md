<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 50: AER-over-UDP Communication

Route spike events between FPGAs or between FPGA and host via standard
Ethernet using the open AER-over-UDP protocol.

## 1. Send Events

```python
from sc_neurocore.comm import AERSender, AEREvent
import numpy as np

sender = AERSender(host="127.0.0.1", port=9000)
events = [AEREvent(timestamp=100, neuron_id=5),
          AEREvent(timestamp=100, neuron_id=12)]
sender.send(events)

# Or from spike vector
spikes = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
sender.send_spikes(spikes, timestamp=200)
sender.close()
```

## 2. Receive Events

```python
from sc_neurocore.comm import AERReceiver

receiver = AERReceiver(port=9000, timeout=1.0)
events = receiver.receive()
for e in events:
    print(f"t={e.timestamp}, neuron={e.neuron_id}")
receiver.close()
```

## 3. Packet Format

Header (8 bytes): magic, sequence, event count, reserved.
Event (8 bytes): timestamp (32-bit), neuron_id (16-bit), data (16-bit).
Max 180 events per 1500-byte Ethernet MTU.

## Use Cases

- Multi-FPGA SNN clusters via standard networking
- FPGA-to-host streaming for live visualization
- Hardware-in-the-loop simulation
- Distributed neuromorphic computing

## Further Reading

- [Tutorial 36: MPI Distributed](36_mpi_distributed.md)
- [API: Communication](../api/comm.md)
