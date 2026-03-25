# AER-over-UDP Communication

Open protocol for inter-FPGA spike routing. No equivalent open standard exists.

- `AERSender` — Pack AER events into UDP packets. Max 180 events per 1500-byte MTU.
- `AERReceiver` — Receive and decode. `receive_as_vector()` returns binary spike vectors.
- `AEREvent` — Single spike event (timestamp, neuron_id, data).

```python
from sc_neurocore.comm import AERSender, AERReceiver
```

See [Tutorial 50: AER Communication](../tutorials/50_aer_communication.md) and [Spike Codec Library](spike_codec.md).

::: sc_neurocore.comm.aer_udp
    options:
      show_root_heading: true
