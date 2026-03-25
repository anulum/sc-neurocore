# Event-Driven Simulation

Event-driven simulation: only update neurons with pending events.

- `EventDrivenSimulator` — Priority queue of spike events. Only neurons with pending input are processed. O(K log N) per spike where K is fan-out. 10,000x speedup vs clock-driven for sparse networks.

Supports: external spike injection, STDP plasticity (trace-based), stats collection (total events, queue peak size).

```python
from sc_neurocore.event_driven import EventDrivenSimulator

sim = EventDrivenSimulator(network)
sim.inject_spikes([(0.0, neuron_id, current)])
sim.run(duration=100.0)
print(f"Events processed: {sim.stats.total_events}")
```

See [Tutorial 65: Event-Driven Simulation](../tutorials/65_event_driven.md).

::: sc_neurocore.event_driven
    options:
      show_root_heading: true
