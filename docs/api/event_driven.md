# Event-Driven Simulation

Event-driven simulation: only update neurons with pending events.

- `EventDrivenSimulator` — Priority queue of spike events over leaky integrate-and-fire neurons. Only neurons with pending input are processed: O(K log N) per spike where K is fan-out. SparseProp (NeurIPS 2023) reported up to 10,000x speedups over clock-driven simulation on 1M-neuron sparse networks.

Each neuron relaxes exactly from its own last update, `v_rest + (v - v_rest) * exp(-dt / tau_mem)`, before an event's weight is added. A neuron at `threshold` spikes, resets to `v_reset` and discards inputs for `refractory` ms. Events due after a run's `duration` stay queued for the next run. Supports external spike and current injection and run statistics (events processed, spikes, queue peak, clock-driven estimate). Invalid parameters, connections and inputs raise `ValueError`.

```python
from sc_neurocore.event_driven import EventDrivenSimulator

# (source, target, weight, delay_ms)
sim = EventDrivenSimulator(n_neurons=3, connectivity=[(0, 1, 1.5, 1.0), (1, 2, 1.5, 1.0)])
sim.inject_spikes([(0.0, 0)])  # (time_ms, neuron_id)
spikes, stats = sim.run(duration=10.0)
print(spikes)  # [(1.0, 1), (2.0, 2)]
print(f"Events processed: {stats.total_events_processed}")
```

See [Tutorial 65: Event-Driven Simulation](../tutorials/65_event_driven.md).

::: sc_neurocore.event_driven
    options:
      show_root_heading: true
