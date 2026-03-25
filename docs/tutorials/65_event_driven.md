# Tutorial 65: Event-Driven Asynchronous Simulation

Only update neurons that have pending events. Massive speedup for sparse networks.

## The Problem

Clock-driven: update ALL 1M neurons every timestep = O(N) per step.
At 1% activity, 99% of compute is wasted on idle neurons.

## Event-Driven Solution

```python
from sc_neurocore.event_driven import EventDrivenSimulator

sim = EventDrivenSimulator(
    n_neurons=10000,
    connectivity=connections,  # [(src, tgt, weight, delay), ...]
    threshold=1.0,
    tau_mem=20.0,
)

sim.inject_spikes([(0.0, 42)])  # spike at t=0, neuron 42
spikes, stats = sim.run(duration=1000.0)  # 1 second

print(stats.summary())
# EventDriven: 500 spikes, 5000 events, speedup=2000.0x
```

## How It Works

1. Events stored in a min-heap sorted by time
2. Pop earliest event, update target neuron (exponential decay + input)
3. If neuron fires: push new events for all downstream targets
4. Skip idle neurons entirely — they decay analytically

## Injecting Input

```python
# External spikes
sim.inject_spikes([(0.0, 0), (5.0, 1), (10.0, 2)])

# Current pulses
sim.inject_current([(0.0, 0, 0.5), (1.0, 0, 0.3)])
```

## When to Use

| Network | Clock-Driven | Event-Driven |
|---------|-------------|--------------|
| 1K neurons, 50% active | ~Same | ~Same |
| 10K neurons, 1% active | 10K ops/step | ~100 ops/step |
| 1M neurons, 0.1% active | 1M ops/step | ~1K ops/step |
