# Tutorial 65: Event-Driven Asynchronous Simulation

Clock-driven simulation updates every neuron at every timestep — O(N)
per step regardless of activity. At 1% firing rate, 99% of compute is
wasted on idle neurons. Event-driven simulation updates only neurons
that receive events, achieving massive speedups for sparse networks.

This maps directly to SC-NeuroCore's event-driven FPGA RTL: the same
principle that makes software simulation fast also makes hardware
implementation power-efficient.

## The Problem

| Scale | Activity | Clock-driven ops/step | Event-driven ops/step |
|-------|----------|----------------------|----------------------|
| 1K neurons | 50% | 1,000 | ~500 |
| 10K neurons | 1% | 10,000 | ~100 |
| 100K neurons | 0.1% | 100,000 | ~100 |
| 1M neurons | 0.01% | 1,000,000 | ~100 |

For large, sparse networks (the biologically realistic regime), event-
driven simulation is 1000-10000× more efficient.

## Event-Driven Simulation

```python
from sc_neurocore.event_driven import EventDrivenSimulator
import numpy as np

# Define connectivity: (source, target, weight, delay_ms)
rng = np.random.default_rng(42)
n_neurons = 10000
n_synapses = 100000  # 10 connections per neuron average
connections = [
    (rng.integers(0, n_neurons), rng.integers(0, n_neurons),
     rng.uniform(0.01, 0.1), rng.uniform(0.5, 5.0))
    for _ in range(n_synapses)
]

sim = EventDrivenSimulator(
    n_neurons=n_neurons,
    connectivity=connections,
    threshold=1.0,
    tau_mem=20.0,  # membrane time constant (ms)
)

# Inject initial spikes to start activity
sim.inject_spikes([(0.0, i) for i in range(100)])  # first 100 neurons fire at t=0

# Simulate 1 second
spikes, stats = sim.run(duration=1000.0)

print(stats.summary())
# Typical output:
# EventDriven: 5234 spikes, 52340 events processed
# Neurons updated: 8721 / 10000 (87.2%)
# Events skipped (subthreshold): 47106
# Effective speedup vs clock-driven: ~190x
```

## How It Works

1. **Priority queue (min-heap):** All pending events sorted by delivery
   time. Pop the earliest event each iteration.

2. **Analytical decay:** Between events, membrane voltage decays
   exponentially: `v(t) = v(t0) * exp(-(t - t0) / tau_m)`. No need
   to step through intermediate timesteps.

3. **Spike propagation:** When a neuron fires, new events are pushed
   for all postsynaptic targets at `t_fire + delay`.

4. **Idle neurons:** Never touched until they receive an event. A neuron
   that receives no input for 100 ms costs zero compute.

```
Algorithm:
  while event_queue not empty and t < duration:
      (t, target, weight) = event_queue.pop_min()
      v[target] = v[target] * exp(-(t - last_update[target]) / tau_m) + weight
      last_update[target] = t
      if v[target] > threshold:
          v[target] = v_reset
          for (post, w, d) in synapses[target]:
              event_queue.push((t + d, post, w))
          record_spike(target, t)
```

## Injecting Input

```python
# External spike train
sim.inject_spikes([
    (0.0, 0),    # neuron 0 at t=0
    (5.0, 1),    # neuron 1 at t=5ms
    (10.0, 2),   # neuron 2 at t=10ms
])

# Current pulses (converted to equivalent spike input)
sim.inject_current([
    (0.0, 0, 0.5),   # 0.5 nA to neuron 0 at t=0
    (1.0, 0, 0.3),   # 0.3 nA to neuron 0 at t=1ms
])
```

## FPGA: Event-Driven RTL

SC-NeuroCore's FPGA modules implement the same event-driven principle
in hardware:

| Module | Function | Benefit |
|--------|----------|---------|
| `aer_encoder.v` | Converts parallel spikes to AER packets | Only active neurons generate traffic |
| `event_neuron.v` | Updates membrane only on input events | Zero power when idle |
| `spike_router.v` | Routes AER packets to target neurons | Bandwidth proportional to activity |

Measured power savings vs clock-driven RTL:

| Activity Rate | Clock-Driven Toggles | Event-Driven Toggles | Reduction |
|--------------|---------------------|---------------------|-----------|
| 10% | baseline | 6.5× fewer | 85% |
| 1% | baseline | 15× fewer | 93% |
| 0.1% | baseline | 39× fewer | 97% |

These are measured toggle counts from RTL simulation, not estimates.

## When to Use Event-Driven vs Clock-Driven

**Event-driven** is better when:
- Network is large (>10K neurons)
- Activity is sparse (<5% firing rate)
- Delays are heterogeneous
- Power efficiency matters (FPGA/ASIC deployment)

**Clock-driven** is better when:
- Network is small (<1K neurons)
- Activity is dense (>20% firing rate)
- You need exact timing alignment with external signals
- Debugging (deterministic step-by-step execution)

## Integration with Studio

The Studio's E-I Network view uses clock-driven simulation (simpler
for interactive parameter exploration). For large-scale deployment,
switch to event-driven:

```python
# In the Studio, design your network on the Canvas
# Export as NIR or project JSON
# Then use event-driven simulation for production runs:

from sc_neurocore.event_driven import EventDrivenSimulator

sim = EventDrivenSimulator.from_nir("my_network.nir.json")
spikes, stats = sim.run(duration=10000.0)  # 10 seconds, no memory issues
```

## References

- Brette et al. (2007). "Simulation of networks of spiking neurons: A
  review of tools and strategies." J. Comp. Neurosci. 23(3):349-398.
- Morrison et al. (2007). "Exact subthreshold integration with
  continuous spike times in discrete-time neural network simulations."
  Neural Computation 19(1):47-79.
- Boahen (2000). "Point-to-point connectivity between neuromorphic
  chips using address events." IEEE TCAS-II 47(5):416-434.
