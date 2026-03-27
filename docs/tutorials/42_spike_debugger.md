<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 42: Spike-Level Debugging

When your SNN gives wrong answers, the spike debugger tells you where
and why. Record full execution traces, compare two runs, find the first
divergence point, and trace causal spike chains backward through time.

## Why Spike-Level Debugging

Standard debugging (print loss, check gradients) tells you *that*
training isn't working. Spike debugging tells you *why*:

| Symptom | Standard Debug | Spike Debug |
|---------|---------------|-------------|
| Loss doesn't decrease | "Gradients are small" | "Layer 2 is dead: 90% neurons never fire" |
| Accuracy plateaus | "Try different LR" | "Neurons 34-67 have identical spike patterns" |
| Hardware mismatch | "Output differs" | "First divergence at t=142, neuron 7, dV=0.003" |

## 1. Record an Execution Trace

```python
from sc_neurocore.debug import SpikeTracer
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

pop = Population(HodgkinHuxleyNeuron, n=10, label="hh")
drive = PoissonInput(n=10, rate_hz=200.0, weight=8.0, dt=0.001, seed=42)
net = Network(pop, drive, SpikeMonitor(pop))

tracer = SpikeTracer(net)
trace = tracer.run(duration=0.05, dt=0.001)
print(f"{trace.n_neurons} neurons, {trace.n_steps} steps, {trace.spike_count} spikes")
```

The trace records every neuron's membrane voltage, input current, and
spike times at every timestep.

## 2. Inspect a Single Neuron

```python
nt = trace.neuron_trace(0)
print(f"Neuron 0: {len(nt['spike_times'])} spikes")
print(f"  Voltage range: [{nt['voltages'].min():.1f}, {nt['voltages'].max():.1f}]")
print(f"  Mean input current: {nt['currents'].mean():.2f}")
```

## 3. Find Divergence Between Two Runs

Compare Python vs Verilog, or float32 vs Q8.8:

```python
from sc_neurocore.debug import find_divergence

pop2 = Population(HodgkinHuxleyNeuron, n=10, label="hh2")
net2 = Network(
    pop2,
    PoissonInput(n=10, rate_hz=200.0, weight=8.0, dt=0.001, seed=99),
    SpikeMonitor(pop2),
)
trace2 = SpikeTracer(net2).run(duration=0.05, dt=0.001)

div = find_divergence(trace, trace2)
if div:
    print(f"First divergence at t={div.timestep}, neuron={div.neuron_id}")
    print(f"  Voltage diff: {div.voltage_diff:.6f}")
```

## 4. Trace Causal Chain

Why did neuron 0 fire? Trace backward through synaptic inputs:

```python
from sc_neurocore.debug import causal_chain

spike_t = trace.spike_times(0)
if len(spike_t) > 0:
    chain = causal_chain(trace, neuron_id=0, timestep=int(spike_t[0]), max_depth=5)
    for depth, e in enumerate(chain):
        indent = "  " * depth
        print(f"{indent}t={e.timestep}, neuron={e.neuron_id}, "
              f"I={e.input_current:.2f}, spike={'yes' if e.spiked else 'no'}")
```

## Use Cases

| Scenario | Tool |
|----------|------|
| ANN-to-SNN conversion divergence | `find_divergence()` |
| Hardware verification (Python vs Verilog) | `find_divergence()` |
| Dead/saturated neuron diagnosis | `neuron_trace()` |
| Unexpected firing patterns | `causal_chain()` |
| Q8.8 precision analysis | `find_divergence()` |

## References

- Brette et al. (2007). "Simulation of networks of spiking neurons:
  A review of tools and strategies." J. Comp. Neurosci.
