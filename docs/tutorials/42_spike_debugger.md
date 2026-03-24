<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 42: Spike-Level Debugging

When your SNN gives wrong answers, the spike debugger tells you where
and why. Record full execution traces, compare two runs, find the first
divergence point, and trace causal spike chains backward through time.

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

## 2. Inspect a Single Neuron

```python
nt = trace.neuron_trace(0)
print(f"Neuron 0: {len(nt['spike_times'])} spikes")
print(f"  Voltage: [{nt['voltages'].min():.1f}, {nt['voltages'].max():.1f}]")
```

## 3. Find Divergence Between Two Runs

```python
from sc_neurocore.debug import find_divergence

# Second run with different seed
pop2 = Population(HodgkinHuxleyNeuron, n=10, label="hh2")
net2 = Network(pop2, PoissonInput(n=10, rate_hz=200.0, weight=8.0, dt=0.001, seed=99), SpikeMonitor(pop2))
trace2 = SpikeTracer(net2).run(duration=0.05, dt=0.001)

div = find_divergence(trace, trace2)
if div:
    print(f"Divergence at t={div.timestep}, neuron={div.neuron_id}, dV={div.voltage_diff:.2f}")
```

## 4. Trace Causal Chain

```python
from sc_neurocore.debug import causal_chain

spike_t = trace.spike_times(0)
if len(spike_t) > 0:
    chain = causal_chain(trace, neuron_id=0, timestep=int(spike_t[0]), max_depth=5)
    for e in chain:
        print(f"  t={e.timestep}, n={e.neuron_id}, I={e.input_current:.2f}, spike={e.spiked}")
```

## Use Cases

- ANN-to-SNN conversion: find where spikes diverge from expected
- Hardware verification: compare Python vs Verilog output
- Training diagnostics: identify dead or saturated neurons

## Further Reading

- [Tutorial 31: Network Simulation Engine](31_network_simulation_engine.md)
- [API: Debug](../api/debug.md)
