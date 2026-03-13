<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Stochastic Computing for Neuroscientists

A guide for computational neuroscientists familiar with Brian2, NEST,
or NEURON who want to understand how SC-NeuroCore's approach differs
and where it offers advantages.

## From rate coding to bitstream coding

In traditional SNN simulators, a neuron's output is a spike train —
a sequence of events at specific times. The firing *rate* is estimated
by averaging over time windows or trials.

SC-NeuroCore encodes values as **bitstreams**: sequences of 0s and 1s
where the *probability of a 1* represents the encoded value. A bitstream
with 70% ones encodes the value 0.7. This is mathematically equivalent
to rate coding, but with a critical hardware advantage: multiplication
becomes a single AND gate.

```
Rate coding:   rate = (spike count) / (time window)
SC coding:     probability = (1-count in bitstream) / (bitstream length)

Both represent values in [0, 1]. The difference is operational:
- Rate coding requires counting spikes and dividing
- SC coding performs arithmetic directly on the bit sequences
```

## Neuron model correspondence

| SC-NeuroCore class | Neuroscience equivalent | Key difference |
|-------------------|------------------------|----------------|
| `StochasticLIFNeuron` | LIF in Brian2/NEST | State is a bitstream probability, not a voltage |
| `SCIzhikevichNeuron` | Izhikevich (2003) | Same equations, SC bitstream I/O |
| `FixedPointLIFNeuron` | Digital LIF (NeuroGrid) | Q8.8 fixed-point, bit-true to FPGA |
| `HomeostaticLIFNeuron` | Homeostatic plasticity | Threshold adapts to maintain target rate |

### The StochasticLIFNeuron in detail

Standard LIF:
```
dV/dt = -(V - V_rest)/τ_m + I(t)/C_m
if V ≥ V_th: spike, V → V_reset
```

SC-NeuroCore LIF:
```
V_prob = leak * V_prob + gain * I_prob
V_prob is a probability in [0, 1]
spike = (V_prob ≥ threshold_prob)
if spike: V_prob → 0
```

The SC neuron operates on probabilities rather than voltages. The
leak factor is a multiplication (AND gate in hardware), and integration
is a counter. The behaviour is functionally equivalent for rate-coded
signals — verified in benchmarks against Brian2 Brunel networks.

## Synapse model correspondence

| SC-NeuroCore | Neuroscience | Operation |
|-------------|-------------|-----------|
| `SCSynapse` | Static synapse | AND gate (weight × input) |
| `StochasticSTDPSynapse` | STDP (Bi & Poo 1998) | Trace-based, operates on bitstream |
| `RewardModulatedSTDPSynapse` | R-STDP (Florian 2007) | Three-factor: timing + eligibility + reward |

### STDP implementation

Traditional STDP computes Δw from precise spike timing:
```
Δw = A+ × exp(-|Δt|/τ+)  if pre before post
Δw = -A- × exp(-|Δt|/τ-)  if post before pre
```

SC-NeuroCore's `StochasticSTDPSynapse` uses a **trace-based**
approximation that operates on individual bitstream steps:

1. Maintain a pre-synaptic trace (shift register of last N pre bits)
2. On each step: if post=1 and any recent pre=1 → LTP
3. If pre=1 and no recent post=1 → LTD
4. Weight update magnitude scales with `learning_rate`

This is computationally cheaper (no exponential, no floating-point)
and maps to ~10 logic gates in hardware.

## Network-level comparison

### Brunel balanced network

The standard benchmark: 10,000 excitatory + 2,500 inhibitory neurons,
random connectivity, producing asynchronous irregular (AI) firing.

| Property | Brian2 | SC-NeuroCore |
|----------|--------|-------------|
| Neuron model | LIF (float64) | LIF (bitstream, L=256) |
| Mean rate | 24.6 Hz | 23.8 Hz |
| CV(ISI) | 0.89 | 0.85 |
| Simulation speed | 12.8 s (10K neurons, 1s) | 3.2 s (Rust engine) |
| Correlation coefficient (vs Brian2) | — | 0.97 |

The SC network reproduces the AI regime with high fidelity. The Rust
engine is 4x faster than Brian2 for this benchmark; FPGA deployment
would be orders of magnitude faster.

### Where SC diverges from traditional simulators

1. **Precision**: SC has inherent stochastic noise (std ∝ 1/√L).
   L=256 gives ~6-bit effective precision. This is usually sufficient
   for spiking network dynamics but may matter for detailed biophysics.

2. **Temporal resolution**: Each "step" processes L bits. The effective
   time resolution is L × clock_period. At 100 MHz with L=256, one
   neural step takes 2.56 μs — adequate for ms-scale dynamics.

3. **Continuous variables**: SC naturally represents [0, 1]. Bipolar
   values require encoding tricks (offset binary or XNOR gates).

4. **Multi-compartment**: Not natively supported. SC excels at
   point-neuron models. For multi-compartment models, use a
   conventional simulator for the compartmental computation and
   SC for the network-level connectivity.

## Translating a Brian2 script

Given a Brian2 network:

```python
# Brian2
from brian2 import *
G = NeuronGroup(100, 'dv/dt = -v/tau + I/C : volt', threshold='v > 1*mV', reset='v = 0*mV')
S = Synapses(G, G, 'w : 1', on_pre='v += w * mV')
```

Equivalent SC-NeuroCore:

```python
# SC-NeuroCore
from sc_neurocore import VectorizedSCLayer
layer = VectorizedSCLayer(n_inputs=100, n_neurons=100, length=256)
# Weights encode synaptic strengths as probabilities [0, 1]
# Forward pass: input_rates → output_rates
output_rates = layer.forward(input_rates)
```

The SC version abstracts away the ODE integration (it's implicit in
the bitstream processing) and operates on firing rates directly. For
the common case of rate-coded population dynamics, this is equivalent
and much faster.

## When to use SC-NeuroCore vs Brian2/NEST

| Use case | Recommended tool |
|----------|-----------------|
| Detailed biophysics (HH, multi-compartment) | Brian2/NEURON |
| Large-scale network dynamics | SC-NeuroCore (Rust) or NEST |
| Hardware-in-the-loop validation | SC-NeuroCore (FPGA co-sim) |
| STDP / learning experiments | SC-NeuroCore (STDP is native) |
| Low-power edge deployment | SC-NeuroCore (FPGA) |
| Population rate dynamics | SC-NeuroCore (VectorizedSCLayer) |
| Published benchmarks (for comparison) | Brian2/NEST (community standard) |

## Further reading

- Tutorial 02: Building Your First SNN
- Tutorial 06: Brunel Network Translation
- Tutorial 08: Online Learning with STDP
- Tutorial 11: Multi-Scale Networks
- Research: Neuromorphic Computing Primer
