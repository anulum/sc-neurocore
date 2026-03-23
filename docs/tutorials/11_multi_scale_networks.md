<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Multi-Scale Network Simulation

Build hierarchical SNN architectures that combine multiple neuron
types, layer types, and timescales. This tutorial shows how to wire
SC-NeuroCore components into networks that span from single-neuron
biophysics to population-level dynamics.

**Prerequisites**: `pip install sc-neurocore matplotlib`

## 1. The multi-scale challenge

Real neural circuits operate across scales:

| Scale | SC-NeuroCore component | Timescale |
|-------|----------------------|-----------|
| Synapse | `BitstreamSynapse`, `StochasticSTDPSynapse` | ~1 ms (bitstream step) |
| Neuron | `StochasticLIFNeuron`, `SCIzhikevichNeuron` | 1-20 ms (membrane τ) |
| Microcircuit | `SCDenseLayer`, `SCRecurrentLayer` | 10-100 ms |
| Population | `VectorizedSCLayer`, ensembles | 100 ms - 1 s |
| Network | Multi-layer pipeline | seconds |

SC-NeuroCore lets you mix these freely because every component
communicates through the same interface: spike probability ∈ [0, 1].

## 2. Build a cortical column model

A simplified cortical column with excitatory and inhibitory populations:

```python
import numpy as np
from sc_neurocore import VectorizedSCLayer

np.random.seed(42)
L = 256

# Excitatory population: 80 neurons
exc = VectorizedSCLayer(n_inputs=20, n_neurons=80, length=L)

# Inhibitory population: 20 neurons (4:1 ratio, Dale's principle)
inh = VectorizedSCLayer(n_inputs=80, n_neurons=20, length=L)

# Recurrent excitatory→excitatory
exc_rec = VectorizedSCLayer(n_inputs=80, n_neurons=80, length=L)
# Scale recurrent weights down to prevent runaway excitation
exc_rec.weights *= 0.3
exc_rec._refresh_packed_weights()

# Inhibitory→excitatory feedback
inh_fb = VectorizedSCLayer(n_inputs=20, n_neurons=80, length=L)
```

## 3. Simulate one timestep

```python
def column_step(external_input):
    """One step of the cortical column.

    Returns excitatory and inhibitory firing rates.
    """
    # Excitatory receives external input
    exc_out = exc.forward(external_input)
    exc_out = np.clip(exc_out, 0.01, 0.99)

    # Inhibitory receives excitatory output
    inh_out = inh.forward(exc_out)
    inh_out = np.clip(inh_out, 0.01, 0.99)

    # Recurrent excitation (positive feedback)
    rec = exc_rec.forward(exc_out)
    rec = np.clip(rec, 0.01, 0.99)

    # Inhibitory feedback (negative, subtract from excitatory drive)
    fb = inh_fb.forward(inh_out)
    # Combine: clip to valid SC range
    combined = np.clip(rec * 0.6 - fb * 0.4 + external_input * 0.4, 0.01, 0.99)

    return exc_out, inh_out, combined
```

## 4. Run the simulation

```python
N_STEPS = 500
exc_rates = np.zeros((N_STEPS, 80))
inh_rates = np.zeros((N_STEPS, 20))

# External stimulus: ramp up, hold, ramp down
stimulus = np.zeros((N_STEPS, 20))
stimulus[50:150] = np.linspace(0, 0.8, 100)[:, None] * np.ones(20)
stimulus[150:350] = 0.8
stimulus[350:450] = np.linspace(0.8, 0, 100)[:, None] * np.ones(20)

for t in range(N_STEPS):
    e, i, _ = column_step(np.clip(stimulus[t], 0.01, 0.99))
    exc_rates[t] = e
    inh_rates[t] = i

print(f"Exc mean rate: {exc_rates[150:350].mean():.3f}")
print(f"Inh mean rate: {inh_rates[150:350].mean():.3f}")
```

## 5. Visualise population activity

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Raster plot (threshold at 0.5 firing rate)
exc_spikes = exc_rates > 0.5
for n in range(80):
    spike_times = np.where(exc_spikes[:, n])[0]
    axes[0].scatter(spike_times, np.full_like(spike_times, n),
                    s=0.5, c="tab:blue", marker="|")
axes[0].set_ylabel("Exc neuron #")
axes[0].set_title("Excitatory Raster")

inh_spk = inh_rates > 0.5
for n in range(20):
    spike_times = np.where(inh_spk[:, n])[0]
    axes[1].scatter(spike_times, np.full_like(spike_times, n),
                    s=0.5, c="tab:red", marker="|")
axes[1].set_ylabel("Inh neuron #")
axes[1].set_title("Inhibitory Raster")

axes[2].plot(exc_rates.mean(axis=1), label="Exc mean", linewidth=0.8)
axes[2].plot(inh_rates.mean(axis=1), label="Inh mean", linewidth=0.8)
axes[2].plot(stimulus[:, 0], label="Stimulus", linewidth=0.8, linestyle="--")
axes[2].set_xlabel("Time step")
axes[2].set_ylabel("Firing rate")
axes[2].legend()

plt.tight_layout()
plt.savefig("cortical_column.png", dpi=150)
```

## 6. Hierarchical: column → area → network

Stack cortical columns into a hierarchy:

```python
class CorticalArea:
    """A brain area with N_COL cortical columns."""

    def __init__(self, n_columns, n_inputs, neurons_per_col=80, length=256):
        self.columns = []
        for _ in range(n_columns):
            col = {
                "exc": VectorizedSCLayer(n_inputs, neurons_per_col, length),
                "inh": VectorizedSCLayer(neurons_per_col, neurons_per_col // 4, length),
            }
            self.columns.append(col)
        self.n_col = n_columns
        self.neurons_per_col = neurons_per_col

    def forward(self, inputs):
        """Process input through all columns, return concatenated output."""
        outputs = []
        for col in self.columns:
            e = col["exc"].forward(np.clip(inputs, 0.01, 0.99))
            e = np.clip(e, 0.01, 0.99)
            outputs.append(e)
        return np.concatenate(outputs)

# Two-area hierarchy
v1 = CorticalArea(n_columns=4, n_inputs=50, neurons_per_col=40, length=128)
v2 = CorticalArea(n_columns=2, n_inputs=160, neurons_per_col=40, length=128)  # 4*40 inputs

test_input = np.random.uniform(0.1, 0.9, size=50)
v1_out = v1.forward(test_input)
v2_out = v2.forward(v1_out)
print(f"V1 output: {v1_out.shape} ({v1_out.mean():.3f})")
print(f"V2 output: {v2_out.shape} ({v2_out.mean():.3f})")
```

## 7. Mixing neuron types

Use Izhikevich neurons for biophysically detailed populations and
LIF for the fast, hardware-mappable populations:

```python
from sc_neurocore import SCIzhikevichNeuron, StochasticLIFNeuron

# Detailed population: Izhikevich regular spiking
detailed_neurons = [
    SCIzhikevichNeuron(a=0.02, b=0.2, c=-65, d=8)
    for _ in range(10)
]

# Fast population: LIF
fast_neurons = [StochasticLIFNeuron(length=256) for _ in range(50)]

# Simulate: Izhikevich neurons drive LIF neurons through a weight matrix
W = np.random.uniform(0.1, 0.9, size=(50, 10))

for t in range(100):
    I_ext = 10.0 if 20 < t < 80 else 0.0

    # Detailed population
    iz_spikes = []
    for n in detailed_neurons:
        spike = n.step(I=I_ext)
        iz_spikes.append(float(spike))
    iz_rates = np.array(iz_spikes)

    # Weight matrix transforms Izhikevich spikes → LIF input probabilities
    lif_input = np.clip(W @ iz_rates, 0.01, 0.99)

    # Fast population
    lif_spikes = []
    for i, n in enumerate(fast_neurons):
        spike, _ = n.step(x_value=int(lif_input[i] * 255))
        lif_spikes.append(spike)
```

## 8. Analysis: population statistics

```python
from sc_neurocore.analysis import firing_rate, cv_isi

# Collect spike trains from a simulation
# spike_trains shape: (n_neurons, n_steps)
spike_trains = (exc_rates > 0.5).astype(float)

rates = firing_rate(spike_trains, dt=1.0)
cvs = cv_isi(spike_trains)

print(f"Mean firing rate: {rates.mean():.2f} Hz")
print(f"Mean CV(ISI): {cvs[~np.isnan(cvs)].mean():.2f}")
print(f"  CV ≈ 1.0 indicates Poisson-like irregularity")
```

## 9. Recording and saving

Use `BitstreamSpikeRecorder` to capture full spike trains for
offline analysis:

```python
from sc_neurocore import BitstreamSpikeRecorder

recorder = BitstreamSpikeRecorder(n_neurons=80)

for t in range(N_STEPS):
    e, _, _ = column_step(np.clip(stimulus[t], 0.01, 0.99))
    spikes = (e > 0.5).astype(int)
    recorder.record(spikes)

data = recorder.get_data()
print(f"Recorded: {data.shape} (neurons × timesteps)")
np.save("column_spikes.npy", data)
```

## What you learned

- SC components communicate through spike probability ∈ [0, 1]
- Cortical column model: excitatory + inhibitory + recurrent + feedback
- Hierarchical areas stack columns with inter-area connections
- Mix Izhikevich (biophysical) and LIF (hardware) neurons in one network
- Population statistics: firing rate, CV(ISI), raster plots
- `BitstreamSpikeRecorder` captures data for offline analysis

## Next steps

- Add STDP to recurrent connections for self-organising topology
- Use the Rust engine for large-scale simulations (>10K neurons)
- Compare population statistics against Brian2 Brunel network
- Deploy the cortical column on FPGA via the HDL generator
