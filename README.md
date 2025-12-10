# SC-NeuroCore

SC-NeuroCore is a small library of **stochastic neuron and synapse models**
designed as building blocks for **stochastic computing** and **neuromorphic**
experiments.

The initial focus is **software-only**:
- Fast Python implementations for prototyping and education.
- Clean interfaces that can later be mapped to FPGA / ASIC / p-bit hardware.

## Features (initial phase)

- `StochasticLIFNeuron` – discrete-time, noisy leaky integrate-and-fire neuron.
- `SCIzhikevichNeuron` – stochastic variant of the Izhikevich neuron.
- Utility functions for:
  - generating random bitstreams,
  - converting probabilities ↔ bitstreams,
  - measuring firing statistics.

## Installation

```bash
git clone https://github.com/<your-user>/sc-neurocore.git
cd sc-neurocore
pip install -e .
```

## Quick Example

```python
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
import numpy as np

neuron = StochasticLIFNeuron(
    v_rest=0.0,
    v_reset=0.0,
    v_threshold=1.0,
    tau_mem=20.0,
    dt=1.0,
    noise_std=0.05
)

T = 1000
input_current = 0.06 * np.ones(T) # constant input
spikes = []

for t in range(T):
    s = neuron.step(input_current[t])
    spikes.append(s)

firing_rate_hz = np.sum(spikes) / (T * neuron.dt) * 1000.0
print("Firing rate ~", firing_rate_hz, "Hz")
```
