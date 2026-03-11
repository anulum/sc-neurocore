# sc-neurocore Quickstart Tutorial

**Version**: 3.10.0
**Target**: New Users
**Time**: 5 Minutes

---

## 1. Minimal Working Example

This tutorial will show you how to create a single **Stochastic Leaky Integrate-and-Fire (LIF) Neuron**, inject current, and observe the resulting spikes.

### The Script: `hello_neurocore.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

# 1. Initialize a Stochastic LIF Neuron
# We'll use a 10ms membrane time constant and 1ms timestep
neuron = StochasticLIFNeuron(
    tau_mem=10.0,
    dt=1.0,
    v_threshold=1.0,
    noise_std=0.05
)

# 2. Simulation Loop
duration = 100 # ms
input_current = 0.15 # Constant current injection
potentials = []
spikes = []

print("Running simulation...")
for t in range(duration):
    spike = neuron.step(input_current)
    potentials.append(neuron.v)
    spikes.append(spike)

# 3. Visualize Results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(potentials, label='Membrane Potential (V)')
plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
plt.ylabel('Voltage')
plt.legend()
plt.title('sc-neurocore: Single Neuron Dynamics')

plt.subplot(2, 1, 2)
plt.stem(spikes, linefmt='g-', markerfmt='go', basefmt=' ')
plt.ylabel('Spike')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.savefig('quickstart_output.png')
print("Simulation complete. Results saved to quickstart_output.png")
```

---

## 2. Running the Example

1.  Ensure you have installed `sc-neurocore` (`pip install sc-neurocore`).
2.  Save the code above to `hello_neurocore.py`.
3.  Execute the script:
    ```bash
    python hello_neurocore.py
    ```
4.  Check the `quickstart_output.png` file. You should see the membrane potential gradually rising and resetting whenever it hits the 1.0 threshold, with occasional jitter from the `noise_std` parameter.

---

## 3. What’s Happening?

*   **`StochasticLIFNeuron`**: This is the core computational unit. It integrates input over time (leaky integration) and generates a discrete event (spike) when its internal state exceeds a threshold.
*   **`step(input)`**: This method advances the neuron by one `dt`. It handles the integration, noise injection, threshold check, and reset logic.
*   **Stochasticity**: By setting `noise_std > 0`, we simulate biological variability, which is a key feature of the SCPN framework's lower layers (L1-L2).

---

## 4. Next Steps

*   **[Neuromorphic Computing Primer](../research/NEUROMORPHIC_COMPUTING_PRIMER.md)**: Learn why we use spikes instead of continuous values.
*   **[Layer-by-Layer Guide](LAYER_BY_LAYER_GUIDE.md)**: Explore more complex neuron types like Izhikevich or Q-bit neurons.
*   **[Tutorial Notebooks]**: Check the `notebooks/` directory for interactive examples of learning rules and pattern recognition.
