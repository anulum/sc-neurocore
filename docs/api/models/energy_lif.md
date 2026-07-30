# EnergyLIFNeuron

`EnergyLIFNeuron` is the source identity for the two-state energy-based LIF of Fardet and Levina (2020), DOI `10.1371/journal.pcbi.1008503`. It follows the authors' Brian example: coupled voltage/energy dynamics, simultaneous fourth-order Runge-Kutta integration at 0.1 ms, a strict `v > v_threshold` and `epsilon > epsilon_c` event gate, voltage reset, and subtraction of the per-event energy cost.

The effective leak reversal is

```text
E_L = E_0 + (E_u - E_0) * (1 - epsilon / epsilon_0)
```

and the coupled derivatives are

```text
dV/dt       = (g_L * (E_L - V) + I) / C_m
depsilon/dt = ((1 - epsilon / (alpha * epsilon_0))^3
               - (V - E_f) / (E_d - E_f)) / tau_e
```

The pinned profile starts at `V=-61 mV`, `epsilon=0.32` with `C_m=100 pF`, `g_L=9 nS`, `E_0=-62.5 mV`, `E_u=-58.5 mV`, `E_d=-40 mV`, `E_f=-62 mV`, `V_th=-59 mV`, `V_reset=-62 mV`, `alpha=1`, `epsilon_0=0.5`, `epsilon_c=0.18`, `delta=0.01`, and `tau_e=200 ms`.

Python, the Rust engine and safety lane, Julia, Go shared library, and Mojo shared library expose the same complete state/configuration contract. The 512-step mixed-current receipt records eight events and SHA-256 `fc0aa0c…23d3`; native traces remain within `2e-12` of Python. Paired TOML/JSON schemas, a signed-Q32.32 pinned-profile RTL co-simulation, Yosys synthesis, and a bounded reset proof close the declared H1 evidence rung. These do not claim transition-property formal proof, universal binary64 equivalence, timing, PPA, or device validation.

The previous normalized exact-flow recurrence is preserved separately as [`SCNormalizedEnergyLIFNeuron`](sc_normalized_energy_lif.md). It is not attributed to Fardet and Levina and does not add a source-catalogue count.

```python
from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron

neuron = EnergyLIFNeuron()
event = neuron.step(80.0)
```
