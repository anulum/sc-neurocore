<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Creating Custom Neuron Models

Extend SC-NeuroCore with your own neuron dynamics. This tutorial
shows how to implement a neuron model that integrates with layers,
the compiler, and the hardware generation pipeline.

**Prerequisites**: `pip install sc-neurocore numpy`

## 1. The neuron interface

Every SC-NeuroCore neuron implements a minimal interface:

```python
class MyNeuron:
    def step(self, **kwargs) -> tuple:
        """Advance one timestep. Return (spike, state)."""
        ...

    def reset(self):
        """Reset to initial conditions."""
        ...
```

The `step()` method takes input (current, bitstream, or probability)
and returns a spike flag plus internal state. That's it.

## 2. Example: Adaptive Exponential (AdEx) neuron

The AdEx model (Brette & Gerstner, 2005) adds exponential
voltage amplification and a slow adaptation variable:

```
dV/dt = -g_L(V - E_L) + g_L·Δ_T·exp((V - V_T)/Δ_T) - w + I
dw/dt = (a·(V - E_L) - w) / τ_w

if V ≥ V_peak: V → V_reset, w → w + b
```

### SC-compatible implementation

```python
import numpy as np
from dataclasses import dataclass, field

@dataclass
class SCAdExNeuron:
    """Adaptive Exponential neuron for stochastic computing.

    Parameters from Brette & Gerstner 2005, Table 1 (regular spiking).
    All voltages in abstract units normalised to [0, 1] for SC.
    """
    # Normalised parameters (original mV values / 100)
    E_L: float = 0.30       # -70 mV → 0.30
    V_T: float = 0.50       # -50 mV → 0.50
    V_reset: float = 0.28   # -72 mV → 0.28
    V_peak: float = 0.80    # -20 mV → 0.80
    delta_T: float = 0.02   # 2 mV → 0.02
    g_L: float = 0.10       # leak conductance
    a: float = 0.02         # subthreshold adaptation
    b: float = 0.05         # spike-triggered adaptation
    tau_w: float = 30.0     # adaptation time constant (steps)
    dt: float = 1.0         # timestep

    V: float = field(init=False)
    w: float = field(init=False)

    def __post_init__(self):
        self.V = self.E_L
        self.w = 0.0

    def step(self, I=0.0):
        """One timestep. I is input current in [0, 1]."""
        # Exponential term (clamp to prevent overflow)
        exp_arg = min((self.V - self.V_T) / max(self.delta_T, 1e-6), 5.0)
        exp_term = self.delta_T * np.exp(exp_arg)

        # Voltage update
        dV = (-self.g_L * (self.V - self.E_L)
              + self.g_L * exp_term
              - self.w
              + I) * self.dt
        self.V += dV

        # Adaptation update
        dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w * self.dt
        self.w += dw

        # Spike check
        spike = self.V >= self.V_peak
        if spike:
            self.V = self.V_reset
            self.w += self.b

        # Clamp to SC range
        self.V = np.clip(self.V, 0.0, 1.0)
        self.w = max(0.0, self.w)

        return int(spike), self.V

    def reset(self):
        self.V = self.E_L
        self.w = 0.0
```

## 3. Test the neuron

```python
neuron = SCAdExNeuron()
spikes = []
voltages = []
adaptations = []

for t in range(300):
    I = 0.15 if 50 < t < 250 else 0.0
    spike, v = neuron.step(I=I)
    spikes.append(spike)
    voltages.append(v)
    adaptations.append(neuron.w)

print(f"Total spikes: {sum(spikes)}")
print(f"Final adaptation: {neuron.w:.4f}")
```

The adaptation variable `w` increases after each spike, causing
the inter-spike interval to grow — spike-frequency adaptation,
a hallmark of the AdEx model.

## 4. Plot the dynamics

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(voltages, linewidth=0.8)
spike_times = [i for i, s in enumerate(spikes) if s]
axes[0].scatter(spike_times, [0.8] * len(spike_times), c="red", s=10, zorder=5)
axes[0].set_ylabel("Voltage")
axes[0].set_title("SCAdExNeuron — Spike Frequency Adaptation")

axes[1].plot(adaptations, linewidth=0.8, color="tab:orange")
axes[1].set_ylabel("Adaptation w")

axes[2].fill_between(range(300), [0.15 if 50 < t < 250 else 0 for t in range(300)],
                     alpha=0.3, color="tab:green")
axes[2].set_ylabel("Input I")
axes[2].set_xlabel("Time step")

plt.tight_layout()
plt.savefig("adex_neuron.png", dpi=150)
```

## 5. Integrate into a layer

Wrap the custom neuron into `SCDenseLayer`-compatible form:

```python
from sc_neurocore import SCSynapse

class AdExLayer:
    """Dense layer using AdEx neurons."""

    def __init__(self, n_inputs, n_neurons, length=256):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.length = length
        self.neurons = [SCAdExNeuron() for _ in range(n_neurons)]
        self.weights = np.random.uniform(0.1, 0.9, (n_neurons, n_inputs))

    def forward(self, x):
        """Forward pass: input probabilities → output firing rates."""
        x = np.clip(x, 0.01, 0.99)
        rates = np.zeros(self.n_neurons)

        for step in range(self.length):
            # Generate input bits
            input_bits = (np.random.rand(self.n_inputs) < x).astype(int)

            for n in range(self.n_neurons):
                # Weighted sum via stochastic AND + count
                weighted = 0
                for i in range(self.n_inputs):
                    w_bit = int(np.random.rand() < self.weights[n, i])
                    weighted += input_bits[i] & w_bit
                I = weighted / self.n_inputs

                spike, _ = self.neurons[n].step(I=I)
                rates[n] += spike

        rates /= self.length
        return rates

    def reset(self):
        for n in self.neurons:
            n.reset()

# Test
adex_layer = AdExLayer(n_inputs=10, n_neurons=5, length=128)
test_input = np.random.uniform(0.2, 0.8, size=10)
output = adex_layer.forward(test_input)
print(f"AdEx layer output: {output}")
```

## 6. Example: Conductance-based neuron

A more biophysical model with excitatory and inhibitory conductances:

```python
@dataclass
class SCConductanceNeuron:
    """Conductance-based LIF for SC.

    Two input channels (excitatory, inhibitory) with reversal potentials.
    """
    V_rest: float = 0.30
    V_E: float = 0.90       # excitatory reversal
    V_I: float = 0.10       # inhibitory reversal
    V_threshold: float = 0.60
    V_reset: float = 0.30
    tau_m: float = 20.0
    tau_E: float = 5.0
    tau_I: float = 10.0
    dt: float = 1.0

    V: float = field(init=False)
    g_E: float = field(init=False)
    g_I: float = field(init=False)

    def __post_init__(self):
        self.V = self.V_rest
        self.g_E = 0.0
        self.g_I = 0.0

    def step(self, I_exc=0.0, I_inh=0.0):
        """One timestep with excitatory and inhibitory input."""
        # Conductance decay
        self.g_E += (-self.g_E / self.tau_E + I_exc) * self.dt
        self.g_I += (-self.g_I / self.tau_I + I_inh) * self.dt
        self.g_E = max(0.0, self.g_E)
        self.g_I = max(0.0, self.g_I)

        # Voltage update
        I_syn = self.g_E * (self.V_E - self.V) + self.g_I * (self.V_I - self.V)
        dV = (-(self.V - self.V_rest) / self.tau_m + I_syn) * self.dt
        self.V += dV
        self.V = np.clip(self.V, 0.0, 1.0)

        spike = self.V >= self.V_threshold
        if spike:
            self.V = self.V_reset

        return int(spike), self.V

    def reset(self):
        self.V = self.V_rest
        self.g_E = 0.0
        self.g_I = 0.0
```

## 7. Fixed-point variant for hardware

For FPGA deployment, implement the same model in Q8.8:

```python
@dataclass
class FixedPointAdExNeuron:
    """Q8.8 fixed-point AdEx for FPGA synthesis verification."""

    E_L: int = 77       # 0.30 × 256
    V_T: int = 128      # 0.50 × 256
    V_reset: int = 72   # 0.28 × 256
    V_peak: int = 205   # 0.80 × 256
    g_L: int = 26       # 0.10 × 256
    a: int = 5          # 0.02 × 256
    b: int = 13         # 0.05 × 256

    V: int = field(init=False)
    w: int = field(init=False)

    def __post_init__(self):
        self.V = self.E_L
        self.w = 0

    def _clamp_s16(self, x):
        return max(-32768, min(32767, x))

    def step(self, I_q8=0):
        """One timestep. I_q8 is Q8.8 input current."""
        # Leak term: g_L * (V - E_L) >> 8
        leak = self._clamp_s16((self.g_L * (self.V - self.E_L)) >> 8)

        # Simplified exponential: linear above V_T (hardware-friendly)
        if self.V > self.V_T:
            exp_term = self._clamp_s16((self.g_L * (self.V - self.V_T)) >> 7)
        else:
            exp_term = 0

        # Voltage update
        dV = -leak + exp_term - (self.w >> 2) + (I_q8 >> 1)
        self.V = self._clamp_s16(self.V + dV)

        # Adaptation update: dw = (a*(V-E_L) - w) / tau_w
        dw = self._clamp_s16(
            ((self.a * (self.V - self.E_L)) >> 8) - (self.w >> 5)
        )
        self.w = self._clamp_s16(self.w + dw)

        # Spike
        spike = self.V >= self.V_peak
        if spike:
            self.V = self.V_reset
            self.w = self._clamp_s16(self.w + self.b)

        return int(spike), self.V

    def reset(self):
        self.V = self.E_L
        self.w = 0
```

## 8. Verification: float vs fixed-point

```python
def verify_adex_parity(n_steps=200):
    """Compare float and Q8.8 AdEx outputs."""
    float_n = SCAdExNeuron()
    fp_n = FixedPointAdExNeuron()

    float_spikes = []
    fp_spikes = []

    for t in range(n_steps):
        I = 0.15 if 50 < t < 150 else 0.0
        I_q8 = int(I * 256)

        fs, _ = float_n.step(I=I)
        qs, _ = fp_n.step(I_q8=I_q8)
        float_spikes.append(fs)
        fp_spikes.append(qs)

    float_count = sum(float_spikes)
    fp_count = sum(fp_spikes)
    print(f"Float spikes: {float_count}")
    print(f"Q8.8 spikes:  {fp_count}")
    print(f"Difference:   {abs(float_count - fp_count)}")

verify_adex_parity()
```

## What you learned

- Custom neurons implement `step()` → (spike, state) and `reset()`
- AdEx model: exponential amplification + spike-frequency adaptation
- Conductance-based model: separate excitatory/inhibitory channels
- Wrap custom neurons into a layer class for network integration
- Fixed-point variant for FPGA: replace float ops with Q8.8 arithmetic
- Always verify float vs fixed-point before hardware deployment

## Next steps

- Add the custom neuron to the compiler for HDL generation
- Implement Hodgkin-Huxley channels in the conductance model
- Create a fixed-point LFSR-based stochastic variant
- Compare AdEx dynamics against Brian2 reference output
