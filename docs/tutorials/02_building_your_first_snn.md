<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Building Your First Spiking Neural Network

**Prerequisites**: Python 3.10+, `pip install sc-neurocore`

SC-NeuroCore implements spiking neural networks (SNNs) using stochastic
computing (SC). Inputs, weights, and activations are encoded as Bernoulli
bitstreams where the density of 1s represents probability. Multiplication
reduces to a single AND gate. This tutorial builds up from a single neuron
to a multi-layer network.

## LIF dynamics

Every neuron in this tutorial is a discrete-time leaky integrate-and-fire
(LIF) model with optional Gaussian noise:

$$
V[t+1] = V[t] - \frac{V[t] - V_\text{rest}}{\tau_\text{mem}} \cdot dt + R \cdot I_\text{syn} \cdot dt + \mathcal{N}(0,\, \sigma^2)
$$

When $V \geq V_\theta$, the neuron emits a spike (output = 1), resets to
$V_\text{reset}$, and enters an optional refractory period.

## 1. Single neuron with constant current

```python
from sc_neurocore import StochasticLIFNeuron, BitstreamSpikeRecorder

neuron = StochasticLIFNeuron(
    v_rest=0.0,
    v_reset=0.0,
    v_threshold=1.0,
    tau_mem=20.0,
    dt=1.0,
    noise_std=0.0,
    resistance=1.0,
    seed=42,
)
recorder = BitstreamSpikeRecorder(dt_ms=1.0)

I_const = 0.08  # sub-threshold alone, but integrates over ~13 steps
for _ in range(200):
    spike = neuron.step(I_const)
    recorder.record(spike)

print(f"Spikes: {recorder.total_spikes()}")
print(f"Firing rate: {recorder.firing_rate_hz():.1f} Hz")
```

With `I=0.08` and `tau_mem=20`, the membrane reaches threshold at
$t \approx 1/\ln(1 - I \cdot \tau_\text{mem} / V_\theta)$ steps. Expected
rate is ~60 Hz. Run it and verify.

## 2. Bitstream synapse: SC multiplication

A `BitstreamSynapse` encodes a weight $w \in [w_\text{min}, w_\text{max}]$
as a bitstream. Applying it to a pre-synaptic bitstream performs AND --
the SC equivalent of multiplication:

$$
P(\text{out}=1) \approx P(\text{pre}=1) \cdot P(\text{weight}=1)
$$

```python
import numpy as np
from sc_neurocore import BitstreamSynapse, BitstreamEncoder, bitstream_to_probability

LENGTH = 4096

encoder = BitstreamEncoder(x_min=0.0, x_max=1.0, length=LENGTH, seed=7)
pre_bits = encoder.encode(0.6)  # ~60% ones

synapse = BitstreamSynapse(w_min=0.0, w_max=1.0, length=LENGTH, w=0.5, seed=99)
post_bits = synapse.apply(pre_bits)

p_pre = bitstream_to_probability(pre_bits)
p_post = bitstream_to_probability(post_bits)
p_expected = 0.6 * 0.5

print(f"P(pre)={p_pre:.3f}  P(post)={p_post:.3f}  expected={p_expected:.3f}")
assert abs(p_post - p_expected) < 0.05, "SC multiplication error exceeds 5%"
```

Error scales as $O(1/\sqrt{L})$ for Bernoulli bitstreams. At $L=4096$, the
standard error is ~0.8%.

## 3. Multi-channel current source

`BitstreamCurrentSource` wires multiple input channels through synapses,
computes a bitstream dot-product, and delivers a scalar current $I(t)$ at
each time step:

```python
from sc_neurocore import BitstreamCurrentSource

source = BitstreamCurrentSource(
    x_inputs=[0.5, 0.8, 0.3],
    x_min=0.0, x_max=1.0,
    weight_values=[0.6, 0.4, 0.9],
    w_min=0.0, w_max=1.0,
    length=2048,
    y_min=0.0, y_max=0.1,
    seed=42,
)

currents = [source.step() for _ in range(100)]
avg_I = np.mean(currents)
print(f"Average current over 100 steps: {avg_I:.4f}")
print(f"Full-length estimate: {source.full_current_estimate():.4f}")
```

## 4. Dense layer: 8 inputs, 4 neurons

`SCDenseLayer` bundles a shared `BitstreamCurrentSource` with $N$ LIF
neurons, each with independent noise seeds:

```python
from sc_neurocore import SCDenseLayer

layer = SCDenseLayer(
    n_neurons=4,
    x_inputs=[0.3, 0.7, 0.5, 0.2, 0.9, 0.4, 0.6, 0.8],
    weight_values=[0.5, 0.6, 0.4, 0.3, 0.7, 0.5, 0.8, 0.2],
    x_min=0.0, x_max=1.0,
    w_min=0.0, w_max=1.0,
    length=2048,
    y_min=0.0, y_max=0.1,
    dt_ms=1.0,
    neuron_params={"tau_mem": 20.0, "v_threshold": 1.0, "noise_std": 0.02},
    base_seed=42,
)

layer.run(T=500)

spikes = layer.get_spike_trains()  # shape (4, 500)
print(f"Spike matrix shape: {spikes.shape}")

for stat in layer.summary()["stats"]:
    print(f"  Neuron {stat['neuron']}: {stat['total_spikes']} spikes, "
          f"{stat['firing_rate_hz']:.1f} Hz")
```

Neurons share the same SC current but differ by noise realisations. With
`noise_std=0.02`, firing rate variance across neurons is small (~5%).

## 5. Two-layer network (manual wiring)

SC-NeuroCore does not enforce a fixed graph topology. Connect layers by
feeding one layer's spike rates as the next layer's input probabilities:

```python
from sc_neurocore import SCDenseLayer

inputs = [0.4, 0.7, 0.3, 0.8, 0.6, 0.5, 0.9, 0.2]
weights_L1 = [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.5, 0.6]

layer1 = SCDenseLayer(
    n_neurons=4,
    x_inputs=inputs,
    weight_values=weights_L1,
    x_min=0.0, x_max=1.0,
    w_min=0.0, w_max=1.0,
    length=2048, y_min=0.0, y_max=0.1,
    neuron_params={"tau_mem": 20.0, "noise_std": 0.01},
    base_seed=100,
)
layer1.run(T=500)

rates_L1 = [s["firing_rate_hz"] / 1000.0 for s in layer1.summary()["stats"]]

weights_L2 = [0.6, 0.5, 0.7, 0.4]
layer2 = SCDenseLayer(
    n_neurons=2,
    x_inputs=rates_L1,
    weight_values=weights_L2,
    x_min=0.0, x_max=0.1,  # match L1 output range
    w_min=0.0, w_max=1.0,
    length=2048, y_min=0.0, y_max=0.1,
    neuron_params={"tau_mem": 20.0, "noise_std": 0.01},
    base_seed=200,
)
layer2.run(T=500)

print("Layer 2 output:")
for stat in layer2.summary()["stats"]:
    print(f"  Neuron {stat['neuron']}: {stat['firing_rate_hz']:.1f} Hz")
```

## 6. Correctness check: firing rate vs. analytic prediction

For a deterministic LIF neuron (`noise_std=0`) with constant current $I$
above threshold, the inter-spike interval is:

$$
T_\text{ISI} = -\tau_\text{mem} \cdot \ln\!\left(1 - \frac{V_\theta - V_\text{reset}}{R \cdot I \cdot \tau_\text{mem}}\right)
$$

and expected firing rate $f = 1000 / (T_\text{ISI} \cdot dt_\text{ms})$ Hz.

```python
import math
from sc_neurocore import StochasticLIFNeuron, BitstreamSpikeRecorder

I = 0.08
TAU = 20.0
V_TH = 1.0
DT = 1.0

neuron = StochasticLIFNeuron(
    v_threshold=V_TH, tau_mem=TAU, dt=DT, noise_std=0.0, seed=0
)
rec = BitstreamSpikeRecorder(dt_ms=DT)
for _ in range(5000):
    rec.record(neuron.step(I))

measured_hz = rec.firing_rate_hz()

arg = 1.0 - V_TH / (I * TAU)
if arg > 0:
    t_isi = -TAU * math.log(arg)
    predicted_hz = 1000.0 / (t_isi * DT)
else:
    predicted_hz = 0.0

error_pct = abs(measured_hz - predicted_hz) / predicted_hz * 100
print(f"Predicted: {predicted_hz:.1f} Hz  Measured: {measured_hz:.1f} Hz  "
      f"Error: {error_pct:.1f}%")
assert error_pct < 2.0, f"Firing rate error {error_pct:.1f}% exceeds 2%"
```

Discrete-time Euler integration introduces a small bias (< 2%) that
decreases with smaller `dt`. Verify the ISI distribution matches:

```python
hist, edges = rec.isi_histogram(bins=15)
print(f"ISI peak: {edges[hist.argmax()]:.1f} ms  (predicted: {t_isi:.1f} ms)")
```

## Summary

| Component | Class | Key params |
|---|---|---|
| Neuron | `StochasticLIFNeuron` | `tau_mem`, `v_threshold`, `noise_std`, `refractory_period` |
| Synapse | `BitstreamSynapse` | `w`, `w_min`, `w_max`, `length` |
| Current source | `BitstreamCurrentSource` | `x_inputs`, `weight_values`, `y_min`, `y_max` |
| Layer | `SCDenseLayer` | `n_neurons`, `x_inputs`, `weight_values`, `neuron_params` |
| Recorder | `BitstreamSpikeRecorder` | `dt_ms` |

Next: [Deploy an SNN on FPGA in 20 Minutes](fpga_in_20_minutes.md)
