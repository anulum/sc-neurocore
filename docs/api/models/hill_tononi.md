# HillTononiNeuron

`HillTononiNeuron` implements the hybrid model neuron used by Hill and Tononi
(2005) for their thalamocortical sleep/wake simulations. The catalogue default
is the paper's cortical-excitatory waking profile.

Reference: S. Hill and G. Tononi, *Modeling sleep and wakefulness in the
thalamocortical system*, Journal of Neurophysiology 93, 1671–1698 (2005),
[doi:10.1152/jn.00915.2004](https://doi.org/10.1152/jn.00915.2004).

## Model boundary

The continuous state is

```text
(V, theta, D, m_h, m_T, h_T)
```

with a finite `spike_timer`. The membrane equation combines sodium and
potassium leaks, persistent sodium `I_NaP`, depolarisation-dependent potassium
`I_DK`, optional hyperpolarisation-activated `I_h`, optional low-threshold
calcium `I_T`, and external current. The dynamic threshold relaxes toward
`theta_eq`; crossing it outside the post-spike interval sets both `V` and
`theta` to `E_Na` and begins the source potassium repolarisation pulse.

`D` is the paper's generic depolarisation measure. It is not intracellular
sodium or calcium concentration. The prior SC-NeuroCore six-state HH/Na-pump
recurrence was therefore not a Hill–Tononi implementation and is retained,
without paper attribution, as the internal
`SCSixStateThalamocorticalNeuron` compatibility model.

## Defaults

The default profile uses `dt = 0.25 ms` and classical RK4:

| Quantity | Default | Meaning |
|---|---:|---|
| `V` | -70 mV | membrane potential |
| `theta` | -51 mV | dynamic threshold |
| `D` | 0.001 | depolarisation measure |
| `g_NaL`, `g_KL` | 0.2, 1.0 | wake sodium/potassium leaks |
| `g_NaP`, `g_DK` | 0.5, 0.5 | persistent-sodium and D-dependent potassium conductances |
| `g_h`, `g_T` | 0, 0 | optional currents disabled for this cortical profile |
| `t_spike`, `tau_spike` | 2, 1.75 ms | post-spike potassium pulse |

The optional `I_h` and `I_T` equations are implemented, but enabling both by
default would incorrectly merge the paper's cortical and thalamic cell types.
Network synapses, miniature events, sleep/wake neuromodulation, and topology
remain outside this scalar catalogue model.

## Compute implementations

The same recurrence is implemented in the Python reference, Rust engine and
safety module, Julia, Go, and executable Mojo kernel. Invalid configuration or
non-finite input is rejected before state commit. The Rust engine remains wired
through its PyO3 and `NetworkRunner` dispatch surfaces.

The committed independent receipt runs 768 mixed-drive steps and records two
events with trace SHA-256
`64aaf9659f1c9c3e4233dfd73f5f21f143a1718e2dc29d69a6db657e1d911b9b`.
The source-bound benchmark adds a 200,000-step constant-drive comparison across
all five maintained runtimes.

```python
from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

neuron = HillTononiNeuron()
events = sum(neuron.step(20.0) for _ in range(200_000))
assert events == 538
```

## Hardware boundary

No RTL, synthesis, formal-equivalence, timing, PPA, or device claim is made for
this model. Its exponential gates and candidate-first binary64 RK4 step require
an explicitly specified and independently validated fixed-point approximation
before a silicon rung can be claimed.

See [Hill–Tononi source-fidelity validation](../../validation/hill_tononi_source_fidelity.md)
for the evidence contract and declared limitations.
