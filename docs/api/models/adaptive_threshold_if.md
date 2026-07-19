# AdaptiveThresholdIFNeuron

**Class:** `sc_neurocore.neurons.models.adaptive_threshold_if.AdaptiveThresholdIFNeuron`
**Module:** `sc_neurocore/neurons/models/adaptive_threshold_if.py`
**Identity:** composite reduced adaptive-threshold leaky integrate-and-fire
**Sources:** Mihalas & Niebur (2009), Neural Computation 21(3), 704–718, [DOI 10.1162/neco.2008.12-07-680](https://doi.org/10.1162/neco.2008.12-07-680); Platkiewicz & Brette (2010), PLoS Computational Biology 6(7), e1000850, [DOI 10.1371/journal.pcbi.1000850](https://doi.org/10.1371/journal.pcbi.1000850)

---

## What this model is (and is not)

The model is an explicitly **composite reduced adaptive-threshold LIF**:

1. an exact constant-input leaky-integrate-and-fire membrane relaxation,
2. the Mihalas–Niebur (2009) threshold equation taken at zero voltage
   coupling (`a = 0`), and
3. the fixed post-spike threshold shift derived in Platkiewicz & Brette
   (2010).

It is **not** an exact Platkiewicz–Brette model: the source derives a
voltage-dependent threshold equilibrium `theta_inf(V)` through sodium
inactivation, which this reduced model does not include. It is **not** an
exact Mihalas–Niebur model either: the voltage-coupling term `a(V - E_L)`
and the adaptation currents are outside this model. Defaults are
catalogue/model-family choices, not source-derived parameters.

## Equations and the exact maintained step

Membrane (leaky integrate-and-fire, exact constant-input relaxation over
one `dt`):

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + I
\quad\Rightarrow\quad
V_{t+dt} = (V_{rest} + I) + (V_t - (V_{rest} + I))\,e^{-dt/\tau_m}$$

Threshold (Mihalas–Niebur equation at `a = 0`, exact relaxation):

$$\frac{d\theta}{dt} = -b(\theta - \theta_\infty),\; b = 1/\tau_\theta
\quad\Rightarrow\quad
\theta_{t+dt} = \theta_{rest} + (\theta_t - \theta_{rest})\,e^{-dt/\tau_\theta}$$

Spike event (candidate crossing) and the fixed Platkiewicz–Brette shift:

$$V_{t+dt} \ge \theta_{t+dt}:\quad
V \leftarrow V_{reset},\;\; \theta \leftarrow \theta_{t+dt} + \Delta\theta$$

The crossing is evaluated on the post-update candidates at maintained step
boundaries; no within-step root localisation is claimed.

## Parameters and state

| Name | Default | Unit | Role |
|---|---|---|---|
| `v` | -65.0 | mV | membrane potential (state) |
| `theta` | -50.0 | mV | adaptive threshold (state) |
| `v_rest` | -65.0 | mV | leak reversal potential |
| `v_reset` | -65.0 | mV | post-spike membrane reset |
| `theta_rest` | -50.0 | mV | baseline threshold; must exceed `v_rest` and `v_reset` |
| `delta_theta` | 5.0 | mV | fixed non-negative post-spike threshold shift |
| `tau_m` | 10.0 | ms | membrane time constant |
| `tau_theta` | 50.0 | ms | threshold relaxation time constant |
| `dt` | 0.1 | ms | piecewise-constant-input sampling interval |

## Scalar and batch use

```python
from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron

neuron = AdaptiveThresholdIFNeuron()
spike = neuron.step(12.0)              # one exact-relaxation interval

result = neuron.simulate([20.0] * 500) # complete batch on the fastest lane
print(result["spike_count"], result["v_final"], result["theta_final"])
```

Every `simulate` batch returns the complete `v`, `theta`, and `spikes`
trajectories plus `v_final`, `theta_final`, and `spike_count` receipts, and
leaves the instance at the final state. Invalid input, configuration, or a
non-finite candidate never mutates state.

## Executable runtimes

| Lane | Surface | Parity to the Python golden |
|---|---|---|
| Python | `AdaptiveThresholdIFNeuron.step` / `simulate(backend="python")` | exact (reference) |
| Rust engine | `sc_neurocore_engine.py_adaptive_threshold_if_simulate` | `1e-12` |
| Rust safety | `accel/rust/safety/adaptive_threshold_if.rs` (standalone `rustc --test`) | `2e-15` |
| Julia | `accel/julia/neurons/adaptive_threshold_if.jl` via juliacall | `1e-12` |
| Go | `accel/go/adaptive_threshold_if/libadaptive_threshold_if.so` (C ABI) | `1e-12` |
| Mojo | `accel/mojo/adaptive_threshold_if/libadaptive_threshold_if.so` (C ABI) | `1e-10` |

## Reproducibility and benchmark evidence

The descriptor pins a 256-step sampled batch (non-default state, sinusoidal
drive `22.0 + 6.0*sin(i*0.037)`) with golden trace SHA-256
`014ba91d…9feb3fb5`. The committed five-runtime benchmark
(`benchmarks/results/bench_adaptive_threshold_if.json`) records 200,000-step
runs on one pinned logical CPU with all five lanes returning matching
traces, final states, and spike counts within the declared tolerances; it is
local regression evidence only, not a speed claim.

## Python-to-Verilog and the formal boundary

Generated Q32.32 RTL tracks the exact relaxations at the enrolled
grid-exact operating point (`-dt/tau == -0.125` on the 0.125-step lookup
grid): the measured maximum state error over a 256-step sign-changing drive
is `1.22e-8` (declared envelope `0.01`), with the complete event vector
identical, every RTL spike resetting `v` to `v_reset` and shifting `theta`
by exactly `delta_theta`. A depth-4 Z3 bounded job proves reset safety only.
No formal equivalence, synthesis timing, device, or PPA claim is made; the
silicon tier is H1.

## Scope boundary

- No voltage-dependent threshold equilibrium `theta_inf(V)` (the central
  Platkiewicz–Brette mechanism) and no Mihalas–Niebur voltage coupling or
  adaptation current.
- Defaults are catalogue/model-family choices, not source parameters.
- Continuous analogue waveform generation, refractory periods, and synaptic
  conductances are outside this reduced point model.

See the [source-fidelity page](../../validation/adaptive_threshold_if_source_fidelity.md)
for the complete primary-source analysis and evidence index.

## References

Mihalas, S. and Niebur, E. (2009). A generalized linear integrate-and-fire
neural model produces diverse spiking behaviors. *Neural Computation* 21(3),
704–718. <https://doi.org/10.1162/neco.2008.12-07-680>

Platkiewicz, J. and Brette, R. (2010). A threshold equation for action
potential initiation. *PLoS Computational Biology* 6(7), e1000850.
<https://doi.org/10.1371/journal.pcbi.1000850>
