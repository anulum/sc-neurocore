# AlphaNeuron

**Class:** `sc_neurocore.neurons.models.alpha.AlphaNeuron`
**Module:** `sc_neurocore/neurons/models/alpha.py`
**Identity:** dual excitatory/inhibitory alpha-synapse leaky integrate-and-fire
**Sources:** Rall (1967), *Distinguishing theoretical synaptic potentials*, J. Neurophysiol. 30(5), 1138–1168 (the alpha kernel); Gerstner & Kistler (2002), *Spiking Neuron Models*, Cambridge University Press, §4.1, [DOI 10.1017/CBO9780511815706](https://doi.org/10.1017/CBO9780511815706)

---

## What this model is (and is not)

The model is a **dual excitatory/inhibitory current-based alpha-synapse
LIF**: a leaky membrane driven by two synaptic currents, each carried by a
two-state alpha cascade reproducing Rall's alpha kernel
`alpha(t) ~ (t/tau) * exp(1 - t/tau)`. The maintained numerical step is the
exact piecewise-constant-input flow: each filter relaxes exactly and the
membrane update integrates the alpha currents with the exact convolution
(including the equal-time-constant limit). That exact flow is the
engineering contract, not a biological publication claim.

An earlier public-docs line citing Rall 1962 (Ann. N.Y. Acad. Sci. 96) was
a misattribution for this artefact: the alpha kernel belongs to Rall 1967.
Defaults (`tau_v=20`, `tau_exc=5`, `tau_inh=10`, dimensionless scale) are
catalogue/model-family choices, not source-derived parameters.

## Equations and the exact maintained step

Membrane (leaky integrate-and-fire, exact relaxation plus exact alpha
convolution):

$$\tau_v \frac{dV}{dt} = -(V - V_{rest}) + i_{exc} - i_{inh}$$

Synaptic cascade per channel (exact filter relaxation):

$$\frac{da}{dt} = -\frac{a - \tau I}{\tau}, \qquad
\frac{di}{dt} = -\frac{i - \tau I}{\tau} + \frac{a - \tau I}{\tau^2}\,dt \;\Rightarrow\;
i(t) \sim \frac{t}{\tau} e^{1 - t/\tau}$$

Spike event (candidate crossing):

$$V_{t+dt} \ge V_{threshold}:\quad V \leftarrow V_{rest}$$

Only the membrane potential resets; the synaptic cascade states evolve
continuously across spikes.

## Parameters and state

| Name | Default | Role |
|---|---|---|
| `v` | 0.0 | membrane potential (state; only state reset at a spike) |
| `a_exc` / `a_inh` | 0.0 | alpha-rise cascade states |
| `i_exc` / `i_inh` | 0.0 | excitatory / inhibitory synaptic currents |
| `v_rest` | 0.0 | leak reversal potential; also the somatic reset |
| `v_threshold` | 1.0 | spike threshold; must exceed `v_rest` |
| `tau_v` | 20.0 | membrane time constant |
| `tau_exc` / `tau_inh` | 5.0 / 10.0 | alpha time constants |
| `dt` | 1.0 | piecewise-constant-input sampling interval |

## Scalar and batch use

```python
from sc_neurocore.neurons.models.alpha import AlphaNeuron

neuron = AlphaNeuron()
spike = neuron.step(2.0, 0.5)                      # one exact-flow interval

result = neuron.simulate([2.0] * 500, 0.5)         # batch on the fastest lane
print(result["spike_count"], result["v_final"])
```

Every `simulate` batch returns the complete `v`, `a_exc`, `i_exc`,
`a_inh`, `i_inh`, and `spikes` trajectories plus the five final-state
receipts and `spike_count`, and leaves the instance at the final state.
Invalid input, configuration, or a non-finite candidate never mutates
state.

## Executable runtimes

| Lane | Surface | Parity to the Python golden |
|---|---|---|
| Python | `AlphaNeuron.step` / `simulate(backend="python")` | exact (reference) |
| Rust engine | `sc_neurocore_engine.py_alpha_simulate` | `1e-12` |
| Rust safety | `accel/rust/safety/alpha.rs` (standalone `rustc --test`) | `2e-15` |
| Julia | `accel/julia/neurons/alpha.jl` via juliacall | `1e-12` |
| Go | `accel/go/alpha/libalpha.so` (C ABI) | `1e-12` |
| Mojo | `accel/mojo/alpha/libalpha.so` (C ABI) | `1e-10` |

## Reproducibility and benchmark evidence

The descriptor pins a 256-step sampled dual-drive batch (non-default state)
with golden trace SHA-256 `de6081c3…ae976eca`. The committed five-runtime
benchmark (`benchmarks/results/bench_alpha.json`) records 200,000-step runs
on one pinned logical CPU with all five lanes returning matching traces,
final states, and spike counts within the declared tolerances; it is local
regression evidence only, not a speed claim.

## Python-to-Verilog and the formal boundary

Generated Q32.32 RTL tracks the exact flow at the enrolled grid-exact
operating point (`tau_v=8`, `tau_exc=4`, `tau_inh=2`: all exponential
arguments on the 0.125-step lookup grid, all rates distinct). The measured
maximum state error over a 256-step sign-changing drive is `1.95e-8`
(declared envelope `0.01`), with the complete 24-event vector identical at
two enrolled inhibitory levels. A depth-4 Z3 bounded job proves reset
safety only. No formal equivalence, synthesis timing, device, or PPA claim
is made; the silicon tier is H1.

## Scope boundary

- The equal-time-constant convolution limit is implemented in every
  production lane; the schemas encode the general branch and document the
  boundary.
- The inhibitory drive is an overridable schema parameter so the generated
  single-input RTL remains well-formed; the production model accepts a
  full per-step inhibitory vector.
- Defaults are catalogue/model-family choices, not source parameters.

See the [source-fidelity page](../../validation/alpha_source_fidelity.md)
for the complete primary-source analysis and evidence index.

## References

Rall, W. (1967). Distinguishing theoretical synaptic potentials computed
for different soma-dendritic distributions of synaptic input. *Journal of
Neurophysiology* 30(5), 1138–1168.

Gerstner, W. & Kistler, W.M. (2002). *Spiking Neuron Models*. Cambridge
University Press, §4.1. <https://doi.org/10.1017/CBO9780511815706>
