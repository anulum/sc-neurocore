# AlphaNeuron

**Module:** `sc_neurocore.neurons.models.alpha`
**Reference:** Rall, Ann. N.Y. Acad. Sci. 96, 1962; Gerstner & Kistler, *Spiking Neuron Models*, §4.1
**Family:** Integrate-and-fire neuron with current-based alpha synapses
**State variables:** `v` (membrane potential), `a_exc`/`i_exc` (excitatory alpha rise/current states), `a_inh`/`i_inh` (inhibitory alpha rise/current states)

---

## Equations

For constant input over one timestep, the maintained implementation advances the full linear alpha-cascade system exactly before applying the spike reset.

### Excitatory alpha synapse

$$\frac{dA_{exc}}{dt} = -\frac{A_{exc}}{\tau_{exc}} + I_{exc,input}$$

$$\frac{dI_{exc}}{dt} = \frac{A_{exc} - I_{exc}}{\tau_{exc}}$$

### Inhibitory alpha synapse

$$\frac{dA_{inh}}{dt} = -\frac{A_{inh}}{\tau_{inh}} + I_{inh,input}$$

$$\frac{dI_{inh}}{dt} = \frac{A_{inh} - I_{inh}}{\tau_{inh}}$$

### Membrane potential

$$\tau_v \frac{dV}{dt} = -(V - V_{rest}) + I_{exc} - I_{inh}$$

### Spike and reset

$$V(t + \Delta t) \geq V_{threshold}: \quad V \leftarrow V_{rest}, \quad \text{return } 1$$

The synaptic alpha states are not reset by a somatic spike; they continue to carry the causal input history.

---

## Alpha-function interpretation

The two-state synaptic cascade is the Rall/Gerstner alpha kernel. A unit impulse injected into `A` with zero initial current gives

$$I(t) = \frac{t}{\tau} e^{-t/\tau}$$

up to the configured input scaling. Constant drive relaxes both `A` and `I` to the same steady state `tau * input`, while the current state rises with the alpha delay rather than jumping as a single-pole filter.

---

## Exact timestep flow

For each synapse, with `S = tau * input`, `D_A = A_0 - S`, `D_I = I_0 - S`, and `q = exp(-dt / tau)`:

$$A_{next} = S + D_A q$$

$$I_{next} = S + q(D_I + D_A\Delta t / \tau)$$

The membrane uses the corresponding closed-form convolution of the two alpha currents with the membrane filter. Candidate `A`, `I`, and `V` values are computed first and committed only if all are finite.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | 0.0 | a.u. | Membrane potential |
| `a_exc` | 0.0 | a.u. | Excitatory alpha rise state |
| `i_exc` | 0.0 | a.u. | Excitatory synaptic current |
| `a_inh` | 0.0 | a.u. | Inhibitory alpha rise state |
| `i_inh` | 0.0 | a.u. | Inhibitory synaptic current |
| `v_rest` | 0.0 | a.u. | Resting potential |
| `v_threshold` | 1.0 | a.u. | Spike threshold |
| `tau_v` | 20.0 | ms | Membrane time constant |
| `tau_exc` | 5.0 | ms | Excitatory synaptic time constant |
| `tau_inh` | 10.0 | ms | Inhibitory synaptic time constant |
| `dt` | 1.0 | ms | Integration timestep |

All state variables, voltage parameters, currents, and candidate updates must be finite. `tau_v`, `tau_exc`, `tau_inh`, and `dt` must be finite and positive at construction and before every step.

---

## Behaviour

### Dual-input firing

`step(exc_current, inh_current=0.0)` accepts separate excitatory and inhibitory drives.

With excitatory drive and no inhibition, `a_exc` rises first, `i_exc` follows with the alpha delay, and the membrane integrates the filtered current toward threshold. With inhibitory drive, `a_inh` and `i_inh` subtract from the membrane drive and can suppress threshold crossings.

### Time constant hierarchy

The default hierarchy is

$$\tau_{exc} = 5\text{ ms} < \tau_{inh} = 10\text{ ms} < \tau_v = 20\text{ ms}$$

Fast excitation, slower inhibition, and slow membrane integration create a finite temporal integration window. The exact flow remains bounded for large `dt` whenever the configured linear system has finite positive time constants.

### Reset semantics

A spike resets only `v` to `v_rest`. The alpha synapse states preserve their continuous trajectory, matching a current-based synaptic input history rather than a voltage-reset artefact.

---

## Pipeline compatibility

The standard SC-NeuroCore pipeline calls `step(current)` with one float, so the inhibitory input defaults to `0.0` during ordinary `Population` and `Network` execution. Standalone users can call `step(exc_current, inh_current)` to exercise both channels.

Python, Go, Julia, and Rust safety mirrors share the same exact alpha-cascade contract. Go and Rust return errors on invalid runtime state; Python and Julia raise before mutation.

---

## Numerical contract

- **Exact linear flow:** no raw timestep increment is used for the alpha or membrane states.
- **Candidate-first update:** all candidate states are computed and checked before mutation.
- **Equal time constants:** the membrane convolution uses the analytic limit when `tau_v` equals a synaptic time constant.
- **Finite-domain boundary:** non-finite currents, corrupted state, non-positive time constants, and non-finite candidates fail before state mutation.

---

## Performance

Fresh local benchmark evidence:

| Runtime | Command | Median |
|---------|---------|--------|
| Python | `PYTHONPATH=src .venv/bin/python benchmarks/bench_model_alpha.py` | 3483.5833 ns/step |

Benchmark artefact: `benchmarks/results/local_python_2026-06-01_alpha.json`.

The benchmark runs 5 repeats of 50,000 candidate-first exact alpha-cascade steps with `exc_current=2.0`, `inh_current=0.35`, and `v_threshold=100.0` to measure the continuous subthreshold flow without reset events.
