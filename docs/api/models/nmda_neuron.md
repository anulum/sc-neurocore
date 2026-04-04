# NMDANeuron

**Module:** `engine/src/neurons/channels.rs`
**Reference:** Jahr & Stevens, J Neurosci 10:1830, 1990; Wang, Neuron 22:409, 1999
**Family:** WB Na+/K+ base + NMDA receptor-gated channel with Mg2+ voltage block
**State variables:** `v`, `h` (Na+ inactivation), `n` (Kdr), `s_nmda` (NMDA synaptic variable)

---

## Biological Context

NMDA (N-methyl-D-aspartate) receptors are ionotropic glutamate receptors that require two simultaneous conditions for activation: glutamate binding (presynaptic) AND membrane depolarisation (postsynaptic, to relieve Mg2+ block). This dual requirement makes NMDA receptors molecular coincidence detectors.

Key features:
- **Mg2+ voltage block**: at resting potential (-65 mV), Mg2+ ions block the channel pore; depolarisation expels Mg2+ and allows current flow
- **Ca2+ permeability**: NMDA channels conduct Ca2+ in addition to Na+/K+, triggering intracellular signalling cascades
- **Slow kinetics**: rise ~10 ms, decay ~100 ms (much slower than AMPA)
- **Synaptic plasticity**: Ca2+ influx through NMDA triggers LTP and LTD
- **Working memory**: NMDA-mediated recurrent excitation in prefrontal cortex

---

## Equations

### Mg2+ block (Jahr & Stevens 1990)

$$B(V) = \frac{1}{1 + \frac{[Mg^{2+}]}{3.57} \exp(-0.062 \cdot V)}$$

### NMDA current

$$I_{NMDA} = g_{NMDA} \cdot s_{NMDA} \cdot B(V) \cdot (V - E_{NMDA})$$

### Synaptic variable

$$\frac{ds}{dt} = \frac{s_{drive} - s}{\tau}$$

where $\tau = \tau_{rise}$ if $s_{drive} > s$, else $\tau = \tau_{decay}$; $s_{drive} = I/(I+5)$.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` (state: v, h, n, s_nmda) |
| NetworkRunner wired | `NeuronVariant::NMDA` |
| `create_neuron("NMDA")` | Yes |
| `supported_models()` | Includes "NMDA" |
| STRONG tests | 12 (fire, silent, Mg block, Mg relief, s builds, s decays, zero Mg, negative, NaN, extreme, reset, performance) |
| Benchmark | `nmda_1k_steps`: **3.29 ms** (3.29 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| nmda_1k_steps | 3.29 ms |
| Per step | **3.29 µs** |

WB gating with 50 sub-steps + NMDA + Mg2+ block. Measured 2026-04-04.

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I=3. Verified.
2. **Silent without input.** No spontaneous firing. Verified.
3. **Mg2+ block at rest.** B < 0.1 at -65 mV. Verified.
4. **Mg2+ relief at depolarised.** B > 0.4 at -20 mV. Verified.
5. **s_nmda builds with input.** Slow rise dynamics. Verified.
6. **s_nmda decays after input removal.** Slow decay (~100 ms). Verified.
7. **Zero Mg2+ increases firing.** Removing block amplifies NMDA current. Verified.
