# OlfactoryReceptorNeuron

**Module:** `engine/src/neurons/sensory.rs`
**Reference:** Rospars et al. 2008; Firestein 2001
**Family:** Spiking sensory receptor, olfactory cAMP cascade with Ca2+/CaM adaptation
**State variables:** `v` (membrane potential), `camp` (normalised cAMP), `adapt` (Ca2+/CaM adaptation)

---

## Biological Context

Olfactory receptor neurons (ORNs) are bipolar neurons in the olfactory epithelium. Each ORN expresses one odorant receptor gene (from ~400 in humans). Their axons project to a single glomerulus in the olfactory bulb.

Key features:
- Odorant binding to GPCR activates Golf -> adenylyl cyclase III -> cAMP production
- cAMP opens CNG channels (CNGA2/B1b) -> depolarisation -> spike generation
- Ca2+ entry through CNG channels activates Ca2+/CaM feedback, which reduces CNG channel sensitivity (adaptation)
- Adaptation allows concentration-invariant odour discrimination (Weber-Fechner law)
- cAMP production follows a Hill function of odorant concentration
- Spiking output to olfactory bulb

The model implements the cAMP cascade with Hill-function odorant binding, Ca2+/CaM adaptation feedback on cAMP, and a LIF spike generator.

---

## Equations

### cAMP production (Hill function with adaptation)

$$cAMP_{target} = \frac{C}{C + 1} \cdot (1 - 0.8 \cdot adapt)$$

where $C$ is odorant concentration (clamped $\geq 0$). The adaptation variable reduces cAMP production by up to 80%.

### cAMP dynamics

$$\frac{d[cAMP]}{dt} = \frac{cAMP_{target} - [cAMP]}{\tau_{cAMP}}$$

cAMP is clamped to $[0, 1]$.

### Membrane dynamics

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + gain \cdot [cAMP] \cdot 50}{\tau}$$

The factor of 50 scales normalised cAMP to millivolt-range drive current.

### Ca2+/CaM adaptation

$$Ca_{proxy} = \begin{cases} \frac{V - V_{rest}}{20} & \text{if } V > V_{rest} \\ 0 & \text{otherwise} \end{cases}$$

$$\frac{d(adapt)}{dt} = \frac{Ca_{proxy} - adapt}{\tau_{adapt}}$$

Adaptation is clamped to $[0, 1]$.

### Spike and reset

$$\text{if } V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad \text{emit spike (1)}$$

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset potential |
| `v_threshold` | -45.0 | mV | Spike threshold |
| `tau` | 5.0 | ms | Membrane time constant |
| `camp` | 0.0 | — | Normalised cAMP concentration [0, 1] |
| `adapt` | 0.0 | — | Ca2+/CaM adaptation level [0, 1] |
| `tau_camp` | 50.0 | ms | cAMP dynamics time constant |
| `tau_adapt` | 500.0 | ms | Adaptation time constant |
| `gain` | 1.5 | — | cAMP-to-current gain |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory.rs` |
| PyO3 wrapper | `py_neuron_default!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | `NeuronVariant::OlfactoryReceptor` |
| `create_neuron("OlfactoryReceptor")` or `create_neuron("OlfactoryReceptorNeuron")` | Yes |
| STRONG tests | 4 (fires with odorant, adapts, no-fire, reset) |
| NaN/extreme input test | Via NetworkRunner `all_models_*` tests |
| Benchmark | `olfactory_10k_steps`: **1.48 ms** (148 ns/step), i5-11600K |

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| olfactory_10k_steps | 1.48 ms |
| Per step | **148 ns** |

The step function evaluates one division (Hill function), three linear ODEs, and two `clamp` calls. No `exp()`. Expected cost in the low nanosecond range per step.

---

## Findings

1. **Ca2+/CaM adaptation reduces firing over time.** The `olfactory_adapts` test confirms that the second 2000-step block at constant odorant produces no more spikes than the first, reflecting adaptation.
2. **Hill function C/(C+1) saturates at high concentration.** cAMP_target approaches 1.0 (before adaptation) regardless of concentration above ~10. This matches the saturating dose-response curves of ORNs.
3. **Adaptation can suppress cAMP by up to 80%.** At full adaptation (adapt = 1.0), cAMP_target is reduced to 20% of its unadapted value. This is the mechanism for concentration-invariant odour identity coding.
4. **tau_adapt = 500 ms provides slow adaptation.** This matches the seconds-scale adaptation observed in olfactory psychophysics and electrophysiology.
5. **Reset clears both cAMP and adaptation.** After `reset()`, the neuron behaves as if encountering the odorant for the first time.
