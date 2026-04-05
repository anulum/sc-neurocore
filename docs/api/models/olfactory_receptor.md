# OlfactoryReceptorNeuron

**Module:** `engine/src/neurons/sensory.rs`
**Reference:** Rospars et al. 2008; Firestein 2001
**Family:** Spiking sensory receptor, olfactory cAMP cascade with Ca²⁺/CaM + PDE4 dual adaptation
**State variables:** `v`, `camp` (cAMP), `adapt` (Ca²⁺/CaM), `pde4` (PDE4 activity)

---

## Biological Context

Olfactory receptor neurons (ORNs) are bipolar neurons in the olfactory epithelium. Each ORN expresses one odorant receptor gene. Their axons project to a single glomerulus in the olfactory bulb.

Key features:
- Odorant binding → Golf → adenylyl cyclase III → cAMP → CNG channels → depolarisation
- **Dual adaptation pathways**:
  1. **Ca²⁺/CaM** (fast, ~500 ms): Ca²⁺ through CNG → CaM → reduces CNG sensitivity
  2. **PDE4** (slow, ~300 ms): cAMP → PKA → PDE4 upregulation → cAMP degradation (negative feedback)
- PDE4 creates a delayed negative feedback loop: sustained odorant → cAMP builds → PKA activates PDE4 → PDE4 degrades cAMP → firing declines
- Spiking output to olfactory bulb

---

## Equations

### cAMP production (Hill + CaM adaptation)

$$cAMP_{prod} = \frac{C}{C + 1} \cdot (1 - 0.8 \cdot adapt)$$

### PDE4 degradation

$$PDE4_{deg} = k_{PDE4} \cdot [PDE4] \cdot [cAMP]$$

### cAMP dynamics

$$\frac{d[cAMP]}{dt} = \frac{\max(cAMP_{prod} - PDE4_{deg}, 0) - [cAMP]}{\tau_{cAMP}}$$

### PDE4 activation (tracks cAMP with delay)

$$\frac{d[PDE4]}{dt} = \frac{[cAMP] - [PDE4]}{\tau_{PDE4}}$$

### Membrane + Ca²⁺/CaM adaptation

Same as before: LIF membrane, Ca²⁺ proxy from depolarisation, slow CaM adaptation.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset |
| `v_threshold` | -45.0 | mV | Spike threshold |
| `tau` | 5.0 | ms | Membrane time constant |
| `camp` | 0.0 | --- | Normalised cAMP [0, 1] |
| `adapt` | 0.0 | --- | Ca²⁺/CaM adaptation [0, 1] |
| `pde4` | 0.0 | --- | PDE4 activity [0, 1] |
| `tau_camp` | 50.0 | ms | cAMP dynamics |
| `tau_adapt` | 500.0 | ms | CaM adaptation |
| `tau_pde4` | 300.0 | ms | PDE4 activation |
| `k_pde4` | 1.5 | --- | PDE4 degradation rate |
| `gain` | 1.5 | --- | cAMP-to-current gain |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory.rs` |
| PyO3 wrapper | `py_neuron_default!` (state: v, camp, adapt, pde4) |
| NetworkRunner wired | `NeuronVariant::OlfactoryReceptor` |
| `create_neuron("OlfactoryReceptorNeuron")` | Yes |
| `supported_models()` | Includes "OlfactoryReceptorNeuron" |
| STRONG tests | 7 (fires, adapts, no-fire, reset, PDE4 activates, PDE4 reduces cAMP, PDE4 enhances adaptation) |
| Benchmark | `olfactory_10k_steps`: **1.48 ms** (148 ns/step), i5-11600K |

---

## Findings

1. **Dual adaptation.** Ca²⁺/CaM (fast) + PDE4 (slow) provide two distinct adaptation timescales.
2. **PDE4 activates with sustained odorant.** PDE4 rises from 0 during prolonged exposure. Verified.
3. **PDE4 reduces steady-state cAMP.** With PDE4, cAMP is lower than without. Verified.
4. **PDE4 enhances adaptation.** Late firing rate is lower with PDE4 than CaM alone. Verified.
5. **Reset clears all state.** cAMP, adapt, and pde4 all return to 0. Verified.
