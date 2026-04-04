# RenshawCell

**Module:** `engine/src/neurons/motor.rs`
**Reference:** Renshaw, J. Neurophysiol. 4, 1941 (discovery); Windhorst, Prog. Neurobiol. 46(5), 1996
**Family:** Wang-Buzsaki HH variant with adaptation, spinal inhibitory interneuron
**State variables:** `v` (membrane potential), `h` (Na+ inactivation), `n` (K+ activation), `adapt` (adaptation current)

---

## Biological Context

Renshaw cells are glycinergic inhibitory interneurons in the ventral horn of the spinal cord. They receive cholinergic excitation from motor neuron axon collaterals and provide recurrent (feedback) inhibition back to the same and synergist motor neuron pools. This recurrent inhibition circuit helps stabilise motor output, prevent oscillation, and sharpen the contrast between agonist and antagonist muscle activation.

Key electrophysiological features:
- High-frequency initial burst in response to motor neuron collateral input (nicotinic cholinergic drive)
- Rapid adaptation: firing rate decays from initial burst within ~50 ms
- Glycinergic output (inhibitory)
- Small soma, moderate input resistance
- Functionally analogous to cortical FS interneurons but with stronger adaptation

The burst-then-decay pattern is modelled by a slow adaptation conductance (g_adapt = 5.0 mS/cm²) that activates with depolarisation and pulls the membrane toward E_K, progressively reducing excitability.

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_{adapt} - I_L + I_{ext}$$

### Wang-Buzsaki Na+/K+ gating

Standard WB alpha/beta rates with $\phi = 5.0$.

### Adaptation current

$$adapt_\infty = \frac{1}{1 + \exp(-(V + 30)/5)}$$

$$\frac{d(adapt)}{dt} = \frac{adapt_\infty - adapt}{\tau_{adapt}}$$

$$I_{adapt} = g_{adapt} \cdot adapt \cdot (V - E_K)$$

$\tau_{adapt} = 50$ ms provides the ~50 ms burst-to-adapted transition.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.8 | — | Na+ inactivation |
| `n` | 0.1 | — | K+ activation |
| `adapt` | 0.0 | — | Adaptation variable |
| `g_na` | 35.0 | mS/cm² | Na+ conductance |
| `g_k` | 9.0 | mS/cm² | Delayed-rectifier K+ |
| `g_adapt` | 5.0 | mS/cm² | Adaptation conductance |
| `g_l` | 0.12 | mS/cm² | Leak |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `phi` | 5.0 | — | Kinetic scaling |
| `tau_adapt` | 50.0 | ms | Adaptation time constant |
| `dt` | 0.01 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

Sub-stepping: 50 per call (0.5 ms real time per call).

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/motor.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` |
| NetworkRunner wired | `NeuronVariant::Renshaw` |
| `create_neuron("Renshaw")` | Yes |
| `supported_models()` | Includes "Renshaw" |
| STRONG tests | 10 (fire, no-fire, negative, burst-adapt, adapt variable, reset, bounded, NaN, extreme, performance) |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `renshaw_1k_steps`: **2.78 ms** (2.78 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| renshaw_1k_steps | 2.78 ms |
| Per step | **2.78 µs** |

WB gating with 50 sub-steps + adaptation. Measured 2026-04-04.

---

## Findings

1. **Burst-then-adapt confirmed.** First 2000-step epoch fires more than second at constant I=4.0. Verified.
2. **Adaptation variable increases.** adapt > baseline after sustained firing. Verified.
3. **No spontaneous firing.** Zero input produces zero spikes. Verified.
4. **Reset deterministic.** Post-reset matches fresh. Verified.
5. **NaN-safe after reset.** Verified.
