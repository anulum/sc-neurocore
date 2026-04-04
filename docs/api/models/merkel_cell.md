# MerkelCell

**Module:** `engine/src/neurons/sensory.rs`
**Reference:** Lesniak et al. 2014
**Family:** Spiking sensory receptor, slowly adapting type I (SAI) mechanoreceptor
**State variables:** `v` (membrane potential), `adapt` (slow adaptation variable)

---

## Biological Context

Merkel cells are slowly adapting type I (SAI) mechanoreceptors located in the basal epidermis, concentrated in fingertips and lips. They form Merkel cell-neurite complexes with Abeta afferents.

Key features:
- Respond to sustained, static skin indentation (pressure)
- Slowly adapting: maintain firing throughout stimulus duration, with gradual rate decrease
- Encode spatial features: texture, edges, fine form, Braille dots
- Two-component response: fast onset transient followed by slow sustained discharge
- High spatial acuity due to small receptive fields (~2-3 mm)
- Firing rate roughly proportional to indentation depth

The model implements a LIF neuron with a slow adaptation variable that subtracts from the drive current, producing the characteristic SAI discharge pattern: initial high-frequency burst followed by sustained lower-frequency firing.

---

## Equations

### Membrane dynamics with adaptation

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + gain \cdot P - w}{\tau}$$

where $P$ is pressure (clamped $\geq 0$) and $w$ is the adaptation variable.

### Slow adaptation

$$\frac{dw}{dt} = \frac{a_{adapt} \cdot (V - V_{rest}) - w}{\tau_{adapt}}$$

The adaptation variable tracks membrane depolarisation with a long time constant (200 ms), gradually reducing the effective drive and lowering firing rate.

### Spike and reset

$$\text{if } V \geq V_{threshold}: \quad V \leftarrow V_{reset}, \quad \text{emit spike (1)}$$

No explicit refractory period; the reset-to-threshold gap and adaptation provide the inter-spike interval.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset potential |
| `v_threshold` | -50.0 | mV | Spike threshold |
| `tau` | 5.0 | ms | Membrane time constant |
| `adapt` | 0.0 | — | Slow adaptation variable |
| `tau_adapt` | 200.0 | ms | Adaptation time constant |
| `a_adapt` | 0.3 | — | Adaptation coupling strength |
| `gain` | 1.5 | — | Pressure-to-current gain |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory.rs` |
| PyO3 wrapper | `py_neuron_default!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | `NeuronVariant::Merkel` |
| `create_neuron("Merkel")` or `create_neuron("MerkelCell")` | Yes |
| STRONG tests | 4 (fires sustained, slow adaptation, no-fire, reset) |
| NaN/extreme input test | Via NetworkRunner `all_models_*` tests |
| Benchmark | `merkel_10k_steps`: **239 µs** (23.9 ns/step), i5-11600K |

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| merkel_10k_steps | 239 µs |
| Per step | **23.9 ns** |

The step function evaluates two linear ODEs and one comparison. No transcendental functions. Expected cost in the low nanosecond range per step.

---

## Findings

1. **Slow adaptation (tau_adapt = 200 ms) produces SAI-like discharge.** First 1000 steps produce more spikes than the next 1000 at constant pressure, but the second half still fires (confirmed by `merkel_slow_adaptation` test).
2. **No firing at zero pressure.** The gain * 0 - adapt term cannot drive V above threshold from rest.
3. **a_adapt = 0.3 balances onset vs sustained rate.** Higher values would produce faster adaptation (more RA-like); lower values would produce nearly non-adapting discharge.
4. **Reset clears adaptation variable.** After `reset()`, the cell returns to its unadapted state, ready for a new stimulus.
