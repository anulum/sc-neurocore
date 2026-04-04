# Nociceptor

**Module:** `engine/src/neurons/sensory.rs`
**Reference:** Basbaum et al. 2009; Gold & Gebhart 2010
**Family:** Spiking sensory receptor, high-threshold pain neuron with sensitisation
**State variables:** `v` (membrane potential), `sensitisation` (threshold reduction)

---

## Biological Context

Nociceptors are high-threshold sensory neurons that detect noxious (tissue-damaging) stimuli. Their cell bodies reside in dorsal root ganglia (DRG) or trigeminal ganglia, with free nerve endings in skin, muscle, and viscera.

Key features:
- High activation threshold: only noxious-intensity stimuli produce firing
- TTX-resistant Na+ channels (Nav1.8, Nav1.9) produce slow, broad action potentials
- Peripheral sensitisation (hyperalgesia): repeated noxious stimulation lowers firing threshold via inflammatory mediators (PGE2, bradykinin, NGF)
- Sensitisation builds with each spike and decays very slowly (seconds to minutes)
- C-fibre nociceptors: unmyelinated, slow conduction (~1 m/s)
- A-delta nociceptors: thinly myelinated, faster conduction (~5-30 m/s)

The model implements a LIF neuron with a high threshold (-30 mV vs typical -50 mV) and a sensitisation variable that progressively lowers the effective threshold after each spike, capped at 10 mV reduction.

---

## Equations

### Membrane dynamics

$$\frac{dV}{dt} = \frac{-(V - V_{rest}) + gain \cdot S}{\tau}$$

where $S$ is noxious stimulus intensity (clamped $\geq 0$).

### Effective threshold with sensitisation

$$V_{eff} = V_{threshold} - sensitisation$$

### Spike, reset, and sensitisation buildup

$$\text{if } V \geq V_{eff}: \quad V \leftarrow V_{reset}, \quad sensitisation \leftarrow \min(sensitisation + sens\_rate, 10.0), \quad \text{emit spike (1)}$$

### Sensitisation decay (no spike)

$$\frac{d(sensitisation)}{dt} = -\frac{sensitisation}{\tau_{sens}}$$

Sensitisation is clamped to $\geq 0$.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `v_rest` | -65.0 | mV | Resting potential |
| `v_reset` | -68.0 | mV | Post-spike reset potential |
| `v_threshold` | -30.0 | mV | Spike threshold (high) |
| `tau` | 8.0 | ms | Membrane time constant |
| `sensitisation` | 0.0 | mV | Current threshold reduction |
| `tau_sens` | 5000.0 | ms | Sensitisation decay time constant |
| `sens_rate` | 0.5 | mV | Sensitisation increment per spike |
| `gain` | 1.0 | — | Stimulus-to-current gain |
| `dt` | 0.5 | ms | Integration timestep |

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/sensory.rs` |
| PyO3 wrapper | `py_neuron_default!` macro in `pyo3_neurons.rs` |
| NetworkRunner wired | `NeuronVariant::NociceptorCell` |
| `create_neuron("Nociceptor")` | Yes |
| STRONG tests | 4 (high threshold, sensitisation buildup + decay, no-fire, reset) |
| NaN/extreme input test | Via NetworkRunner `all_models_*` tests |
| Benchmark | `nociceptor_10k_steps`: **370 µs** (37 ns/step), i5-11600K |

---

## Benchmark (i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| nociceptor_10k_steps | 370 µs |
| Per step | **37 ns** |

The step function evaluates one linear ODE, one comparison, and conditional sensitisation update. No transcendental functions. Expected cost in the low nanosecond range per step.

---

## Findings

1. **High threshold (-30 mV) requires strong drive.** At gain = 1.0, stimulus intensity ~5.0 does not produce spikes (confirmed by `nociceptor_high_threshold` test). Stimulus = 50.0 drives the cell to fire.
2. **Sensitisation lowers threshold progressively.** Each spike adds 0.5 mV of sensitisation (capped at 10 mV), lowering the effective threshold from -30 mV toward -40 mV. This models peripheral hyperalgesia.
3. **Sensitisation decays very slowly (tau_sens = 5000 ms).** After 50,000 steps at zero stimulus, sensitisation measurably decays but does not fully resolve. This matches the clinical observation that hyperalgesia persists for minutes after noxious stimulation ceases.
4. **v_reset = -68 mV (close to v_rest = -65 mV).** The shallow reset reflects the slow repolarisation of TTX-resistant Na+ channel action potentials. This also limits maximum firing rate.
