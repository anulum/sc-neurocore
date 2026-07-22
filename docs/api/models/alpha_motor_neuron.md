# AlphaMotorNeuron

**Module:** `engine/src/neurons/motor/alpha_motor_neuron.rs`
**Reference:** Powers & Binder, J. Neurophysiol. 86, 2001; Heckman & Enoka, Compr. Physiol. 2(4), 2012
**Family:** Hodgkin-Huxley variant, spinal motor neuron with PIC and AHP
**State variables:** `v` (membrane potential), `h` (Na+ inactivation), `n` (K+ activation), `m_pic` (PIC activation), `ca` (intracellular Ca2+)

---

## Biological Context

Alpha motor neurons are the final common pathway from the central nervous system to skeletal muscle. Located in the ventral horn of the spinal cord, they innervate extrafusal muscle fibres to produce force. They are among the largest neurons in the nervous system (soma diameter 50-80 µm), with correspondingly lower input resistance and higher rheobase than cortical neurons.

Key electrophysiological features:
- Persistent inward current (PIC) from L-type Ca2+ channels enables plateau potentials — self-sustained firing after brief input, a mechanism for graded force control
- Ca2+-dependent afterhyperpolarisation (AHP) from SK channels limits firing rate and produces the characteristic linear f-I relationship
- Larger soma → higher membrane capacitance (C_m = 1.5 µF/cm²) → slower dynamics
- Slow and fast motor units have different AHP durations (50-150 ms)

The interplay of PIC (amplifying) and AHP (limiting) creates a bistable system that converts brief cortical commands into sustained, rate-modulated motor output.

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_{PIC} - I_{AHP} - I_L + I_{ext}$$

### Wang-Buzsaki Na+/K+ gating

$$I_{Na} = g_{Na} m_\infty^3 h (V - E_{Na}), \quad I_K = g_K n^4 (V - E_K)$$

Standard WB alpha/beta rates with kinetic scaling $\phi = 4.0$.

### Persistent inward current (L-type Ca2+)

$$m_{PIC,\infty} = \frac{1}{1 + \exp(-(V + 50)/5)}$$

$$\tau_{PIC} = 50 \text{ ms}$$

$$I_{PIC} = g_{PIC} \, m_{PIC} \, (V - E_{Ca})$$

Activates at subthreshold potentials (~-50 mV half-activation), providing depolarising current that contributes to plateau potentials. Slow dynamics (50 ms) allow integration over the inter-spike interval.

### Ca2+-dependent AHP (SK channels)

$$\frac{d[Ca^{2+}]}{dt} = -\frac{[Ca^{2+}]}{\tau_{Ca}} + I_{Ca,entry} + I_{Ca,spike}$$

$$AHP_\infty = \frac{[Ca^{2+}]}{[Ca^{2+}] + 0.5}$$

$$I_{AHP} = g_{AHP} \, AHP_\infty \, (V - E_K)$$

Ca2+ entry is proportional to PIC activation and includes a transient spike-related component. The AHP limits firing rate by activating SK channels proportionally to accumulated Ca2+.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential |
| `h` | 0.8 | — | Na+ inactivation |
| `n` | 0.1 | — | K+ activation |
| `m_pic` | 0.0 | — | PIC activation |
| `ca` | 0.0 | µM | Intracellular Ca2+ |
| `g_na` | 35.0 | mS/cm² | Na+ conductance |
| `g_k` | 9.0 | mS/cm² | Delayed-rectifier K+ |
| `g_pic` | 0.5 | mS/cm² | Persistent inward current |
| `g_ahp` | 3.0 | mS/cm² | Ca2+-dependent K+ (AHP) |
| `g_l` | 0.15 | mS/cm² | Leak (larger soma) |
| `e_na` | 55.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_ca` | 120.0 | mV | Ca2+ reversal |
| `e_l` | -65.0 | mV | Leak reversal |
| `c_m` | 1.5 | µF/cm² | Membrane capacitance (large soma) |
| `phi` | 4.0 | — | Kinetic scaling |
| `tau_ca` | 150.0 | ms | Ca2+ decay time constant |
| `dt` | 0.01 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

Sub-stepping: 50 steps per call (0.5 ms real time per call).

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/motor/alpha_motor_neuron.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` |
| NetworkRunner wired | `NeuronVariant::AlphaMotor` |
| `create_neuron("AlphaMotor")` | Yes |
| `supported_models()` | Includes "AlphaMotor" |
| coverage tests | 11 (fire, no-fire, negative, AHP rate-limit, PIC, Ca2+, reset, bounded, NaN, extreme, performance) |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `alpha_motor_1k_steps`: **34.2 ms** (34.2 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| alpha_motor_1k_steps | 34.2 ms |
| Per step | **34.2 µs** |

Slower than PV+ (4.35 ms/1k) due to higher C_m (1.5 vs 1.0), extra PIC and AHP current evaluations, and Ca2+ dynamics per sub-step. Measured 2026-04-04.

---

## Comparison with Related Models

| Property | Alpha Motor (this) | WangBuzsaki | PV+ FS | Izhikevich |
|----------|-------------------|-------------|--------|------------|
| PIC | Yes (L-type Ca2+) | No | No | No |
| AHP (Ca2+-dependent) | Yes (SK channels) | No | No | d param |
| Plateau potential | Yes | No | No | No |
| C_m | 1.5 µF/cm² | 1.0 | 1.0 | — |
| Soma size | 50-80 µm | ~20 µm | ~20 µm | — |
| f-I relationship | Linear (AHP limited) | Non-linear | Non-linear | Type-dependent |

---

## Findings

1. **AHP limits firing rate.** With g_ahp=3.0: fewer spikes than g_ahp=0 at the same input current. Verified.
2. **PIC responds to depolarisation.** m_pic increases from baseline during sustained input, contributing depolarising current.
3. **Ca2+ accumulates during spiking.** Verified: ca > 0 after 5000 steps at I=5.0.
4. **Reset restores deterministic state.** Post-reset neuron matches fresh neuron bit-exactly.
5. **NaN-safe after reset.** NaN input corrupts state, reset() restores finite values.
6. **Extreme input tolerant.** ±1e6 input does not cause panics; reset restores finite state.
