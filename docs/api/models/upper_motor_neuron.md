# UpperMotorNeuron

**Module:** `engine/src/neurons/motor.rs`
**Reference:** Pospischil et al., Biol. Cybern. 99(4-5), 2008 (RS variant); Larkum, Trends Neurosci. 36(3), 2013
**Family:** Hodgkin-Huxley variant, L5 pyramidal regular-spiking with dendritic Ca2+
**State variables:** `v` (membrane potential), `m` (Na+ activation), `h` (Na+ inactivation), `n` (K+ activation), `p` (M-current activation), `s` (high-threshold Ca2+ activation)

---

## Biological Context

Upper motor neurons are layer 5 pyramidal cells in the primary motor cortex (M1) that project through the corticospinal tract to spinal motor neuron pools. They translate cortical motor plans into descending commands that drive alpha and gamma motor neurons.

Key electrophysiological features:
- Regular-spiking (RS) pattern with spike-frequency adaptation via M-current (Kv7)
- High-threshold dendritic Ca2+ spikes (L-type Ca2+) enable coincidence detection between basal and apical dendritic input (BAC firing, Larkum 2013)
- Standard cortical pyramidal morphology (apical dendrite reaching L1)
- Moderate firing rates (10-40 Hz sustained)

The M-current provides the characteristic RS adaptation: initial high-frequency firing that decays to a lower sustained rate. The dendritic Ca2+ current models the ability of L5 pyramidals to generate Ca2+ spikes in the apical dendrite, which amplify distal input.

---

## Equations

### Membrane potential

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_M - I_{Ca} - I_L + I_{ext}$$

### Pospischil gating ($V_T = -56.2$ mV)

Standard alpha/beta rates for Na+ (m, h) and K+ (n) — see SSTNeuron docs for full expressions.

### M-current (adaptation)

$$p_\infty = \frac{1}{1 + e^{-(V+35)/10}}, \quad \tau_p = \frac{400}{3.3 e^{(V+35)/20} + e^{-(V+35)/20}}$$

### High-threshold Ca2+ (dendritic)

$$s_\infty = \frac{1}{1 + e^{-(V+20)/5}}, \quad \tau_s = 10 \text{ ms}$$

$$I_{Ca} = g_{Ca} \, s^2 \, (V - E_{Ca})$$

Activates at depolarised potentials (half-activation -20 mV), modelling high-voltage-activated L-type Ca2+ channels in the apical dendrite.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -70.0 | mV | Membrane potential |
| `m` | 0.05 | — | Na+ activation |
| `h` | 0.6 | — | Na+ inactivation |
| `n` | 0.3 | — | K+ activation |
| `p` | 0.0 | — | M-current activation |
| `s` | 0.0 | — | Ca2+ activation |
| `g_na` | 50.0 | mS/cm² | Na+ conductance |
| `g_k` | 5.0 | mS/cm² | Delayed-rectifier K+ |
| `g_m` | 0.07 | mS/cm² | M-current (Pospischil RS) |
| `g_ca` | 0.3 | mS/cm² | High-threshold Ca2+ |
| `g_l` | 0.1 | mS/cm² | Leak |
| `e_na` | 50.0 | mV | Na+ reversal |
| `e_k` | -90.0 | mV | K+ reversal |
| `e_ca` | 120.0 | mV | Ca2+ reversal |
| `e_l` | -70.0 | mV | Leak reversal |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `dt` | 0.025 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |

Sub-stepping: 4 steps per call (0.1 ms real time per call).

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/motor.rs` |
| PyO3 wrapper | `pyo3_neurons.rs` via `py_neuron_default!` |
| NetworkRunner wired | `NeuronVariant::UpperMotor` |
| `create_neuron("UpperMotor")` | Yes |
| `supported_models()` | Includes "UpperMotor" |
| STRONG tests | 10 (fire, no-fire, negative, adaptation, Ca2+, reset, bounded, NaN, extreme, performance) |
| Pipeline integration | Covered by `create_neuron_all_supported` |
| Benchmark | `upper_motor_1k_steps`: **3.24 ms** (3.24 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| upper_motor_1k_steps | 3.24 ms |
| Per step | **3.24 µs** |

Pospischil-style gating with 4 sub-steps + Ca2+ current. Measured 2026-04-04.

---

## Findings

1. **M-current drives adaptation.** Second 5000-step epoch fires fewer spikes than first at constant input. Verified.
2. **Ca2+ gate activates during spiking.** s > baseline after sustained input. Verified.
3. **No spontaneous firing.** Zero input produces zero spikes. Verified.
4. **Reset deterministic.** Post-reset matches fresh neuron. Verified.
5. **NaN-safe after reset.** Verified.
