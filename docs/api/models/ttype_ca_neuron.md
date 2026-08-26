# TTypeCaNeuron

**Module:** `engine/src/neurons/channels/t_type_ca.rs`
**Rust struct:** `TTypeCaNeuron`
**Reference:** Huguenard, Annu Rev Physiol 58:329, 1996; Destexhe, Bal, McCormick & Sejnowski, J Neurophysiol 76:2049, 1996

The model is a repository composite: the threshold-reset event and the
spike-triggered inactivation collapse (`s *= 0.3`) are
repository-specific specialisations, not publication-exact claims.
**Family:** Wang–Buzsáki Na⁺/K⁺ base + T-type Ca²⁺ (IT, low-voltage-activated)
**State variables:** `v` (membrane potential), `h` (Na⁺ inactivation), `n` (Kdr activation), `s` (T-type inactivation)

---

## Biological Context

T-type (transient, tiny) calcium channels are **low-voltage-activated (LVA)** channels
that occupy a unique niche in neuronal excitability. Unlike high-voltage-activated (HVA)
channels (L-type, N-type, P/Q-type) that require strong depolarisation to open, T-type
channels activate near the resting potential (-65 to -50 mV) and inactivate with
sustained depolarisation.

Three genes encode T-type channels in mammals:
- **CaV3.1 (α1G):** Fastest kinetics, dominant in thalamic relay neurons
- **CaV3.2 (α1H):** Intermediate kinetics, hippocampus, sensory neurons
- **CaV3.3 (α1I):** Slowest kinetics, thalamic reticular nucleus

### The low-threshold spike (LTS)

The defining electrophysiological signature of T-type channels is the **low-threshold
spike (LTS)** — a broad, slow calcium depolarisation (duration ~20–50 ms, amplitude
~20–30 mV) that is distinct from the fast Na⁺ action potential. The LTS can trigger
a **burst of Na⁺ spikes** riding on top of the calcium plateau.

The LTS requires a specific sequence:
1. **Prolonged hyperpolarisation** (below -75 mV for >100 ms): removes inactivation
   (de-inactivation), priming T-type channels for opening
2. **Release from hyperpolarisation** or mild depolarisation: T-type channels open,
   producing a regenerative Ca²⁺ influx (the LTS)
3. **Na⁺ burst:** The LTS depolarisation is sufficient to trigger 2–5 fast Na⁺ APs
4. **T-type inactivation:** The depolarisation during the burst inactivates T-type,
   terminating the LTS

### Physiological roles

1. **Rebound bursting:** In thalamocortical relay neurons, inhibitory input from the
   thalamic reticular nucleus (TRN) hyperpolarises relay cells, de-inactivating T-type.
   On release from inhibition, the resulting rebound burst is the cellular mechanism
   underlying sleep spindles (7–14 Hz) and delta oscillations (0.5–4 Hz).

2. **Sleep spindle generation:** The reciprocal TRN–relay circuit generates spindles:
   TRN inhibits relay → relay rebounds via T-type → relay excites TRN → TRN inhibits
   relay → cycle repeats at 7–14 Hz.

3. **Tonic vs burst mode switching:** Thalamocortical neurons operate in two modes:
   - **Tonic mode** (depolarised, T-type inactivated): faithful relay of sensory input
   - **Burst mode** (hyperpolarised, T-type de-inactivated): generates rhythmic bursts,
     disrupts faithful relay (associated with sleep/drowsiness)

4. **Pain signalling:** T-type channels in dorsal root ganglia neurons contribute to
   low-threshold mechano-sensation and pain signalling. CaV3.2 is a validated
   analgesic target.

5. **Cortical layer V:** Pyramidal neurons in cortical layer V exhibit T-type-dependent
   burst firing, important for corticothalamic feedback.

### Window current

T-type channels exhibit a "window current" in the voltage range where activation and
inactivation curves overlap (approximately -65 to -50 mV). In this range, a small
fraction of channels are both activated and not yet inactivated, producing a sustained
(non-transient) Ca²⁺ influx. The model uses g_t = 0.1 mS/cm² (reduced compared to
biological estimates) specifically to avoid excessive window current at rest.

---

## Mathematical Model

### Overview

The TTypeCaNeuron model extends the Wang–Buzsáki conductance-based framework with a
T-type Ca²⁺ current. The T-type current uses instantaneous activation (m_T,∞²) and a
slow inactivation gate (s) with voltage-dependent kinetics plus spike-triggered
inactivation.

The model has **four state variables**: V, h (Na⁺ inactivation), n (Kdr activation),
and s (T-type inactivation).

### Membrane equation

$$C_m \frac{dV}{dt} = -I_{Na} - I_K - I_T - I_L + I_{ext}$$

where $C_m = 1.0 \; \mu\text{F/cm}^2$ and $I_{ext} = \text{gain} \times I_{input}$.

### Sodium current (transient, WB)

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$

$$m_\infty = \frac{\alpha_m}{\alpha_m + \beta_m}$$

$$\alpha_m(V) = \frac{0.1 \, (V + 35)}{1 - \exp\!\bigl(-(V + 35)/10\bigr)}$$

$$\beta_m(V) = 4 \, \exp\!\bigl(-(V + 60)/18\bigr)$$

### Na⁺ inactivation gate h

$$\frac{dh}{dt} = \phi \, \bigl[\alpha_h (1 - h) - \beta_h \, h\bigr]$$

$$\alpha_h(V) = 0.07 \, \exp\!\bigl(-(V + 58)/20\bigr)$$

$$\beta_h(V) = \frac{1}{1 + \exp\!\bigl(-(V + 28)/10\bigr)}$$

### Delayed-rectifier K⁺ current (WB)

$$I_K = g_K \, n^4 \, (V - E_K)$$

$$\frac{dn}{dt} = \phi \, \bigl[\alpha_n (1 - n) - \beta_n \, n\bigr]$$

$$\alpha_n(V) = \frac{0.01 \, (V + 34)}{1 - \exp\!\bigl(-(V + 34)/10\bigr)}$$

$$\beta_n(V) = 0.125 \, \exp\!\bigl(-(V + 44)/80\bigr)$$

### T-type Ca²⁺ current

$$I_T = g_T \, m_{T,\infty}^2 \, s \, (V - E_{Ca})$$

Note: E_Ca = 120 mV, so I_T is always inward (depolarising) for physiological V.

**Instantaneous activation:**

$$m_{T,\infty}(V) = \frac{1}{1 + \exp\!\bigl(-(V + 52)/5\bigr)}$$

| V (mV) | m_T,∞ | m_T,∞² | Interpretation |
|---------|-------|--------|----------------|
| -80 | 0.004 | 1.4×10⁻⁵ | Negligible |
| -65 | 0.07 | 0.005 | Minimal activation |
| -55 | 0.27 | 0.07 | Moderate (window current range) |
| -52 | 0.50 | 0.25 | Half-maximal |
| -45 | 0.80 | 0.64 | Strong activation |
| -30 | 0.99 | 0.97 | Near maximal |

**Inactivation gate s:**

$$s_\infty(V) = \frac{1}{1 + \exp\!\bigl((V + 81)/4\bigr)}$$

Note the **positive** sign and the steep slope factor (k = 4 mV) — T-type
inactivation is sharp:

| V (mV) | s_∞ | Interpretation |
|---------|-----|----------------|
| -100 | 0.99 | Fully de-inactivated |
| -90 | 0.98 | Nearly de-inactivated |
| -85 | 0.73 | Partial de-inactivation |
| -81 | 0.50 | Half-inactivation |
| -75 | 0.12 | Mostly inactivated |
| -65 | 0.01 | Almost fully inactivated |
| -50 | 10⁻⁷ | Completely inactivated |

**Time constant:**

$$\tau_s(V) = 30 + \frac{100}{1 + \exp\!\bigl((V + 75)/10\bigr)}$$

| V (mV) | τ_s (ms) | Interpretation |
|---------|----------|----------------|
| -100 | 129 | Slow de-inactivation |
| -90 | 124 | |
| -80 | 109 | |
| -75 | 80 | Half-range |
| -65 | 42 | Faster at depolarised potentials |
| -50 | 31 | Near minimum |
| -30 | 30 | Minimum (30 ms floor) |

The asymmetry is important: de-inactivation during hyperpolarisation is slow
(~100–130 ms), while inactivation during depolarisation is faster (~30–40 ms).
This means recovery from inactivation requires prolonged hyperpolarisation.

### Spike-triggered T-type inactivation

On each Na⁺ spike (V crossing threshold):

$$s \leftarrow 0.3 \times s$$

This multiplicative reduction models the strong inactivation of T-type channels
during the action potential. Each spike reduces s by 70%. After a burst of 3 spikes:
$s \to 0.3^3 \times s_0 = 0.027 \times s_0$ — essentially complete inactivation.

This mechanism is key to burst termination: T-type provides the initial LTS, but
as Na⁺ spikes ride on top and each inactivates T-type, the LTS collapses.

### Leak current

$$I_L = g_L \, (V - E_L)$$

Note: g_L = 0.2 mS/cm² (doubled from standard WB), as in IhNeuron.

### Numerical integration

Forward Euler, 50 sub-steps per call:
$$\Delta t_{sub} = \frac{0.5}{50} = 0.01 \; \text{ms}$$

The s gate is updated **without** the φ scaling factor (same as Ih r gate):
`self.s += sub_dt * (s_inf - self.s) / tau_s`

### Safety bounds and invalid-input atomicity (fail-closed contract)

| Variable | Lower | Upper |
|----------|-------|-------|
| V | -100 mV | +60 mV |
| h | 0.0 | 1.0 |
| n | 0.0 | 1.0 |
| s | 0.0 | 1.0 |

`step(current)` validates before touching state and computes the whole
update on candidate values: a non-finite `current` (NaN, ±∞) raises
`ValueError("current must be finite")` with no state change and no
spike; an out-of-bounds configuration (descriptor ranges in
`TTypeCaNeuron.toml`) raises `ValueError` at construction and at each
step; a candidate that becomes non-finite mid-integration raises
`ValueError("T-type candidate state became non-finite")` with the
pre-step state preserved exactly. The production Rust engine
(`try_step`), the PyO3 binding (typed `ValueError`), the standalone
safety Rust, Go (`TryStep`), and Julia (`ArgumentError`) enforce the
same contract; the engine `step` and Go `Step` legacy wrappers fail
closed by returning 0 without mutating state.

---

## Analytical Properties

### Rebound burst mechanism

Step-by-step analysis of the rebound burst:

1. **Hyperpolarisation phase** (V held at -90 mV for 500 ms):
   - s_∞(-90) = 0.98, τ_s(-90) = 124 ms → s recovers toward 0.98
   - After 500 ms (~4τ_s): s ≈ 0.97 (de-inactivated)
   - m_T,∞(-90) = 0.004 → essentially no T-type current yet

2. **Release from hyperpolarisation** (input removed):
   - V rises toward V_rest (-65 mV) driven by leak
   - As V crosses -65 mV: m_T,∞ ≈ 0.07, m²_T ≈ 0.005
   - But s ≈ 0.97, so I_T = 0.1 × 0.005 × 0.97 × (-65-120) = -0.090 µA/cm²
   - This is a depolarising current (V < E_Ca)

3. **LTS generation** (regenerative):
   - As V rises, m_T,∞ increases sharply (slope factor = 5 mV)
   - At V = -55: m²_T ≈ 0.07, I_T = 0.1 × 0.07 × 0.97 × 185 = -1.26 µA/cm²
   - This large inward current drives further depolarisation → LTS

4. **Na⁺ burst:**
   - LTS brings V above Na⁺ spike threshold (-20 mV)
   - Na⁺ AP fires, resets V to -65, s *= 0.3 → s ≈ 0.29
   - Second spike (if LTS still active): s *= 0.3 → s ≈ 0.09
   - Third spike: s *= 0.3 → s ≈ 0.026 → T-type essentially off → burst ends

### Window current at rest

At V = -65 mV (resting potential):
- m_T,∞² = 0.005
- s_∞ = 0.01
- I_T = 0.1 × 0.005 × 0.01 × (-65 - 120) = -9.25 × 10⁻⁶ µA/cm²

Negligible. The reduced g_t = 0.1 ensures window current doesn't destabilise rest.

### Tonic vs burst mode

| Mode | Holding V | s value | T-type state | Firing pattern |
|------|-----------|---------|-------------|----------------|
| Tonic | > -60 mV | ~0.01 | Inactivated | Regular spiking (WB-like) |
| Burst | < -80 mV | ~0.95 | De-inactivated | Rebound bursts on release |
| Transition | -65 mV | ~0.1–0.5 | Mixed | Single spikes or weak bursts |

---

## Comparison: T-type vs Other Ca²⁺ Channels

| Property | T-type (this model) | L-type (CaV1.x) | N-type (CaV2.2) |
|----------|--------------------|--------------------|------------------|
| Activation range | LVA (-65 to -50) | HVA (>-30) | HVA (>-20) |
| Inactivation | Fast (30–130 ms) | Very slow or none | Intermediate |
| Unitary conductance | ~7 pS (tiny) | ~25 pS | ~13 pS |
| Function | LTS, rebound burst | Sustained Ca²⁺ entry | Neurotransmitter release |
| Pharmacology | Mibefradil, NiCl₂ | Nifedipine, DHPs | ω-conotoxin GVIA |

---

## Effect of Parameters on Behaviour

### T-type conductance (g_T)

| g_T (mS/cm²) | Expected behaviour |
|---------------|-------------------|
| 0.0 | Pure WB model (no LTS, no rebound) |
| 0.05 | Weak LTS, single rebound spike |
| 0.1 (default) | Moderate LTS, 2–3 spike burst |
| 0.2 | Strong LTS, longer bursts |
| 0.5 | Dominant T-type, spontaneous bursting possible |

### Ca²⁺ reversal potential (E_Ca)

| E_Ca (mV) | Driving force at V=-55 | Effect |
|-----------|----------------------|--------|
| 120 (default) | 175 mV | Standard |
| 80 | 135 mV | Reduced LTS amplitude |
| 60 | 115 mV | Weak LTS |

### Spike inactivation factor

The code uses `s *= 0.3` on spike. If changed:

| Factor | Per-spike s reduction | Burst length |
|--------|----------------------|-------------|
| 0.1 | 90% | Very short (1–2 spikes) |
| 0.3 (default) | 70% | Short (2–3 spikes) |
| 0.5 | 50% | Medium (3–5 spikes) |
| 0.8 | 20% | Long (5–10 spikes) |
| 1.0 | 0% (no inactivation) | Unbounded (no termination mechanism) |

---

## Parameters

All defaults from `TTypeCaNeuron::new()` in `channels/t_type_ca.rs:54`:

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -65.0 | mV | Membrane potential (initial) |
| `h` | 0.6 | — | Na⁺ inactivation gate |
| `n` | 0.32 | — | Kdr activation gate |
| `s` | 0.9 | — | T-type Ca²⁺ inactivation gate |
| `g_na` | 35.0 | mS/cm² | Na⁺ maximal conductance |
| `g_k` | 9.0 | mS/cm² | Delayed-rectifier K⁺ conductance |
| `g_t` | 0.1 | mS/cm² | T-type Ca²⁺ conductance |
| `g_l` | 0.2 | mS/cm² | Leak conductance (doubled) |
| `e_na` | 55.0 | mV | Na⁺ reversal potential |
| `e_k` | -90.0 | mV | K⁺ reversal potential |
| `e_ca` | 120.0 | mV | Ca²⁺ reversal potential |
| `e_l` | -65.0 | mV | Leak reversal potential |
| `c_m` | 1.0 | µF/cm² | Membrane capacitance |
| `phi` | 5.0 | — | Kinetic temperature scaling (Na⁺, K⁺ only) |
| `dt` | 0.5 | ms | Integration timestep |
| `v_threshold` | -20.0 | mV | Spike detection threshold |
| `gain` | 1.0 | — | Input current scaling factor |

### Note on s initial value

The default s = 0.9 corresponds to a **partially de-inactivated** state. At the resting
potential V = -65 mV, s_∞ = 0.01 (fully inactivated). The initial s = 0.9 means the
neuron starts as if it has been hyperpolarised — ready to produce a rebound burst with
the first depolarising input. After a few hundred ms at rest, s will decay to ~0.01.

---

## Implementation Details

### Code structure (`channels/t_type_ca.rs:76–134`)

```
step(current) → i32:
    input = gain × current
    sub_steps = 50
    sub_dt = dt / 50

    for each sub-step:
        // WB Na⁺ gating (m instantaneous)
        α_m, β_m → m∞

        // Na⁺ inactivation, Kdr activation
        α_h, β_h, α_n, β_n

        // T-type Ca²⁺ gating
        m_T,∞ = σ(V+52, k=5)
        s∞ = σ(-(V+81), k=4)     ← positive sign in exp
        τ_s = 30 + 100/(1+exp((V+75)/10))

        // Gate updates (s has NO φ scaling)
        h += sub_dt · φ · [α_h(1-h) - β_h·h]
        n += sub_dt · φ · [α_n(1-n) - β_n·n]
        s += sub_dt · (s∞ - s) / τ_s

        // Ionic currents
        I_Na = g_Na · m∞³ · h · (V - E_Na)
        I_K  = g_K  · n⁴  · (V - E_K)
        I_T  = g_T  · m_T,∞² · s · (V - E_Ca)
        I_L  = g_L  · (V - E_L)

        // Voltage update
        dV = (-I_Na - I_K - I_T - I_L + input) / C_m
        V += sub_dt · dV

        // Spike detection + T-type inactivation
        if V ≥ V_threshold:
            fired = 1
            V = -65.0
            s *= 0.3    ← spike-triggered inactivation

    // Post-loop clamps on the candidate, then commit
    V ∈ [-100, +60], h ∈ [0,1], n ∈ [0,1], s ∈ [0,1]
    non-finite input or candidate → ValueError, state unchanged
```

### Key implementation notes

1. **s gate lacks φ scaling:** Like Ih's r gate, the T-type inactivation s evolves on
   its intrinsic timescale (30–130 ms), not accelerated by φ = 5.

2. **m_T is instantaneous and squared:** `m_t_inf.powi(2)` — the squared activation
   models the cooperative gating of T-type channels (two activation particles).

3. **Spike-triggered s *= 0.3:** This is a multiplicative reduction, not a reset.
   Multiple spikes compound: s → 0.3s → 0.09s → 0.027s. This provides a natural
   burst-length limiter.

4. **g_t = 0.1 is reduced:** The code comment says "Reduced to avoid window current at
   rest." Biological estimates for g_T in thalamocortical neurons are ~0.5–2.0 mS/cm².
   The model uses a smaller value to maintain stability.

5. **g_l = 0.2 (doubled):** Same as IhNeuron. This compensates for the tonic T-type
   window current at rest.

---

## Clinical and Pharmacological Relevance

### Pharmacology

| Agent | Action | Model equivalent |
|-------|--------|-----------------|
| Mibefradil | T-type blocker | Set g_t = 0 |
| NiCl₂ (low conc.) | Preferential T-type block | Reduce g_t |
| TTA-P2 | Selective T-type blocker | Set g_t = 0 |
| Ethosuximide | Partial T-type block | Reduce g_t by ~30% |
| Zonisamide | T-type modulator | Reduce g_t |

### Clinical conditions

1. **Absence epilepsy:** Enhanced T-type currents in thalamocortical circuits produce
   3 Hz spike-wave discharges. Ethosuximide (first-line treatment) partially blocks
   T-type channels.

2. **Neuropathic pain:** CaV3.2 upregulation in DRG neurons contributes to pain
   hypersensitivity. T-type blockers are under investigation as analgesics.

3. **Sleep disorders:** T-type mediates the transition between sleep stages.
   Alterations in T-type expression affect sleep spindle generation.

---

## FPGA Implementation Notes

### Resource estimates (Zynq-7020, analytical)

| Component | Resource | Estimate |
|-----------|----------|----------|
| Multipliers | DSP48E1 | 18–22 slices |
| State registers | Flip-flops | ~256 bits (4 × 64-bit state) |
| Exponentials | LUT-based | 6 exp() per sub-step |
| Total LUTs | | ~3,500–4,500 |
| Pipeline depth | Cycles | ~15–20 per sub-step |
| Total latency | Cycles | ~750–1,000 at 100 MHz → 7.5–10 µs |

**Key optimisation:** The spike-triggered `s *= 0.3` is a single multiply that occurs
only during spike sub-steps. In FPGA, this can be a conditional multiply-by-constant
(shift+add approximation of 0.3 ≈ 0.25 + 0.03125 + ...).

**Note:** These are analytical estimates, not measured synthesis results.

---

## Pipeline Status

| Checklist | Status |
|-----------|--------|
| Rust implementation | `engine/src/neurons/channels/t_type_ca.rs` (`try_step`, atomic rejection) |
| PyO3 wrapper | `engine/src/bindings/channels/t_type_ca_neuron.rs` (typed `ValueError`; state: v, h, n, s) |
| NetworkRunner wired | `NeuronVariant::TTypeCa` (`engine/src/network_runner/model_factory.rs`) |
| `create_neuron("TTypeCa")` | Yes |
| `supported_models()` | Includes "TTypeCa" |
| Standalone safety Rust | `src/sc_neurocore/accel/rust/safety/ttype_ca_neuron.rs` (full recurrence) |
| Go service | `src/sc_neurocore/accel/go/services/ttype_ca_neuron.go` (`TryStep`, full recurrence) |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/ttype_ca_neuron.jl` (atomic `ArgumentError`) |
| Mojo | not implemented; no kernel exists and no parity is claimed |
| Silicon / RTL | not implemented; no HDL parity claimed |
| Module-owned tests | `tests/test_model_ttype_ca_neuron.py`, `tests/test_ttype_ca_neuron_backends.py` |
| Backend parity | Rust engine, safety Rust, Go, Julia vs Python: 64-step complete state ≤ 1e-12 |
| Benchmark | `ttype_ca_1k_steps`: **3.94 ms** (3.94 µs/step), i5-11600K |

---

## Benchmark (Criterion, i5-11600K @ 3.90 GHz)

| Benchmark | Median |
|-----------|-------:|
| ttype_ca_1k_steps | 3.94 ms |
| Per step | **3.94 µs** |

**Context:** Comparable to BKNeuron (3.16 µs/step). The T-type computation adds
~25% overhead vs bare WB, primarily from the 3 extra exponentials per sub-step
(m_T,∞, s_∞, τ_s).

Measured 2026-04-04 on i5-11600K @ 3.90 GHz, Criterion.rs, 100 iterations.

---

## Usage Example

### Python

```python
from sc_neurocore_engine import TTypeCaNeuron

neuron = TTypeCaNeuron()

# Demonstrate rebound burst: hyperpolarise then release
voltages = []
spikes = []
for step in range(6000):  # 3 seconds
    if 200 <= step < 2000:  # Hyperpolarise from 100–1000 ms
        fired = neuron.step(-3.0)
    else:
        fired = neuron.step(0.0)  # Release at 1000 ms
    voltages.append(neuron.v)
    if fired:
        spikes.append(step)

# Expected: burst of 2-3 spikes around step 2000 (rebound)
print(f"Rebound spikes: {[s for s in spikes if 1900 < s < 2200]}")
print(f"s after burst: {neuron.s:.3f}")
```

### Rust

```rust
use sc_neurocore_engine::neurons::channels::TTypeCaNeuron;

let mut neuron = TTypeCaNeuron::new();

// Hyperpolarise for 500 ms
for _ in 0..1000 { neuron.step(-3.0); }

// Release — expect rebound burst
let mut burst_spikes = 0;
for _ in 0..200 {
    burst_spikes += neuron.step(0.0);
}

println!("Rebound spikes: {}, s: {:.3}", burst_spikes, neuron.s);
```

---

## Findings

1. **Fires with excitatory input.** Sustained spiking with I = 2. Verified.
2. **Silent without input.** No spontaneous firing at rest. Verified.
3. **Rebound burst.** After prolonged hyperpolarisation, de-inactivated T-type
   facilitates burst firing on release. Verified.
4. **s gate de-inactivates at hyperpolarised potentials.** s increases during negative
   input, approaching s_∞(-90) ≈ 0.98. Verified.
5. **Spike inactivates T-type.** Each spike reduces s by 70% (s *= 0.3). Verified.
6. **Reset clears dynamic state only.** V = -65, h = 0.6, n = 0.32, s = 0.9;
   parameters are preserved. Verified.
7. **Invalid input is rejected atomically.** Non-finite drive, configuration,
   or candidate raises `ValueError` with the pre-step state preserved. Verified.
8. **Gating bounds.** h ∈ [0,1], n ∈ [0,1], s ∈ [0,1] enforced. Verified.

---

## References

1. Huguenard JR (1996). Low-threshold calcium currents in central nervous system neurons.
   *Annu Rev Physiol* 58:329–348.

2. Destexhe A, Bal T, McCormick DA, Sejnowski TJ (1996). Ionic mechanisms underlying
   synchronized oscillations and propagating waves in a model of ferret thalamic slices.
   *J Neurophysiol* 76:2049–2070.

3. Wang X-J, Buzsáki G (1996). Gamma oscillation by synaptic inhibition in a hippocampal
   interneuronal network model. *J Neurosci* 16:6402–6413.

4. Perez-Reyes E (2003). Molecular physiology of low-voltage-activated T-type calcium
   channels. *Physiol Rev* 83:117–161.

5. McCormick DA, Huguenard JR (1992). A model of the electrophysiological properties of
   thalamocortical relay neurons. *J Neurophysiol* 68:1384–1400.

6. Destexhe A, Neubig M, Ulrich D, Huguenard JR (1998). Dendritic low-threshold calcium
   currents in thalamic relay cells. *J Neurosci* 18:3574–3588.

7. Steriade M, McCormick DA, Sejnowski TJ (1993). Thalamocortical oscillations in the
   sleeping and aroused brain. *Science* 262:679–685.

8. Coulter DA, Huguenard JR, Prince DA (1989). Characterisation of ethosuximide reduction
   of low-threshold calcium current in thalamic neurons. *Ann Neurol* 25:582–593.

9. Crunelli V, Cope DW, Hughes SW (2006). Thalamic T-type Ca²⁺ channels and NREM sleep.
   *Cell Calcium* 40:175–190.

10. Bourinet E, Alloui A, Monteil A, et al. (2005). Silencing of the CaV3.2 T-type calcium
    channel gene in sensory neurons demonstrates its major role in nociception. *EMBO J*
    24:315–324.

11. Llinás RR, Steriade M (2006). Bursting of thalamic neurons and states of vigilance.
    *J Neurophysiol* 95:3297–3308.

12. Jahnsen H, Llinás R (1984). Electrophysiological properties of guinea-pig thalamic
    neurones: an in vitro study. *J Physiol* 349:205–226.
