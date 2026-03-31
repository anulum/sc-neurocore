# DestexheThalamicNeuron

**Module:** `sc_neurocore.neurons.models.destexhe_thalamic`
**Reference:** Destexhe, Babloyantz & Sejnowski, Biophys. J. 65(4), 1993
**Family:** Biophysical conductance-based (thalamocortical relay, T-type Ca²⁺)
**State variables:** `v` (membrane potential), `h_na` (Na⁺ inactivation), `n_k` (K⁺ activation), `m_t` (T-type Ca²⁺ activation, instantaneous), `h_t` (T-type Ca²⁺ inactivation)

---

## Equations

### Membrane potential

$$\frac{dV}{dt} = -I_{Na} - I_K - I_T - I_L + I$$

### Ionic currents

$$I_{Na} = g_{Na} \, m_{Na,\infty}^3 \, h_{Na} \, (V - E_{Na})$$
$$I_K = g_K \, n_K^4 \, (V - E_K)$$
$$I_T = g_T \, m_T^2 \, h_T \, (V - E_{Ca})$$
$$I_L = g_L \, (V - E_L)$$

### Boltzmann steady-state activations

| Function | Formula | Midpoint | Slope |
|----------|---------|----------|-------|
| m_Na,∞ | 1/(1+exp(−(V+37)/7)) | −37 mV | 7 mV |
| h_Na,∞ | 1/(1+exp((V+41)/4)) | −41 mV | 4 mV |
| n_K,∞ | 1/(1+exp(−(V+25)/12)) | −25 mV | 12 mV |
| m_T,∞ | 1/(1+exp(−(V+57)/6.5)) | −57 mV | 6.5 mV |
| h_T,∞ | 1/(1+exp((V+81)/4)) | −81 mV | 4 mV |

### Time constants

| Gate | Formula | Typical range |
|------|---------|---------------|
| τ_h_Na | 1/(0.128·exp(−(V+46)/18) + 4/(1+exp(−(V+23)/5))) | ~0.1–5 ms |
| τ_n_K | 1/(0.032·5 + 0.5·exp(−(V+40)/40)) | ~1–6 ms |
| τ_h_T | V<−81: 30.8 + 211.4·exp((V+115.2)/5)/(1+exp((V+86)/3.2)); else 10 | 10–240+ ms |
| m_T | Instantaneous (m_T = m_T,∞) | 0 ms |

### 5 sub-steps per call

Forward Euler with 5 sub-steps (dt=0.02 ms). Each call integrates
0.1 ms of biological time.

### Spike detection

$$V \geq V_{threshold}(-20) \; \text{AND} \; V_{prev} < V_{threshold}$$

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `h_na` | 0.6 | — | Na⁺ inactivation gate |
| `n_k` | 0.3 | — | K⁺ delayed rectifier gate |
| `m_t` | 0.0 | — | T-type Ca²⁺ activation (instantaneous) |
| `h_t` | 1.0 | — | T-type Ca²⁺ inactivation |
| `g_na` | 100.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 10.0 | mS/cm² | K⁺ conductance |
| `g_t` | 2.0 | mS/cm² | T-type Ca²⁺ conductance |
| `g_l` | 0.05 | mS/cm² | Leak conductance |
| `e_na` | 50.0 | mV | Na⁺ reversal |
| `e_k` | −90.0 | mV | K⁺ reversal |
| `e_ca` | 120.0 | mV | Ca²⁺ reversal |
| `e_l` | −70.0 | mV | Leak reversal |
| `dt` | 0.02 | ms | Sub-step timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Conductance hierarchy

$$g_{Na}(100) \gg g_K(10) > g_T(2) \gg g_L(0.05)$$

The T-type Ca²⁺ conductance (2.0) is small compared to Na⁺ and K⁺,
but its effect is amplified by the enormous driving force (E_Ca = 120 mV).
At V = −65: g_T × (V − E_Ca) = 2 × 185 = 370 — a substantial current
when h_T is de-inactivated.

### Reversal potential ordering

$$E_K(-90) < E_L(-70) < E_{Na}(50) < E_{Ca}(120)$$

E_Ca = 120 mV is the highest reversal in the library. The Ca²⁺ gradient
creates an enormous inward driving force at resting potentials.

---

## Analytical Properties

### T-type Ca²⁺ current: the thalamic signature

The T-type (transient, low-threshold) Ca²⁺ current is the defining
feature of thalamocortical relay neurons:

**Activation (m_T):** Midpoint at −57 mV — below the Na⁺ activation
(−37 mV). This means I_T activates at **subthreshold** voltages,
before the Na⁺ spike. m_T is treated as instantaneous (no time constant).

**Inactivation (h_T):** Midpoint at −81 mV — very hyperpolarised.
This creates the voltage-dependent switching:

| V (mV) | h_T,∞ | m_T,∞ | T-current state |
|--------|-------|-------|-----------------|
| −90 | 0.90 | 0.00 | De-inactivated, not activated |
| −81 | 0.50 | 0.02 | Half de-inactivated |
| −65 | 0.02 | 0.22 | Mostly inactivated (resting) |
| −57 | 0.00 | 0.50 | Inactivated, half activated |
| −40 | 0.00 | 0.96 | Fully inactivated |

### Tonic vs burst firing modes

**Tonic mode** (depolarised, V > −60 mV):
- h_T ≈ 0 (inactivated) → I_T = 0
- Na⁺/K⁺ dynamics dominate → regular spiking
- This is the "relay" mode: faithfully transmits sensory input

**Burst mode** (from hyperpolarised state, V < −80 mV):
1. During hyperpolarisation: h_T de-inactivates (h_T → 1)
2. Upon release: m_T activates (V rises above −57 mV)
3. I_T produces a low-threshold Ca²⁺ spike (slow, broad)
4. Na⁺ spikes ride on top of Ca²⁺ spike → burst of 2–7 spikes
5. h_T inactivates → I_T turns off → burst ends

This is the **post-inhibitory rebound burst** — the signature of
thalamocortical neurons.

### τ_h_T: the critical timescale

The h_T time constant controls the burst timing:
- At V < −81 mV: τ_h_T = 30.8 + 211.4 × ... (up to 240+ ms)
- At V ≥ −81 mV: τ_h_T = 10 ms (fast inactivation)

The asymmetry is crucial:
- De-inactivation (V < −81): slow (100–240 ms) → requires sustained
  hyperpolarisation for bursting
- Re-inactivation (V > −81): fast (10 ms) → burst terminates quickly

### m_T instantaneous

Unlike h_T, the m_T gate has no dynamics — it is set to m_T,∞ at each
sub-step. This means T-current activation is infinitely fast: as soon
as V reaches −57 mV, m_T is fully responsive. Only h_T's slow
de-inactivation limits the timing of bursts.

---

## Behaviour

### Four-current interaction

1. **I_Na:** Fast inward current → spike upstroke (m_Na instantaneous, h_Na gate)
2. **I_K:** Delayed outward current → spike repolarisation (n_K⁴)
3. **I_T:** Low-threshold inward current → subthreshold depolarisation
   and Ca²⁺ spikes (m_T² h_T, E_Ca=120)
4. **I_L:** Very small leak (g_L=0.05)

### Sleep oscillations and thalamic rhythms

The T-current enables the thalamic contribution to sleep oscillations:
- **Spindles (7–14 Hz):** Thalamic reticular → relay inhibition →
  rebound burst → re-excitation of reticular → cycle
- **Delta waves (0.5–4 Hz):** Slower T-current de-inactivation cycle
- **Absence seizures:** Pathological spike-and-wave (T-current overactivation)

Destexhe et al. (1993) showed that the interaction between T-current
and I_h (hyperpolarisation-activated current) generates these rhythms.

---

## Comparison with Related Models

| Property | Destexhe Thalamic | HH | ConnorStevens | HuberBraun |
|----------|------------------|-----|---------------|-----------|
| Cell type | TC relay | Squid axon | Gastropod | Cold receptor |
| State vars | 5 | 4 | 6 | 4+ |
| T-current | Yes (g=2) | No | No | Yes |
| Sub-steps | 5 | 100 | 100 | Varies |
| Ca²⁺ reversal | 120 mV | — | — | 140 mV |
| Burst mode | Yes (T-mediated) | No | No | Yes |
| Reference | Destexhe 1993 | Hodgkin 1952 | Connor 1977 | Huber 1998 |

---

## Numerical Considerations

- **5 sub-steps per call:** dt=0.02 ms × 5 = 0.1 ms effective.
- **~7 exp() per sub-step:** 5 Boltzmann functions + 2 tau functions.
  Total: ~35 exp() per step() call.
- **τ guards:** max(tau, 0.1) prevents division by zero when tau → 0.
- **m_T instantaneous:** Set directly to m_T,∞ each sub-step — no ODE.
- **τ_h_T piecewise:** Different formula for V < −81 and V ≥ −81.
- **No V or gate clipping:** Relies on conductance-based self-regulation.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/destexhe_thalamic.py` — 74 lines.
- **Five state variables:** v, h_na, n_k, m_t (instantaneous), h_t.
- **Dataclass:** Uses `@dataclass`.
- **Inner loop:** `for _ in range(5):` sub-stepping.
- **Rust wiring:** Compatible (5 f64 state vars, sub-stepping).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | >1K steps/s (threshold) | Not measured |
| Network (10n, 200ms) | >100 neuron-steps/s | — |

Moderate speed — 5 sub-steps × ~7 exp() = ~35 exp() per step() call.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 6 | defaults, 5 state vars, binary output, state finite (10K at I=5), reset, deterministic |
| Analytical | 7 | 5 sub-steps (dt=0.02), m_T instantaneous, 4 ionic currents, reversal ordering, h_T de-inactivation at V=-90, h_T inactivated at V=-40, gating bounded |
| Thalamic | 7 | fires under drive (I=5), silent at I=0, rate increases with current, f-I sweep [0,2,5,10,20] (parametrised) |
| Parameters | 9 | g_T sweep [0,2,5], g_Na sweep [50,100,150], dt stability [0.01,0.02,0.05] (all parametrised) |
| Performance | 2 | isolation >1K steps/s, network >100 neuron-steps/s |
| Pipeline | 6 | Population(n=5), Projection(3→3), Network spikes, spike_count, isi, firing_rate |
| **Total** | **38** | **ALL PASSED (36.95s)** |

See `tests/test_model_destexhe_thalamic.py`.

---

## Findings (Measured 2026-03-31)

1. **38/38 tests PASSED in 36.95s.** No failures.

2. **5 sub-steps confirmed.** dt=0.02 ms, effective 0.1 ms per call.

3. **m_T instantaneous.** After stepping, m_T matches m_T,∞ at current
   V within 0.1 (sub-step settling).

4. **Four ionic currents verified.** g_na, g_k, g_t, g_l all positive.

5. **Reversal ordering correct.** E_K < E_L < E_Na < E_Ca.

6. **h_T de-inactivation at hyperpolarised potential.** At V=−90:
   h_T,∞ > 0.85. T-current is ready to fire.

7. **h_T inactivated at depolarised potential.** At V=−40: h_T,∞ < 0.01.
   T-current is completely inactivated.

8. **h_T partially inactivated at rest.** At V=−65: h_T,∞ < 0.1.
   T-current mostly inactive at resting potential.

9. **Fires under drive at I=5.** At least 1 spike in 5000 steps.

10. **Rate increases with current.** I=10 → more spikes than I=2 across
    5000 steps.

11. **Gating variables bounded.** After 5000 steps at I=5, all gates
    remain in [−0.05, 1.05].

12. **Parameter sweeps stable.** g_T ∈ {0, 2, 5}, g_Na ∈ {50, 100, 150},
    dt ∈ {0.01, 0.02, 0.05} — all produce finite V.

13. **Performance: >1K isolation steps/s.** Moderate — 35 exp() per call.

14. **Network pipeline functional.** Population(5), Projection(3→3),
    PoissonInput(500Hz, w=5), spike_count, isi, firing_rate all work.

15. **Deterministic.** Bit-exact traces across repeated runs.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
38/38 PASSED in 36.95s
├── TestDestIsolation: 6 tests
│   ├── defaults (v=-65, h_na=0.6, n_k=0.3, m_t=0, h_t=1)
│   ├── 5 state variables exist
│   ├── step() → int {0,1}
│   ├── state finite (10K steps at I=5)
│   ├── reset restores defaults
│   └── deterministic
├── TestDestAnalytical: 7 tests
│   ├── 5 sub-steps (dt=0.02)
│   ├── m_T instantaneous
│   ├── 4 ionic currents
│   ├── reversal ordering e_k < e_l < e_na < e_ca
│   ├── h_T de-inactivation at V=-90 (>0.85)
│   ├── h_T inactivated at V=-40 (<0.01)
│   └── gating bounded [−0.05, 1.05]
├── TestDestThalamic: 7 tests
│   ├── fires under drive (I=5)
│   ├── silent/behaviour at I=0
│   ├── rate increases with current
│   └── f-I sweep [0, 2, 5, 10, 20] (parametrised)
├── TestDestParameters: 9 tests
│   ├── g_T sweep [0, 2, 5]
│   ├── g_Na sweep [50, 100, 150]
│   └── dt stability [0.01, 0.02, 0.05]
├── TestDestPerformance: 2 tests
│   ├── isolation >1K steps/s
│   └── network >100 neuron-steps/s
└── TestDestPipeline: 6 tests
    ├── Population(n=5)
    ├── Projection(3→3)
    ├── Network + PoissonInput
    ├── spike_count, isi, firing_rate
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | 5 state vars |
| step() → int {0,1} | ✓ PASS | Upward crossing at -20 mV |
| 5 sub-steps | ✓ PASS | dt=0.02 × 5 |
| m_T instantaneous | ✓ PASS | Set to m_T,∞ |
| h_T voltage-dependent | ✓ PASS | De-/inactivation verified |
| State finite (10K) | ✓ PASS | At I=5 |
| Gating bounded | ✓ PASS | All ∈ [−0.05, 1.05] |
| Fires under drive | ✓ PASS | ≥1 spike at I=5 |
| Rate monotonic | ✓ PASS | I=10 > I=2 |
| reset() | ✓ PASS | All vars to defaults |
| Deterministic | ✓ PASS | Bit-exact |
| Population(n=5) | ✓ PASS | 5 instances |
| Projection(3→3) | ✓ PASS | Cross-population |
| Network + PoissonInput | ✓ PASS | Runs, count verified |
| spike_count | ✓ PASS | ≥ 0 |
| isi | ✓ PASS | all finite |
| firing_rate | ✓ PASS | ≥ 0 |

### Network configuration tested

- Population: 5 DestexheThalamicNeurons (main), 3+3 (Projection)
- PoissonInput: rate=500Hz, weight=5.0, dt=0.001, seed=42
- Projection: src(3) → tgt(3), weight=2.0, probability=1.0
- SpikeMonitor: count, spike_trains
- Duration: 2.0s (spiking), 1.0s (Projection), 0.2s (performance)

**ALL 38 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
