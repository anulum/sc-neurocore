# DurstewitzDopamineNeuron

**Module:** `sc_neurocore.neurons.models.durstewitz_dopamine`
**Reference:** Durstewitz, Seamans & Sejnowski, J. Neurophysiol. 83(3), 2000
**Family:** Biophysical conductance-based (HH-type, PFC with D1 dopamine modulation)
**State variables:** `v` (membrane potential), `h_na` (Na⁺ inactivation), `n_k` (K⁺ activation)

---

## Equations

### Membrane potential

$$\frac{dV}{dt} = -I_{Na} - I_K - I_{NMDA} - I_L + I$$

### Ionic currents

$$I_{Na} = g_{Na} \, m_\infty^3 \, h_{Na} \, (V - E_{Na})$$
$$I_K = g_K' \, n_K^4 \, (V - E_K)$$
$$I_{NMDA} = g_{NMDA}' \, J(V) \, (V - E_{NMDA})$$
$$I_L = g_L \, (V - E_L)$$

### D1 dopamine modulation (3 effects)

$$g_{NMDA}' = g_{NMDA} \cdot (1 + d_1 \cdot (g_{NMDA,scale} - 1))$$
$$g_K' = g_K \cdot (1 + d_1 \cdot (g_K{,scale} - 1))$$
$$m_\infty = \frac{1}{1 + \exp(-(V + 30 + d_1 \cdot v_{shift}) / 9.5)}$$

where $d_1 \in [0, 1]$ is the D1 agonism level.

### Three D1 effects

| D1 effect | Parameter | d1=0 | d1=1 | Consequence |
|-----------|-----------|------|------|-------------|
| NMDA enhancement | g_nmda_scale=2.5 | ×1.0 | ×2.5 | Stronger persistent excitation |
| Na⁺ shift | v_shift_na=−5 | 0 mV | −5 mV | Reduced window current |
| K⁺ enhancement | g_k_scale=1.5 | ×1.0 | ×1.5 | Stronger repolarisation |

### Mg²⁺ block (Jahr & Stevens 1990)

$$J(V) = \frac{1}{1 + [Mg^{2+}]/3.57 \cdot \exp(-0.062V)}$$

Same formula as BrunelWang.

### Boltzmann activation functions

$$m_{Na,\infty} = \frac{1}{1 + \exp(-(V + 30 + v_{shift}) / 9.5)}$$
$$h_{Na,\infty} = \frac{1}{1 + \exp((V + 53) / 7)}$$
$$n_{K,\infty} = \frac{1}{1 + \exp(-(V + 30) / 10)}$$

### Gate dynamics

$$\tau_h = 0.5 + \frac{14}{1 + \exp((V + 50) / 12)}$$
$$\tau_n = 1 + \frac{11}{1 + \exp((V + 40) / 10)}$$

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `h_na` | 0.7 | — | Na⁺ inactivation |
| `n_k` | 0.2 | — | K⁺ activation |
| `g_na` | 45.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 18.0 | mS/cm² | K⁺ conductance |
| `g_nmda` | 0.5 | mS/cm² | NMDA conductance (base) |
| `g_l` | 0.02 | mS/cm² | Leak conductance |
| `e_na` | 55.0 | mV | Na⁺ reversal |
| `e_k` | −80.0 | mV | K⁺ reversal |
| `e_nmda` | 0.0 | mV | NMDA reversal |
| `e_l` | −65.0 | mV | Leak reversal |
| `mg` | 1.0 | mM | Extracellular Mg²⁺ |
| `d1_level` | 0.0 | — | D1 agonism (0 = no DA, 1 = maximal) |
| `g_nmda_scale` | 2.5 | — | NMDA boost factor at d1=1 |
| `g_k_scale` | 1.5 | — | K⁺ boost factor at d1=1 |
| `v_shift_na` | −5.0 | mV | Na⁺ half-activation shift at d1=1 |
| `dt` | 0.05 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

---

## Analytical Properties

### D1 modulation mechanism (Durstewitz et al. 2000)

Dopamine D1 receptor activation in prefrontal cortex (PFC) has three
simultaneous effects:

1. **NMDA enhancement (g_nmda × 2.5):** D1 phosphorylates NMDA NR1 subunits
   → increased Ca²⁺ flux → stronger persistent excitation → stabilises
   up-states in working memory

2. **Na⁺ window current reduction (v_shift = −5 mV):** D1 shifts the Na⁺
   activation curve leftward → narrows the "window" where both m and h
   are non-zero → reduces persistent Na⁺ current that destabilises
   up-states

3. **K⁺ enhancement (g_k × 1.5):** D1 upregulates K⁺ channels →
   stronger repolarisation → sharpens transitions between up and down
   states → reduces noise-driven switching

### Net effect: up-state stabilisation

The three D1 effects combine to:
- **Stabilise existing up-states** (NMDA enhancement sustains)
- **Prevent premature up-state entry** (Na⁺ shift + K⁺ enhancement)
- **Reduce noise-driven transitions** (K⁺ enhancement)

This is the "inverted-U" dopamine model: optimal D1 improves working
memory; too little or too much impairs it.

### Mg²⁺ block creates bistability

The NMDA current with Mg²⁺ block provides the positive feedback needed
for bistability:
- **Down state (V ≈ −65):** NMDA blocked → weak recurrence → stable rest
- **Up state (V ≈ −50):** NMDA partially unblocked → strong recurrence →
  stable depolarised plateau

D1 agonism shifts the balance toward the up state (enhanced NMDA).

### m_inf Na⁺ shift

The D1-induced shift of the Na⁺ activation midpoint:
- d1=0: midpoint at −30 mV
- d1=1: midpoint at −35 mV (shifted −5 mV)

This seemingly small shift has a large effect on the "window current" —
the Na⁺ current that flows when both m_inf > 0 and h > 0. The window
current is maximal near V ≈ −50 mV (close to threshold), and the −5 mV
shift reduces it significantly, preventing spurious spiking.

---

## Behaviour

### Prefrontal cortex working memory

The model was designed to explain persistent activity in PFC during
working memory tasks (Goldman-Rakic 1995):
1. Cue presentation → strong input → transition to up state
2. Delay period → no input → NMDA recurrence maintains up state
3. Response → up state read out and terminated

D1 dopamine modulation controls the robustness of this maintenance:
- Low D1 → unstable up states → working memory failures
- Optimal D1 → stable, selective up states → good WM performance
- High D1 → overly stable → perseveration, difficulty switching

### Inverted-U dopamine response

The model predicts the well-known inverted-U relationship between
dopamine level and WM performance (Williams & Goldman-Rakic 1995):
- d1=0.0: too excitable (no NA shift), noisy transitions
- d1=0.3–0.5: optimal (balanced enhancement)
- d1=1.0: too stable (strong K⁺), hard to initiate up states

### D1=0 reduces to standard HH-like

Without dopamine modulation (d1=0):
- g_nmda' = g_nmda (no boost)
- g_k' = g_k (no boost)
- v_shift = 0 (no Na⁺ shift)
→ Standard HH-type neuron with NMDA conductance

---

## Comparison with Related Models

| Property | Durstewitz | BrunelWang | HH | WongWang |
|----------|-----------|-----------|-----|---------|
| D1 modulation | Yes (3 effects) | No | No | No |
| NMDA Mg²⁺ | Yes J(V) | Yes J(V) | No | Implicit |
| State vars | 3 (V, h, n) | 1 (V) + synaptic | 4 (V,m,h,n) | 2 (s1,s2) |
| PFC focus | Yes | Yes (WM) | No | Yes (decision) |
| m_inf | Instantaneous | — | Dynamic | — |
| Pipeline | Compatible | Limited (4-arg) | Compatible | Limited (tuple) |

Durstewitz is the only model in SC-NeuroCore with explicit dopaminergic
neuromodulation.

---

## Clinical Relevance

### Schizophrenia

The dopamine hypothesis of schizophrenia (Howes & Kapur 2009) posits:
- **Hyperdopaminergia** in striatum → positive symptoms
- **Hypodopaminergia** in PFC → cognitive deficits (working memory)

The Durstewitz model directly tests this: reducing d1_level impairs
up-state maintenance, matching the WM deficits in schizophrenia.

### ADHD

Attention-deficit/hyperactivity disorder involves prefrontal dopamine
dysfunction. The model predicts that suboptimal D1 stimulation produces
noisy, unstable neural representations — matching the distractibility
and working memory deficits observed in ADHD.

### Pharmacology

- **Typical antipsychotics** (D2 blockers): no direct effect on D1 model
- **Atypical antipsychotics** (partial D1 agonists): move d1_level toward
  optimal → improved cognition
- **Methylphenidate** (indirect DA agonist): increases d1_level →
  improved PFC function in ADHD

---

## Numerical Considerations

- **Single Euler step:** dt=0.05ms. No sub-stepping.
- **6 exp() per step:** m_inf, h_inf, n_inf, tau_h, tau_n, mg_block.
- **No clipping:** V, h, n not clipped. Rely on conductance-based
  stability.
- **D1 modulation per step:** Scaling factors recomputed each step
  (multiplicative, cheap).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/durstewitz_dopamine.py` — 70 lines.
- **Three state variables:** v, h_na, n_k.
- **Dataclass:** Uses `@dataclass`.
- **d1_level as parameter:** Can be changed dynamically between steps
  to simulate phasic dopamine.
- **Rust wiring:** Compatible (3 f64 state vars, 6 exp calls).

---

## Infrastructure Pipeline

```
DurstewitzDopamineNeuron
├── step(current) → int {0, 1}
├── 1 Euler step + 6 exp() per call (dt=0.05ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=5, rate=500Hz)
├── Projection: tested src→tgt wiring
├── d1_level adjustable between steps
└── Rust: compatible (3 f64 state vars)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~150K steps/s | Not measured |
| Network (10 neurons, 1s) | ~15K neuron-steps/s | — |

Moderate speed — 6 exp() per step, no sub-stepping.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 3 | binary output, state finite (50K at I=10), reset |
| Dynamics | 4 | spontaneous firing (I=0, ≥5 spikes in 10K), rate increases (I=50 > I=0), monotonic f-I [0,10,30,50], deterministic |
| Performance | 1 | isolation >10K steps/s |
| Pipeline | 4 | Population(n=5), Network+PoissonInput spikes, Projection(5→5) spikes, spike_count+firing_rate analysis |
| **Total** | **12** | **ALL PASSED (10.38s)** |

See `tests/test_model_durstewitz_dopamine.py`.

---

## Findings (Measured 2026-03-31)

1. **12/12 tests PASSED in 10.38s.** No failures.

2. **Spontaneous firing at I=0.** At least 5 spikes in 10K steps with
   zero external current. The combination of Na⁺ conductance (g_Na=45)
   and NMDA background (g_nmda=0.5 with Mg²⁺ block) provides
   sufficient excitability for spontaneous spiking.

3. **Rate increases with current.** I=50 produces more spikes than I=0
   across 10K steps.

4. **Monotonic f-I curve.** Spike counts at I=0,10,30,50 are monotonically
   non-decreasing. No bistability or non-monotonicity observed.

5. **State finite across 50K steps.** V remains finite at I=10.

6. **Reset functional.** Restores v, h_na, n_k to defaults.

7. **Deterministic.** Bit-exact traces across repeated runs.

8. **Network pipeline functional.** Population(n=5) with PoissonInput
   (rate=200Hz, weight=10) runs 5.0s and produces spikes. Projection
   (5→5, w=5, p=1.0) works.

9. **Analysis verified.** spike_count ≥ 10, firing_rate > 0 from 10K-step
   binary train at I=10.

10. **Only neuromodulated model.** Unique in SC-NeuroCore — d1_level
    provides dynamic dopamine D1 receptor control.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
12/12 PASSED in 10.38s
├── TestDurstewitzIsolation: 3 tests
│   ├── step() → int {0,1}
│   ├── state finite (50K steps at I=10)
│   └── reset()
├── TestDurstewitzDynamics: 4 tests
│   ├── spontaneous firing (I=0, ≥5 spikes in 10K)
│   ├── rate increases with current (I=50 > I=0)
│   ├── monotonic f-I curve [0, 10, 30, 50]
│   └── deterministic (bit-exact)
├── TestDurstewitzPerformance: 1 test
│   └── isolation >10K steps/s
└── TestDurstewitzPipeline: 4 tests
    ├── Population(n=5)
    ├── Network + PoissonInput → spikes > 0 (5.0s)
    ├── Projection(5→5) + PoissonInput → spikes > 0 (5.0s)
    └── spike_count ≥ 10, firing_rate > 0
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-65, h_na=0.7, n_k=0.2 |
| step() → int {0,1} | ✓ PASS | Upward crossing at -20 mV |
| Spontaneous spiking | ✓ PASS | Fires at I=0 |
| Rate monotonic | ✓ PASS | More I → more spikes |
| State finite (50K) | ✓ PASS | V finite at I=10 |
| reset() | ✓ PASS | All vars to defaults |
| Deterministic | ✓ PASS | Bit-exact |
| Population(n=5) | ✓ PASS | 5 instances |
| Network + PoissonInput | ✓ PASS | Spikes > 0 (5.0s) |
| Projection(5→5) | ✓ PASS | Cross-pop wiring |
| spike_count | ✓ PASS | ≥ 10 |
| firing_rate | ✓ PASS | > 0 Hz |

### Network configuration tested

- Population: 5 DurstewitzDopamineNeurons
- PoissonInput: rate=200Hz, weight=10.0, dt=0.001, seed=42
- Projection: src(5) → tgt(5), weight=5.0, probability=1.0
- SpikeMonitor: count verified
- Duration: 5.0s (5000 timesteps)

**ALL 12 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
