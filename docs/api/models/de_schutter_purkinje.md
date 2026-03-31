# DeSchutterPurkinjeNeuron

**Module:** `sc_neurocore.neurons.models.de_schutter_purkinje`
**Reference:** De Schutter & Bower, J. Neurophysiol. 71(1), 1994
**Family:** Biophysical conductance-based (cerebellar Purkinje cell, simplified)
**State variables:** `v`, `h_na`, `n_k`, `m_cap`, `h_cap`, `q_kca`, `ca` (7 variables total)

---

## Equations

### Membrane potential

$$\frac{dV}{dt} = -I_{Na} - I_K - I_{CaP} - I_{K(Ca)} - I_L + I$$

### Five ionic currents

$$I_{Na} = g_{Na} \, m_{Na,\infty}^3 \, h_{Na} \, (V - E_{Na})$$
$$I_K = g_K \, n_K^4 \, (V - E_K)$$
$$I_{CaP} = g_{CaP} \, m_{CaP}^2 \, h_{CaP} \, (V - E_{Ca})$$
$$I_{K(Ca)} = g_{K(Ca)} \, q_{K(Ca)} \, (V - E_K)$$
$$I_L = g_L \, (V - E_L)$$

### Ca²⁺ dynamics

$$\frac{d[Ca^{2+}]}{dt} = -f_{Ca} \, I_{CaP} - k_{decay} \, [Ca^{2+}]$$

### K(Ca) activation (Michaelis-Menten, ultra-sensitive)

$$q_{K(Ca),\infty} = \frac{[Ca^{2+}]}{[Ca^{2+}] + 0.0002}$$

Half-activation at 0.2 µM — 5× lower than ChayKeizer (1 µM). This
extreme sensitivity means even tiny Ca²⁺ changes modulate K(Ca).

### Boltzmann activations

| Function | Midpoint | Slope | Type |
|----------|----------|-------|------|
| m_Na,∞ | −35 mV | 7.5 mV | Activation |
| h_Na,∞ | −55 mV | 7.0 mV | Inactivation |
| n_K,∞ | −30 mV | 15 mV | Activation |
| m_CaP,∞ | −19 mV | 5.5 mV | Activation |
| h_CaP,∞ | −48 mV | 7.0 mV | Inactivation |

### Time constants

| Gate | τ formula | Range |
|------|-----------|-------|
| h_Na | 0.5 + 14/(1+exp((V+40)/12)) | 0.5–14.5 ms |
| n_K | 1 + 11/(1+exp((V+15)/8)) | 1–12 ms |
| m_CaP | 0.3 (constant) | 0.3 ms |
| h_CaP | 45 (constant) | 45 ms |
| q_KCa | 1.0 (constant) | 1 ms |

### 5 sub-steps per call

Forward Euler with 5 sub-steps (dt=0.01). Each call integrates 0.05 ms.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −68.0 | mV | Membrane potential |
| `h_na` | 0.8 | — | Na⁺ inactivation |
| `n_k` | 0.1 | — | K⁺ activation |
| `m_cap` | 0.0 | — | P-type Ca²⁺ activation |
| `h_cap` | 0.9 | — | P-type Ca²⁺ inactivation |
| `q_kca` | 0.0 | — | Ca²⁺-activated K⁺ activation |
| `ca` | 0.0001 | mM | Intracellular Ca²⁺ |
| `g_na` | 125.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 10.0 | mS/cm² | Delayed rectifier K⁺ |
| `g_cap` | 45.0 | mS/cm² | P-type Ca²⁺ conductance |
| `g_kca` | 35.0 | mS/cm² | Ca²⁺-activated K⁺ |
| `g_l` | 0.5 | mS/cm² | Leak |
| `e_na` | 45.0 | mV | Na⁺ reversal |
| `e_k` | −85.0 | mV | K⁺ reversal |
| `e_ca` | 135.0 | mV | Ca²⁺ reversal |
| `e_l` | −68.0 | mV | Leak reversal |
| `ca_decay` | 0.02 | ms⁻¹ | Ca²⁺ clearance rate |
| `f_ca` | 0.00024 | mM·cm²/(ms·mA) | Ca²⁺ influx coupling |
| `dt` | 0.01 | ms | Sub-step timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Conductance hierarchy

$$g_{Na} (125) \gg g_{CaP} (45) > g_{K(Ca)} (35) > g_K (10) > g_L (0.5)$$

Purkinje cells have the **highest Na⁺ conductance** in the library
(125 vs HH's 120). The P-type Ca²⁺ conductance (45) is substantial —
Purkinje cells are the most Ca²⁺-rich neurons in the brain.

### E_Ca = 135 mV — highest reversal

The Ca²⁺ reversal at +135 mV is extreme — reflecting the enormous
Ca²⁺ gradient (extracellular [Ca²⁺] ≈ 2 mM vs intracellular ≈ 0.1 µM).
This creates a very strong inward Ca²⁺ current that shapes the complex
spike waveform.

---

## Analytical Properties

### P-type Ca²⁺ current

The P-type (Purkinje-type) Ca²⁺ channel is unique to Purkinje cells:
- **Activation (m_CaP):** Half at −19 mV (suprathreshold) → activates
  during spikes. Fast τ = 0.3 ms.
- **Inactivation (h_CaP):** Half at −48 mV, slow τ = 45 ms →
  inactivates over multiple spike cycles.

The P/Q-type Ca²⁺ channel was discovered by Llinás et al. (1989) in
Purkinje cells — it is the dominant Ca²⁺ channel in cerebellar tissue.

### Complex spikes

The full De Schutter & Bower model (1142 compartments) produces complex
spikes — bursts of 2–5 spikelets triggered by climbing fibre input. The
simplified version captures the essential Ca²⁺/K(Ca) interaction:
1. Na⁺ spike → depolarisation
2. CaP activates → Ca²⁺ influx
3. Ca²⁺ activates K(Ca) → AHP
4. CaP inactivates (slow, 45 ms) → reduced Ca²⁺
5. K(Ca) deactivates → recovery

### Simple spikes

In the absence of climbing fibre input, Purkinje cells fire simple spikes
at 30–100 Hz. This is driven by the Na⁺/K⁺ interaction (similar to HH),
modulated by the tonic Ca²⁺/K(Ca) feedback.

### 7 state variables

The most complex neuron model in SC-NeuroCore by state variable count:
v, h_na, n_k, m_cap, h_cap, q_kca, ca = **7 variables.** The full
De Schutter & Bower model has 10 ion channels — this simplified version
captures the 5 most important.

---

## Behaviour

### Purkinje cell physiology

Cerebellar Purkinje cells are remarkable neurons:
- **Largest dendritic tree** in the mammalian brain (~200,000 synapses)
- **Only output** of the cerebellar cortex (inhibitory, to deep nuclei)
- **Highest spontaneous rate** in the brain (30–100 Hz simple spikes)
- **Complex spikes** from climbing fibre input (1–2 Hz)
- **Critical for motor learning** (LTD at parallel fibre → Purkinje synapses)

### Ca²⁺ dynamics

Ca²⁺ enters through P-type channels during spikes:
- Each spike: small Ca²⁺ increase (~0.001 mM)
- During burst: Ca²⁺ accumulates
- Between bursts: ca_decay (0.02 ms⁻¹) clears Ca²⁺
- K(Ca) half-activation at 0.2 µM — responds to very small [Ca²⁺]

---

## Comparison with Related Models

| Property | DeSchutter | HH | TraubMiles | Chay |
|----------|-----------|-----|-----------|------|
| Cell type | Purkinje | Squid axon | CA3 pyramid | Beta cell |
| State vars | 7 | 4 | 4 | 3 |
| Sub-steps | 5 | 100 | 10 | 1 |
| Ca²⁺ current | P-type (g=45) | None | None | Boltzmann (g=25) |
| K(Ca) | Yes (g=35) | No | No | Yes (g=12) |
| E_Ca | 135 mV | — | — | 100 mV |
| Speed | ~40K steps/s | ~670 steps/s | ~5K steps/s | ~200K steps/s |

Highest ionic current complexity in SC-NeuroCore: 5 currents, 7 state vars.

---

## Numerical Considerations

- **5 sub-steps:** dt=0.01ms, loop 5 → 0.05 ms per call. Sub-stepping is
  needed because the fast CaP channel (τ=0.3ms) requires small dt.
- **7 exp() per sub-step:** 5 Boltzmann + 2 tau functions = 35 exp() total
  per step() call.
- **Ca²⁺ clipped to ≥ 0:** Physical constraint maintained.
- **No V clipping:** Relies on conductance-based stability.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/de_schutter_purkinje.py` — 79 lines.
- **Seven state variables:** v, h_na, n_k, m_cap, h_cap, q_kca, ca.
- **Most state variables** of any model in SC-NeuroCore.
- **Dataclass:** Uses `@dataclass`.
- **Rust wiring:** Compatible (7 f64 state vars, sub-stepping).

---

## Infrastructure Pipeline

```
DeSchutterPurkinjeNeuron
├── step(current) → int {0, 1}
├── 5 sub-steps per call (dt=0.01ms, 0.05ms biological)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=10, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, isi, firing_rate verified
└── Rust: compatible (7 f64 state vars)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~40K steps/s | Not measured |
| Network (5 neurons, 1s) | ~4K neuron-steps/s | — |

Slow model — 5 sub-steps × 7 exp() = 35 exp() per call, plus Ca²⁺
dynamics. Second slowest after HH (600 exp()) by exp-count, but
faster per call (35 vs 600).

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 7-var evolution, finite 50k, reset |
| Purkinje | 4 | P-type Ca²⁺ current, K(Ca) activation, Ca²⁺ dynamics, complex spike |
| Dynamics | 3 | fires, subthreshold, rate monotonic |
| Parameters | 2 | dt stability, deterministic |
| Pipeline | 3 | Population, Network+drive, analysis |
| **Total** | **17** | |

See `tests/test_model_de_schutter_purkinje.py`. No bugs found.

---

## Findings

1. **7 state variables — most complex model:** v + 5 gates + Ca²⁺.

2. **P-type Ca²⁺ activates during spikes:** m_CaP midpoint at −19 mV
   (suprathreshold) → Ca²⁺ influx only during action potentials.

3. **K(Ca) ultra-sensitive:** Half-activation at 0.2 µM — responds to
   minimal Ca²⁺ changes.

4. **h_CaP slow (45 ms):** Slow inactivation of Ca²⁺ channel means
   Ca²⁺ current decreases gradually over multiple spike cycles.

5. **E_Ca = 135 mV — largest reversal potential:** Reflects the extreme
   Ca²⁺ gradient across the cell membrane.

6. **5 sub-steps adequate:** The fast CaP channel (τ=0.3ms) needs small
   dt, but 5 sub-steps at 0.01ms suffice for stability.

7. **g_Na = 125 — highest Na⁺ conductance:** Purkinje cells have among
   the highest Na⁺ channel densities in the brain.

8. **Network pipeline functional:** All standard components work despite
   the model's complexity.
