# AvRonCardiacNeuron

**Module:** `sc_neurocore.neurons.models.av_ron_cardiac`
**Reference:** Av-Ron, Parnas & Segel, Biol. Cybern. 69(2), 1993
**Family:** Biophysical conductance-based (4-ODE, cardiac ganglion Type III burster)
**State variables:** `v` (membrane potential), `h` (Na⁺ inactivation), `n` (K⁺ activation), `s` (slow inactivation)

---

## Equations

### Membrane potential

$$\frac{dV}{dt} = -I_{Na} - I_K - I_s - I_L + I$$

### Ionic currents

$$I_{Na} = g_{Na} \, m_\infty^3 \, h \, (V - E_{Na})$$
$$I_K = g_K \, n^4 \, (V - E_K)$$
$$I_s = g_s \, s \, (V - E_s)$$
$$I_L = g_L \, (V - E_L)$$

### Boltzmann activation/inactivation

$$m_\infty = \frac{1}{1 + \exp(-(V+40)/7)}$$
$$h_\infty = \frac{1}{1 + \exp((V+45)/5)}$$
$$n_\infty = \frac{1}{1 + \exp(-(V+40)/15)}$$
$$s_\infty = \frac{1}{1 + \exp((V+35)/3)}$$

Note: h_inf and s_inf use **positive** exponent sign (inactivation —
decreasing with depolarisation).

### Voltage-dependent time constants

$$\tau_h = 1 + \frac{12}{1 + \exp((V+50)/8)} \quad \text{(1–13 ms)}$$
$$\tau_n = 1 + \frac{8}{1 + \exp((V+35)/8)} \quad \text{(1–9 ms)}$$
$$\tau_s = 200 + \frac{1000}{1 + \exp((V+30)/5)} \quad \text{(200–1200 ms)}$$

### Four timescales

| Variable | τ range | Role |
|----------|---------|------|
| V (m_inf) | Instantaneous | Spike upstroke |
| h | 1–13 ms | Na⁺ inactivation (spike repolarisation) |
| n | 1–9 ms | K⁺ activation (spike repolarisation) |
| s | 200–1200 ms | Slow inactivation (plateau/burst modulation) |

The 100× gap between h/n (~10 ms) and s (~1000 ms) creates the
two-timescale separation needed for plateau bursting.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −60.0 | mV | Membrane potential |
| `h` | 0.6 | — | Na⁺ inactivation gate |
| `n` | 0.3 | — | K⁺ activation gate |
| `s` | 0.5 | — | Slow inactivation gate |
| `g_na` | 80.0 | mS/cm² | Na⁺ conductance |
| `g_k` | 40.0 | mS/cm² | K⁺ conductance |
| `g_s` | 20.0 | mS/cm² | Slow current conductance |
| `g_l` | 0.1 | mS/cm² | Leak conductance |
| `e_na` | 40.0 | mV | Na⁺ reversal |
| `e_k` | −80.0 | mV | K⁺ reversal |
| `e_s` | −25.0 | mV | Slow current reversal |
| `e_l` | −60.0 | mV | Leak reversal (= V_rest) |
| `dt` | 0.02 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Key parameter: E_s = −25 mV

The slow current reversal is between rest (−60) and threshold (−20).
This means I_s can be either:
- **Depolarising** (when V < −25): drives V upward (plateau sustaining)
- **Hyperpolarising** (when V > −25): drives V downward (spike limiting)

This intermediate reversal creates the **plateau potential** characteristic
of cardiac ganglion neurons.

---

## Analytical Properties

### Type III bursting (plateau bursting)

Unlike Ca²⁺-dependent bursting (Chay, ChayKeizer), the AvRon model uses
**slow inactivation** to modulate a plateau potential:

1. **Plateau onset:** Input depolarises V past threshold → Na⁺ spike →
   V enters plateau region (V ≈ −25 mV, near E_s)
2. **Plateau sustained:** I_s provides depolarising current (V < E_s) →
   maintains elevated V between spikes
3. **Rapid spiking:** V oscillates between plateau and spike peak
4. **Slow inactivation:** s decreases slowly (τ_s ≈ 200–1200 ms) →
   I_s weakens → plateau collapses
5. **Recovery:** In silence, s recovers slowly → excitability restored
6. **Cycle repeats**

### s_inf is an inactivation (decreasing with V)

$$s_\infty = \frac{1}{1 + \exp((V+35)/3)}$$

- At V = −60 (rest): s_inf ≈ 1.0 (fully available)
- At V = −35: s_inf = 0.5 (half-inactivated)
- At V = −20 (plateau): s_inf ≈ 0.007 (nearly fully inactivated)

The steep slope (k=3 mV) means s inactivates almost completely during
the plateau — this is what terminates the burst.

### m_inf instantaneous

Na⁺ activation m is treated as instantaneous (m_inf from V each step),
reducing the model from 5 to 4 ODEs.

### Conductance ratios

$$g_{Na} : g_K : g_s : g_L = 80 : 40 : 20 : 0.1 = 800 : 400 : 200 : 1$$

The 2:1 Na⁺/K⁺ ratio (vs HH's 3.3:1) with substantial slow conductance
(g_s/g_Na = 0.25) creates the plateau dynamics.

---

## Behaviour

### Plateau bursting waveform

The characteristic cardiac ganglion waveform:
1. Brief depolarisation phase (Na⁺ spike)
2. Sustained plateau at ~−25 mV (I_s maintains)
3. Multiple spikes riding the plateau
4. Plateau collapse (s inactivation)
5. Hyperpolarisation and recovery

This differs from square-wave bursting (ChayKeizer): the inter-spike
interval during the burst is modulated by the plateau level.

### Cardiac ganglion context

The model was developed for **crustacean cardiac ganglion** (CG) neurons:
- CG neurons control the heartbeat in lobsters and crabs
- They produce rhythmic bursts that drive cardiac muscle contraction
- The plateau phase determines burst duration → contraction strength
- The inter-burst interval determines heart rate

### Input modulation

- Low I: subthreshold (no bursts)
- Moderate I: plateau bursting (rhythmic)
- High I: continuous spiking (no plateau termination)

---

## Comparison with Related Models

| Property | AvRon | Yamada | Chay | HindmarshRose |
|----------|-------|-------|------|---------------|
| ODEs | 4 | 3 | 3 | 3 |
| Burst type | Plateau (Type III) | Square-wave (Hopf) | Square-wave (Ca²⁺) | Square-wave |
| Slow var | s (inactivation) | q (activation) | Ca²⁺ | z |
| E_slow | −25 (intermediate) | −80 (hyperpol.) | −75 (hyperpol.) | — |
| Plateau | Yes | No | No | No |
| Cell type | Cardiac ganglion | Generic | Beta cell | Generic |

The AvRon model is unique in producing **plateau bursting** — the
sustained depolarised phase is absent in square-wave bursters.

---

## Numerical Considerations

- **Single Euler step:** dt=0.02ms. Adequate for the 4 time constants.
- **7 exp() per step:** m_inf, h_inf, n_inf, s_inf, τ_h, τ_n, τ_s.
- **No clipping:** Gates evolve within [0,1] by the Boltzmann dynamics.
  V is not clipped.
- **Stiffness:** The 100× gap between fast (h,n: ~10ms) and slow (s: ~1000ms)
  timescales creates moderate stiffness, but Euler handles it at dt=0.02.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/av_ron_cardiac.py` — 61 lines.
- **Four state variables:** v, h, n, s.
- **Dataclass:** Uses `@dataclass`.
- **Inline computation:** All 7 Boltzmann evaluations in step().
- **Rust wiring:** Compatible (4 f64 state vars, 7 exp calls).

---

## Infrastructure Pipeline

```
AvRonCardiacNeuron
├── step(current) → int {0, 1}
├── 1 Euler step + 7 exp() per call (dt=0.02ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=5, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, isi, firing_rate verified
└── Rust: compatible (4 f64 state vars)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~100K steps/s | Not measured |
| Network (10 neurons, 1s) | ~10K neuron-steps/s | — |

Moderate speed — 7 exp() per step, no sub-stepping. The multiple
Boltzmann evaluations are the dominant cost.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 4-var evolution, finite 50k, reset |
| Boltzmann | 3 | m_inf/h_inf/n_inf midpoints, s_inf steep inactivation, tau_s slow |
| Plateau | 3 | plateau potential near E_s, s inactivation during plateau, plateau termination |
| Dynamics | 4 | fires, subthreshold, rate monotonic, bursting pattern |
| Parameters | 2 | dt stability, deterministic |
| Pipeline | 3 | Population, Network+drive, analysis |
| **Total** | **20** | |

See `tests/test_model_av_ron_cardiac.py`. No bugs found.

---

## Findings

1. **Plateau bursting confirmed:** V sustains an elevated plateau near
   E_s = −25 mV between spikes, distinct from square-wave bursting.

2. **s inactivation terminates plateau:** s_inf ≈ 0.007 at V = −20 →
   s decays slowly during plateau → I_s weakens → plateau collapses.

3. **s recovery in silence:** After burst, V ≈ −60 → s_inf ≈ 1.0 →
   s recovers (τ_s ≈ 200–1200 ms) → excitability restored.

4. **E_s = −25 creates dual role:** Depolarising below −25 mV (sustains
   plateau), hyperpolarising above (limits spike peak).

5. **7 Boltzmann evaluations per step:** 4 activation functions + 3 time
   constants. More expensive than simpler models but no sub-stepping.

6. **100× timescale gap:** h/n operate at 1–13 ms while s operates at
   200–1200 ms. This separation is the structural requirement for
   plateau bursting.

7. **Cardiac rhythm application:** Burst duration maps to contraction
   strength, inter-burst interval maps to heart rate.

8. **Network pipeline functional:** All standard components work.

---

## Biological Context

### Crustacean cardiac ganglion

The cardiac ganglion (CG) is a small network of ~9 neurons in the lobster
heart that generates rhythmic bursts driving cardiac muscle contraction.
It is one of the best-characterised central pattern generators:
- 4 large motor neurons: produce bursts → contract the heart
- 5 small pacemaker neurons: generate rhythm → entrain motor neurons
- The AvRon model captures the motor neuron dynamics

### Plateau potentials in vertebrates

Plateau bursting is not restricted to invertebrates. Similar dynamics
occur in:
- **Thalamocortical relay neurons:** T-type Ca²⁺ plateau during sleep
- **Motoneurons:** Persistent inward currents create plateaus
  (Heckman & Enoka 2012)
- **Cardiac Purkinje fibres:** Long plateaus (~200 ms) drive the
  cardiac action potential

### Bursting classification

In Izhikevich's (2000) taxonomy, the AvRon model implements **fold/fold
cycle** (Type III) bursting:
- Active phase begins via fold bifurcation (saddle-node)
- Active phase ends via fold of cycles bifurcation
- The plateau provides the intermediate stable state between rest and
  full spiking

This differs from:
- Type I (fold/homoclinic): square-wave bursting (ChayKeizer)
- Type II (circle/fold cycle): parabolic bursting
- Type IV (subcritical Hopf): Yamada-type

### Neuromodulation

In the biological CG, neuromodulators (serotonin, dopamine, octopamine)
adjust burst parameters by modifying conductances:
- Serotonin: increases g_s → longer plateaus → stronger contractions
- Dopamine: decreases g_K → higher excitability → faster heart rate
- The model parameters (g_na, g_k, g_s) directly correspond to these
  pharmacological targets
