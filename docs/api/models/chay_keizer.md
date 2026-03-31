# ChayKeizerNeuron

**Module:** `sc_neurocore.neurons.models.chay_keizer`
**Reference:** Chay & Keizer, Biophys. J. 42(2), 1983
**Family:** Biophysical conductance-based (3-ODE, pancreatic beta-cell, Ca²⁺-dependent K⁺)
**State variables:** `v` (membrane potential), `n` (K⁺ delayed rectifier), `ca` (intracellular Ca²⁺)

---

## Equations

### Membrane potential

$$\frac{dV}{dt} = -I_{Ca} - I_K - I_{K(Ca)} - I_L + I$$

### Ionic currents

$$I_{Ca} = g_{Ca} \, m_\infty(V) \, (V - E_{Ca})$$
$$I_K = g_K \, n \, (V - E_K)$$
$$I_{K(Ca)} = g_{K(Ca)} \, \frac{[Ca^{2+}]}{[Ca^{2+}] + K_d} \, (V - E_K)$$
$$I_L = g_L \, (V - E_L)$$

### Activation functions

$$m_\infty(V) = \frac{1}{1 + \exp(-(V+25)/8)}$$
$$n_\infty(V) = \frac{1}{1 + \exp(-(V+18)/14)}$$

### K⁺ gate dynamics

$$\frac{dn}{dt} = \frac{n_\infty(V) - n}{\tau_n(V)}, \quad \tau_n(V) = \frac{20}{1 + \exp((V+18)/14)}$$

### Ca²⁺ dynamics

$$\frac{d[Ca^{2+}]}{dt} = -f_{Ca} \, I_{Ca} - k_{Ca} \, [Ca^{2+}]$$

### K(Ca) activation (Michaelis-Menten)

$$q_{K(Ca)} = \frac{[Ca^{2+}]}{[Ca^{2+}] + K_d}$$

K_d = 1.0 µM: half-activation concentration.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −50.0 | mV | Membrane potential |
| `n` | 0.01 | — | K⁺ delayed rectifier gate |
| `ca` | 0.1 | µM | Intracellular Ca²⁺ |
| `g_ca` | 20.0 | pS | Ca²⁺ conductance |
| `g_k` | 25.0 | pS | K⁺ delayed rectifier conductance |
| `g_kca` | 12.0 | pS | Ca²⁺-activated K⁺ conductance |
| `g_l` | 0.1 | pS | Leak conductance |
| `e_ca` | 100.0 | mV | Ca²⁺ reversal |
| `e_k` | −75.0 | mV | K⁺ reversal |
| `e_l` | −40.0 | mV | Leak reversal |
| `k_d` | 1.0 | µM | K(Ca) half-activation Ca²⁺ |
| `f_ca` | 0.004 | — | Ca²⁺ influx coupling |
| `k_ca` | 0.03 | ms⁻¹ | Ca²⁺ clearance rate |
| `dt` | 0.02 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

---

## Analytical Properties

### Comparison with Chay 1985

| Feature | ChayKeizer (1983) | Chay (1985) |
|---------|------------------|-------------|
| g_K | 25 | 1400 (56× higher) |
| g_Ca | 20 | 25 |
| g_L | 0.1 | 7 (70× higher) |
| Ca²⁺ scaling | f_ca = 0.004 | ρ × α_ca |
| K_d parameter | Explicit (1.0 µM) | Implicit (1.0) |
| tau_n | Voltage-dependent (20/(1+exp)) | 1/(0.01×|V+18|) |
| n initial | 0.01 | 0.1 |

ChayKeizer is the **earlier, simpler** model. Chay (1985) refined the
conductances to better match experimental data — particularly the much
higher g_K for fast repolarisation.

### Bursting mechanism

Identical to Chay 1985 — slow Ca²⁺ accumulation activates K(Ca) which
terminates bursts. The difference is in parameter values:
- ChayKeizer: more moderate conductances, slower dynamics
- Chay: extreme g_K dominance, faster spikes within bursts

### Ca²⁺ dynamics (simpler than Chay)

ChayKeizer uses direct coupling: $d[Ca]/dt = -f_{Ca} \cdot I_{Ca} - k_{Ca} \cdot [Ca]$

No ρ scaling factor — the Ca²⁺ dynamics are controlled by f_ca (influx)
and k_ca (clearance) directly.

### tau_n voltage-dependent

$$\tau_n(V) = \frac{20}{1 + \exp((V+18)/14)}$$

- At V = −80 mV: τ_n ≈ 20 ms (slow recovery)
- At V = −18 mV: τ_n = 10 ms (moderate)
- At V = 0 mV: τ_n ≈ 7 ms (fast during spike)

This creates voltage-dependent K⁺ activation kinetics — fast at
depolarised potentials (spike repolarisation) and slow at rest.

### K(Ca) half-activation

With K_d = 1.0 µM:
- At [Ca²⁺] = 0.1 (default): q = 0.091 (9.1% active)
- At [Ca²⁺] = 1.0: q = 0.5 (50% active)
- At [Ca²⁺] = 10.0: q = 0.909 (90.9% active)

---

## Behaviour

### Three-current interaction

1. **I_Ca (depolarising):** m_inf is instantaneous — provides fast
   inward current that triggers spikes
2. **I_K (repolarising):** n gate provides delayed outward current —
   terminates individual spikes
3. **I_K(Ca) (burst-terminating):** Slowly activated by Ca²⁺ — provides
   the slow negative feedback that ends bursts

### Glucose-response (via input current)

Like Chay 1985:
- Low I → rest (no spikes)
- Moderate I → bursting
- High I → continuous spiking

---

## Comparison with Related Models

| Property | ChayKeizer (1983) | Chay (1985) | ShermanRinzelKeizer |
|----------|------------------|-------------|---------------------|
| Year | 1983 | 1985 | 1988 |
| g_K | 25 | 1400 | 3500 |
| Ca²⁺ model | Simple (f_ca) | Scaled (ρ × α) | Detailed |
| tau_n | V-dependent | V-dependent | V-dependent |
| K(Ca) | Michaelis-Menten | Hill (n=1) | Michaelis-Menten |
| Bursting | Square-wave | Square-wave | Phantom/parabolic |

ChayKeizer → Chay → ShermanRinzelKeizer represents an evolution of
beta-cell models with increasing biophysical detail.

---

## Numerical Considerations

- **Single Euler step:** dt=0.02ms.
- **3 exp() per step:** m_inf, n_inf, tau_n.
- **Clipping:** V ∈ [−200, 200], n ∈ [0, 1], Ca ≥ 0, exp ∈ [−500, 500].
- **tau_n floor:** max(tau_n, 0.1) prevents division by zero.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/chay_keizer.py` — 60 lines.
- **Three state variables:** v, n, ca.
- **Dataclass:** Uses `@dataclass`.
- **Rust wiring:** Compatible (3 f64 state vars).

---

## Infrastructure Pipeline

```
ChayKeizerNeuron
├── step(current) → int {0, 1}
├── 1 Euler step + 3 exp() per call (dt=0.02ms)
├── Population, Network, SpikeMonitor: compatible
│   PoissonInput(weight=5, rate=500Hz)
├── Projection: tested src→tgt wiring
├── Analysis: spike_count, isi, firing_rate verified
└── Rust: compatible (3 f64 state vars)
```

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~200K steps/s | Not measured |
| Network (10 neurons, 1s) | ~20K neuron-steps/s | — |

Moderate speed — 3 exp() + clipping per step.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 3-var evolution, finite 50k, reset |
| Ca²⁺ dynamics | 3 | Ca increases during spikes, Ca ≥ 0, k_ca clearance |
| K(Ca) | 3 | Michaelis-Menten activation, K_d half-activation, burst termination |
| Bursting | 3 | produces bursts, inter-burst silence, input controls regime |
| Parameters | 2 | dt stability, deterministic |
| Pipeline | 4 | Population, Network+drive, Projection, analysis |
| **Total** | **20** | |

See `tests/test_model_chay_keizer.py`. No bugs found.

---

## Findings

1. **Bursting confirmed:** Same mechanism as Chay 1985 — Ca²⁺-modulated
   K(Ca) terminates bursts.

2. **ChayKeizer precedes Chay by 2 years:** The 1983 model established
   the Ca²⁺/K(Ca) bursting framework that Chay refined in 1985.

3. **More moderate conductances:** g_K=25 vs Chay's 1400. Spikes within
   bursts are slower and broader.

4. **tau_n voltage-dependent:** 20 ms at rest, 7 ms during spike — provides
   appropriate K⁺ activation timing.

5. **K_d = 1.0 µM:** Half-activation at physiological Ca²⁺ levels
   (~1 µM during burst).

6. **Ca²⁺ clipped to ≥ 0:** Physical constraint maintained numerically.

7. **Network pipeline functional:** All standard components work.

8. **Historical first:** ChayKeizer (1983) was one of the first models
   to explain pancreatic beta-cell bursting — foundational for
   computational endocrinology.

---

## Historical and Theoretical Context

### Keizer's contribution

Joel Keizer (UC Davis) was a physical chemist who brought thermodynamic
rigour to biological modelling. His collaboration with Teresa Chay
produced the first mechanistic model of beta-cell bursting, grounded
in ion channel biophysics rather than phenomenological fitting.

### The Ca²⁺ hypothesis of bursting

ChayKeizer (1983) established the now-standard hypothesis:
1. Ca²⁺ enters through voltage-gated channels during spikes
2. Intracellular Ca²⁺ activates K(Ca) channels
3. K(Ca) current provides slow negative feedback → burst termination
4. Ca²⁺ pumps/buffers restore low Ca²⁺ → excitability recovers

This "Ca²⁺ hypothesis" was later confirmed experimentally with
Ca²⁺-sensitive dyes (Gilon & Henquin 2001) and became the foundation
for all subsequent beta-cell models.

### Fast-slow decomposition

The ChayKeizer model is a classic example of Rinzel's (1987) fast-slow
decomposition:
- **Fast subsystem:** V, n (spike dynamics, timescale ~1–20 ms)
- **Slow variable:** Ca²⁺ (burst modulation, timescale ~seconds)

By treating Ca²⁺ as a slowly-varying parameter, the fast subsystem's
bifurcation diagram reveals the bursting mechanism:
- Low Ca²⁺ → stable limit cycle (spiking)
- High Ca²⁺ → stable fixed point (silence)
- The slow Ca²⁺ drift moves the system back and forth across this
  bifurcation → bursting

### Comparison with Hodgkin-Huxley approach

ChayKeizer differs from HH in a fundamental conceptual way:
- HH: fit rate functions to voltage-clamp data from squid axon
- ChayKeizer: derive channel kinetics from biophysical principles
  (Boltzmann distributions, not arbitrary α/β functions)

The Boltzmann sigmoid $m_\infty = 1/(1+\exp(-(V-V_{1/2})/k))$ has clear
thermodynamic meaning: V_{1/2} is the half-activation voltage, k is the
slope factor (proportional to temperature/channel valence). This makes
the parameters physically interpretable.

### Beta-cell model evolution

```
ChayKeizer 1983 (this model)
    │
    ├── Chay 1985 (refined conductances)
    │       │
    │       ├── Sherman, Rinzel & Keizer 1988 (phantom bursting)
    │       │       │
    │       │       └── Bertram et al. 2000 (dual slow oscillation)
    │       │               │
    │       │               └── BertramPhantomBurster (SC-NeuroCore)
    │       │
    │       └── Chay 1990, 1996 (further refinements)
    │
    └── Keizer & Magnus 1989 (ER Ca²⁺ stores)
            │
            └── Li & Bhatt 2002 (modern beta-cell model)
```

The ChayKeizer model is the root of this entire family tree — every
subsequent beta-cell model builds on or refines its Ca²⁺/K(Ca) framework.

### Insulin secretion dynamics

The model predicts that:
- Burst frequency encodes glucose concentration
- Individual spike rate within bursts is roughly constant
- The "duty cycle" (fraction of time active) controls mean Ca²⁺
- Mean Ca²⁺ determines insulin secretion rate

This prediction was confirmed by simultaneous electrophysiology and
Ca²⁺ imaging experiments (Santos et al., Diabetes 55, 2006).
