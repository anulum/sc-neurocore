# ChayNeuron

**Module:** `sc_neurocore.neurons.models.chay`
**Reference:** Chay, Biophys. J. 47(3), 1985
**Family:** Biophysical conductance-based (3-ODE, pancreatic beta-cell burster)
**State variables:** `v` (membrane potential), `n` (K⁺ delayed rectifier), `ca` (intracellular Ca²⁺)

---

## Equations

### Membrane potential

$$\frac{dV}{dt} = -I_{Ca} - I_K - I_{K(Ca)} - I_L + I$$

### Ionic currents

$$I_{Ca} = g_{Ca} \, m_\infty(V) \, (V - E_{Ca})$$
$$I_K = g_K \, n \, (V - E_K)$$
$$I_{K(Ca)} = g_{K(Ca)} \, \frac{[Ca^{2+}]}{[Ca^{2+}] + 1} \, (V - E_K)$$
$$I_L = g_L \, (V - E_L)$$

### Ca²⁺ dynamics

$$\frac{d[Ca^{2+}]}{dt} = \rho \left(-\alpha_{Ca} \, I_{Ca} - k_{Ca} \, [Ca^{2+}]\right)$$

### Steady-state activation

$$m_\infty(V) = \frac{1}{1 + \exp(-(V+25)/8)}$$
$$n_\infty(V) = \frac{1}{1 + \exp(-(V+18)/14)}$$

### K⁺ delayed rectifier

$$\frac{dn}{dt} = \frac{n_\infty(V) - n}{\tau_n(V)}, \quad \tau_n = \frac{1}{0.01 \, |V+18|}$$

### K(Ca) activation (Hill function)

$$\text{kca\_act} = \frac{[Ca^{2+}]}{[Ca^{2+}] + 1}$$

This is a Hill function with coefficient 1 (Michaelis-Menten form).
Half-activation at [Ca²⁺] = 1. At [Ca²⁺] = 0.1 (default): activation ≈ 0.091.

### Implementation

```python
def step(self, current: float) -> int:
    m_inf = 1 / (1 + exp(clip(-(v+25)/8, -500, 500)))
    n_inf = 1 / (1 + exp(clip(-(v+18)/14, -500, 500)))
    tau_n = 1 / (0.01 * max(|v+18|, 0.01))
    i_ca = g_ca * m_inf * (v - e_ca)
    kca_act = ca / (ca + 1)
    i_k = g_k * n * (v - e_k)
    i_kca = g_kca * kca_act * (v - e_k)
    i_l = g_l * (v - e_l)
    v += (-i_ca - i_k - i_kca - i_l + current) * dt
    v = clip(v, -200, 200)
    n += (n_inf - n) / max(tau_n, 0.01) * dt
    n = clip(n, 0, 1)
    ca = max(0, ca + rho * (-alpha_ca * i_ca - k_ca * ca) * dt)
    return 1 if crossing else 0
```

Forward Euler, single step per call. V clipped to [−200, 200]. n clipped
to [0, 1]. Ca clipped to ≥0.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −50.0 | mV | Membrane potential |
| `n` | 0.1 | — | K⁺ delayed rectifier gate |
| `ca` | 0.1 | µM | Intracellular Ca²⁺ concentration |
| `g_ca` | 25.0 | pS | Ca²⁺ conductance |
| `g_k` | 1400.0 | pS | K⁺ delayed rectifier conductance |
| `g_kca` | 12.0 | pS | Ca²⁺-activated K⁺ conductance |
| `g_l` | 7.0 | pS | Leak conductance |
| `e_ca` | 100.0 | mV | Ca²⁺ reversal |
| `e_k` | −75.0 | mV | K⁺ reversal |
| `e_l` | −40.0 | mV | Leak reversal |
| `rho` | 0.00015 | — | Ca²⁺ dynamics scaling |
| `alpha_ca` | 0.002 | — | Ca²⁺ influx coupling (from I_Ca) |
| `k_ca` | 0.04 | ms⁻¹ | Ca²⁺ clearance rate |
| `dt` | 0.02 | ms | Integration timestep |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

### Conductance hierarchy

$$g_K (1400) \gg g_{Ca} (25) > g_{K(Ca)} (12) > g_L (7)$$

The K⁺ delayed rectifier dominates by 56×. This ensures strong
repolarisation after each spike — characteristic of fast-spiking
pancreatic beta cells.

---

## Analytical Properties

### Bursting mechanism (slow Ca²⁺ modulation)

The 3-ODE system produces bursting via:

1. **Silent phase:** Low Ca²⁺ → weak K(Ca) → excitable
2. **Spike initiation:** Input triggers first spike → Ca²⁺ influx
3. **Active phase:** Rapid spiking, Ca²⁺ accumulates with each spike
4. **Burst termination:** High Ca²⁺ → strong K(Ca) → hyperpolarisation
   suppresses spiking
5. **Recovery:** Ca²⁺ decays (k_ca clearance) → K(Ca) weakens → excitable
6. **Cycle repeats**

### Ca²⁺ as slow variable

The Ca²⁺ dynamics are very slow due to ρ = 0.00015:
$$\frac{d[Ca]}{dt} = 0.00015 \times (\text{influx} - \text{clearance})$$

This makes Ca²⁺ the slowest variable — it modulates excitability on the
timescale of seconds (bursting period), while V and n operate on ms.

### K(Ca) channel as burst terminator

The Ca²⁺-activated K⁺ current $I_{K(Ca)} = g_{K(Ca)} \cdot [Ca]/(Ca+1)
\cdot (V - E_K)$ is always outward (hyperpolarising) when V > E_K.
As Ca²⁺ accumulates during a burst:
- kca_act increases: 0.091 (rest) → 0.5 (Ca=1) → 0.9 (Ca=9)
- I_K(Ca) grows → overwhelms excitatory drive → burst ends

### V and n clipping

Both V and n are explicitly clipped (V ∈ [−200, 200], n ∈ [0, 1]):
- V clipping prevents Euler divergence from the fast Ca²⁺ current
- n clipping ensures the gating variable stays physical

### Ca²⁺ clipping to ≥ 0

Ca²⁺ concentration cannot be negative. The `max(0, ...)` ensures physical
validity even during numerical overshoot.

### exp() clipping to [−500, 500]

The Boltzmann sigmoids clip the exp argument to prevent IEEE overflow.
This is conservative — exp(500) ≈ 1.4 × 10²¹⁷ would overflow float64.

---

## Behaviour

### Pancreatic beta-cell bursting

The Chay model was designed to explain the oscillatory electrical activity
of pancreatic beta cells, which exhibit:
- Bursts of action potentials (5–15 seconds active)
- Silent inter-burst intervals (5–20 seconds)
- Burst period controlled by glucose concentration (mapped to I_ext)

### Insulin secretion coupling

In real beta cells:
- Spikes open voltage-gated Ca²⁺ channels
- Ca²⁺ influx triggers insulin vesicle exocytosis
- Burst frequency encodes glucose concentration
- This is the fundamental mechanism of glucose-stimulated insulin secretion

The model's Ca²⁺ variable directly represents this physiological [Ca²⁺]_i.

### Input (glucose) controls regime

- Low I (low glucose): resting, no spikes
- Moderate I: bursting (periodic spiking/silence)
- High I (high glucose): continuous spiking (no bursting)

This matches the experimentally observed dose-response curve of beta cells.

---

## Comparison with Related Models

| Property | Chay | ChayKeizer | HindmarshRose | Yamada |
|----------|------|-----------|---------------|-------|
| ODEs | 3 | 3 | 3 | 3 |
| Slow var | Ca²⁺ (physiological) | Ca²⁺ | z (phenomenological) | q |
| K(Ca) | Yes (Hill) | Yes | No | No |
| Currents | Ca, K, K(Ca), L | Ca, K, K(Ca) | None explicit | Na, K, q, L |
| Cell type | Beta cell | Beta cell | Generic | Generic |
| Ca²⁺ influx | −α·I_Ca | −α·I_Ca | — | — |
| Clipping | V, n, Ca, exp | Similar | None | None |

Chay and ChayKeizer are sister models for beta cells. HindmarshRose and
Yamada produce similar bursting from different mathematical mechanisms.

---

## Numerical Considerations

- **Single Euler step:** dt=0.02ms. Small timestep needed because the
  Ca²⁺ current with g_ca=25 and E_ca=100 creates fast dynamics near
  the reversal potential.
- **5 clipping operations:** V [−200,200], n [0,1], Ca ≥0, 2× exp clip.
  These are essential for numerical robustness.
- **2 exp() per step:** m_inf and n_inf Boltzmann sigmoids.
- **ρ = 0.00015:** Very small scaling → Ca²⁺ changes slowly.
  dt/ρ = 133 → many steps per Ca²⁺ time constant.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/chay.py` — 55 lines.
- **Three state variables:** v, n, ca.
- **Dataclass:** Uses `@dataclass`.
- **Extensive clipping:** 5 clip operations for numerical robustness.
- **Rust wiring:** Compatible (3 f64 state vars, clip + exp).

---

## Infrastructure Pipeline

```
ChayNeuron
├── step(current) → int {0, 1}
├── 1 Euler step + 2 exp() per call (dt=0.02ms)
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

Moderate speed — 2 exp() + 5 clips per step, no sub-stepping.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 3-var evolution, finite 50k, reset |
| Ca²⁺ dynamics | 4 | Ca increases during spikes, Ca decays between bursts, Ca ≥ 0, rho scaling |
| K(Ca) | 3 | activation Hill function, burst termination via K(Ca), g_kca=0 no bursting |
| Bursting | 4 | produces bursts, inter-burst silence, input controls regime, burst period |
| Parameters | 3 | dt stability, g_kca sweep, deterministic |
| Pipeline | 4 | Population, Network+drive, Projection, analysis |
| **Total** | **23** | |

See `tests/test_model_chay.py`. No bugs found.

---

## Findings

1. **Bursting confirmed:** Alternating epochs of rapid spiking and silence,
   modulated by slow Ca²⁺ dynamics.

2. **Ca²⁺ accumulates during burst:** Each spike increases Ca²⁺ via
   −α_ca × I_Ca (I_Ca is negative/inward → −α × negative = positive Δ[Ca]).

3. **K(Ca) terminates burst:** High Ca²⁺ → kca_act → strong I_K(Ca) →
   hyperpolarisation → burst ends.

4. **Ca²⁺ clears between bursts:** k_ca × Ca decay brings Ca²⁺ back
   toward 0, restoring excitability for the next burst.

5. **g_kca=0 eliminates bursting:** Without K(Ca), Ca²⁺ has no effect
   on membrane → continuous spiking.

6. **Input controls regime:** Low I → rest, moderate I → burst, high I → tonic.

7. **V clipped to [−200, 200]:** Prevents Euler divergence from the
   large g_K=1400 conductance.

8. **Physiological Ca²⁺:** Unlike phenomenological slow variables (z in
   HindmarshRose), Ca²⁺ is a real measurable quantity — the model can
   be validated against fluorescent Ca²⁺ imaging data.
