# WongWangUnit

**Module:** `sc_neurocore.neurons.models.wong_wang`
**Reference:** Wong & Wang, J. Neurosci. 26(4), 2006
**Family:** Reduced mean-field attractor model (decision-making)
**State variables:** `s1`, `s2` (NMDA synaptic gating variables for two competing pools)

---

## Equations

### Two-pool NMDA dynamics

$$\frac{ds_1}{dt} = -\frac{s_1}{\tau_s} + (1 - s_1) \gamma r_1$$

$$\frac{ds_2}{dt} = -\frac{s_2}{\tau_s} + (1 - s_2) \gamma r_2$$

### Synaptic currents (with cross-inhibition)

$$I_1 = J_N s_1 - J_{cross} s_2 + I_0 + \text{stim}_1 + \sigma \xi_1$$

$$I_2 = J_N s_2 - J_{cross} s_1 + I_0 + \text{stim}_2 + \sigma \xi_2$$

### Input-output transfer function (Abbott-Chance)

$$r = \Phi(I_{syn}) = \frac{aI_{syn} - b}{1 - \exp(-d(aI_{syn} - b))}$$

where $a = 270$ Hz/nA, $b = 108$ Hz, $d = 0.154$ s.

At the singularity ($aI - b \approx 0$): $\Phi \rightarrow 1/d \approx 6.5$ Hz.

### Output

Returns `(r1, r2)` — firing rates (Hz) of both pools. **Returns tuple of
floats, not binary spike.**

### Implementation

```python
def step(self, stim1=0.0, stim2=0.0) -> tuple[float, float]:
    i1 = j_n * s1 - j_cross * s2 + i_0 + stim1 + sigma * randn()
    i2 = j_n * s2 - j_cross * s1 + i_0 + stim2 + sigma * randn()
    r1, r2 = phi(i1), phi(i2)
    s1 += (-s1/tau_s + (1-s1)*gamma*r1) * dt
    s2 += (-s2/tau_s + (1-s2)*gamma*r2) * dt
    s1, s2 = clip([s1, s2], 0, 1)
    return (r1, r2)
```

Forward Euler, single step per call. Gaussian noise on each pool.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `s1` | 0.1 | — | NMDA gating variable, pool 1 |
| `s2` | 0.1 | — | NMDA gating variable, pool 2 |
| `tau_s` | 0.1 | s | NMDA decay time constant (100 ms) |
| `gamma` | 0.641 | — | Saturation factor (kinetic parameter) |
| `j_n` | 0.2609 | nA | Same-pool recurrent NMDA weight |
| `j_cross` | 0.0497 | nA | Cross-pool inhibitory weight |
| `i_0` | 0.3255 | nA | Background tonic current |
| `sigma` | 0.02 | nA | Noise standard deviation |
| `dt` | 0.001 | s | Integration timestep (1 ms) |

### Parameter derivation (Wong & Wang 2006)

The parameters are derived from a mean-field reduction of the Wang 2002
spiking model (1600 neurons → 2 ODEs). The key mappings:

- J_N = 0.2609 nA: effective NMDA self-coupling (from 240 exc → 240 exc
  projection with g_NMDA connectivity)
- J_cross = 0.0497 nA: cross-inhibition via shared inhibitory pool
  (exc → inh → opp. exc, reduced to effective negative coupling)
- I_0 = 0.3255 nA: background drive from non-selective neurons + external

The ratio J_N / J_cross ≈ 5.25 controls the strength of competition.

---

## Analytical Properties

### Transfer function Φ(I)

- **Monotonic:** Φ is strictly increasing for I > b/a
- **Low rate regime:** For I < I_0, Φ ≈ 0 (below activation threshold)
- **Singularity handling:** At aI - b = 0, L'Hôpital gives Φ = 1/d ≈ 6.5 Hz
- **Linear regime:** For moderate I, Φ ≈ (aI - b) (rate proportional to current)
- **High rate saturation:** Φ → aI - b for large I (denominator → 1)

### Attractor landscape

The system has three key fixed points (for equal stimuli):
1. **Spontaneous state:** s1 ≈ s2 ≈ 0.1 (both pools at low activity)
2. **Pool 1 wins:** s1 ≈ 0.7, s2 ≈ 0.1 (decision for option 1)
3. **Pool 2 wins:** s1 ≈ 0.1, s2 ≈ 0.7 (decision for option 2)

The spontaneous state is unstable — any asymmetry in stimuli drives the
system toward one of the two "winner" attractors via positive feedback
(NMDA recurrence) and negative feedback (cross-inhibition).

### Decision mechanism

1. Both pools start at s1 = s2 = 0.1 (spontaneous state)
2. Asymmetric stimuli (stim1 ≠ stim2) create a bias
3. NMDA recurrence amplifies the bias: the pool with more input increases
   its firing rate → more NMDA current → higher rate (positive feedback)
4. Cross-inhibition suppresses the losing pool: higher s_winner →
   J_cross × s_winner subtracted from loser's current
5. The system settles into a winner-take-all attractor

### Reaction time

The time to reach a decision (escape from the spontaneous state) depends on:
- Stimulus difference (stim1 − stim2): larger → faster
- Noise amplitude σ: larger → faster but less accurate
- This speed-accuracy tradeoff matches psychophysical data (Roitman &
  Shadlen 2002)

### NMDA time constant

τ_s = 0.1 s (100 ms) — much slower than AMPA (~5 ms). This slow dynamics
is critical for the attractor mechanism: it provides temporal integration
over the decision-relevant timescale (~500 ms).

### s bounded in [0, 1]

Both s1 and s2 are clipped to [0, 1] after each step. The NMDA gating
variable represents a fraction of open channels — physically bounded.

---

## Behaviour

### Decision-making with asymmetric input

With stim1 > stim2:
- s1 increases, s2 decreases
- After ~500 ms, s1 ≫ s2 → "decision for option 1"
- The decision is irreversible (attractor basin)

### Equal stimuli: noise-driven decision

With stim1 = stim2, noise (σ=0.02) breaks the symmetry:
- One pool randomly wins
- The outcome is probabilistic (~50/50 for equal stimuli)
- This models the random-dot motion discrimination task

### Noise amplitude effects

- σ = 0: deterministic, no decision possible with equal stimuli (stays
  at spontaneous state indefinitely)
- σ = 0.02: moderate noise, decisions in ~500 ms
- σ = 0.1: high noise, faster but less accurate decisions

### Stochastic dynamics

Two runs with the same parameters produce different outcomes because of
`np.random.randn()` noise. This is fundamental — the stochastic decision
dynamics are the model's core feature.

---

## Pipeline Compatibility

### Returns tuple, not int

**Critical limitation:** `step()` returns `(r1, r2)` — a tuple of two
firing rates. The SC-NeuroCore Network pipeline expects `step() → int`.

**Recommended use:** Standalone decision-making simulation. Track s1 and s2
over time, define a decision threshold (e.g., s > 0.6), and measure
reaction time and accuracy.

### Population compatible

`Population(WongWangUnit, n=10, label="ww")` works for construction.
Network simulation will produce incorrect results due to tuple return.

---

## Comparison with Related Models

| Property | Wong-Wang (reduced) | Wang 2002 (spiking) | Usher-McClelland |
|----------|--------------------|--------------------|------------------|
| Variables | 2 (s1, s2) | ~1600 neurons | 2 (x1, x2) |
| Type | Mean-field ODE | Spiking network | Rate ODE |
| NMDA | Explicit τ_s | Explicit channels | Implicit |
| Noise | Additive Gaussian | Synaptic noise | Diffusion |
| Speed | ~500K steps/s | ~10 neuron-steps/s | ~500K steps/s |
| Pipeline | Tuple return (limited) | Full spiking | Tuple return |

The Wong-Wang model achieves the same attractor dynamics as the 1600-neuron
Wang 2002 model with just 2 ODEs — a factor of 800× reduction in
computational cost.

---

## Numerical Considerations

- **dt = 0.001 s (1 ms):** Typical for mean-field decision models. The
  slow NMDA dynamics (τ_s = 100 ms) are well-resolved at this timestep.
- **Clipping:** s1, s2 clipped to [0, 1] after each step. This prevents
  numerical overshoot from Euler integration at the bounds.
- **Singularity in Φ:** The transfer function has a removable singularity
  at aI − b = 0. Handled with |x| < 1e-6 guard → returns 1/d ≈ 6.5 Hz.
- **Global RNG:** Uses `np.random.randn()` — not per-instance reproducible.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/wong_wang.py` — 61 lines.
- **Two state variables:** s1, s2 (NMDA gating fractions).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Two-argument step:** `step(stim1, stim2)` — different from other models
  which take a single current argument.
- **Rust wiring:** Not in the Rust NeuronVariant enum (tuple return,
  two-argument step, global RNG).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~200K steps/s | Not applicable |
| Network | Limited (tuple return) | — |

Fast model — 2 Euler updates, 2 Φ evaluations (2 exp() calls), 2 randn()
calls per step. No sub-stepping.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, tuple return, both s evolve, finite 100k, reset |
| Transfer function | 4 | Φ monotonic, singularity protection, low/high rate regimes |
| Decision dynamics | 5 | asymmetric input → winner, equal input noise-driven, s bounded, σ=0 no decision, reaction time |
| Parameters | 3 | dt stability, σ sweep, j_n/j_cross ratio |
| Pipeline | 2 | Population creates, tuple return documented |
| **Total** | **19** | |

See `tests/test_model_wong_wang.py`. No bugs found.

---

## Findings

1. **Asymmetric input drives decisions:** stim1 > stim2 consistently leads
   to s1 > s2 after sufficient time.

2. **Equal stimuli produce 50/50 decisions:** With noise (σ=0.02), both
   pools have equal probability of winning.

3. **Transfer function singularity handled:** Φ(I) at aI−b=0 returns
   1/d ≈ 6.5 Hz correctly.

4. **s bounded by clipping:** After 100,000 steps, s1 and s2 remain in
   [0, 1] despite Euler integration.

5. **σ=0 prevents spontaneous decisions:** Without noise, equal stimuli
   leave the system at the symmetric spontaneous state indefinitely.

6. **NMDA recurrence amplifies:** J_N × s provides the positive feedback
   that drives the winner-take-all dynamics.

7. **Cross-inhibition suppresses:** J_cross × s_opponent provides the
   negative feedback that shuts down the losing pool.

8. **Stochastic outcomes:** Two runs produce different decisions — this is
   the model's core feature, not a bug.

9. **J_N/J_cross ≈ 5.25:** The self/cross coupling ratio determines the
   strength of competition. Higher ratio → sharper decisions.

10. **Pipeline-limited:** Tuple return and two-argument step prevent
    standard Network integration. The model is designed for standalone
    decision-making simulations.
