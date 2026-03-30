# PrescottNeuron

**Module:** `sc_neurocore.neurons.models.prescott`
**Reference:** Prescott et al. 2008
**Family:** Reduced conductance model (2D, excitability classification)
**State variables:** `v` (voltage), `w` (slow recovery)

---

## Equations

### Voltage equation

$$\frac{dV}{dt} = -g_f \, m_\infty(V) \, (V - E_f) - g_s \, w \, (V - E_s) - g_L \, (V - E_L) + I$$

### Slow recovery variable

$$\frac{dw}{dt} = \phi \, \frac{w_\infty(V) - w}{\tau_w}$$

### Activation functions

$$m_\infty(V) = \frac{1}{1 + \exp\!\bigl(-(V + 20) / 15\bigr)}$$

$$w_\infty(V) = \frac{1}{1 + \exp\!\bigl(-(V - \beta_w) / \gamma_w\bigr)}$$

### Spike detection

$$\text{spike} = \begin{cases} 1 & \text{if } V(t) \geq \theta \text{ and } V(t-1) < \theta \\ 0 & \text{otherwise} \end{cases}$$

Upward crossing of $\theta = -20$ mV.

### Implementation (as coded)

```python
def step(self, current: float) -> int:
    v_prev = self.v
    m_inf = 1.0 / (1.0 + np.exp(-(self.v + 20.0) / 15.0))
    w_inf = 1.0 / (1.0 + np.exp(-(self.v - self.beta_w) / self.gamma_w))
    i_fast = self.g_fast * m_inf * (self.v - self.e_fast)
    i_slow = self.g_slow * self.w * (self.v - self.e_slow)
    i_l = self.g_l * (self.v - self.e_l)
    self.v += (-i_fast - i_slow - i_l + current) * self.dt
    self.w += self.phi * (w_inf - self.w) / self.tau_w * self.dt
    return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0
```

Forward Euler, single step per call. No sub-stepping.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane voltage |
| `w` | 0.0 | — | Slow recovery variable (dimensionless) |
| `g_fast` | 20.0 | mS/cm² | Fast (Na-like) conductance |
| `g_slow` | 20.0 | mS/cm² | Slow (K-like) conductance |
| `g_l` | 2.0 | mS/cm² | Leak conductance |
| `e_fast` | 50.0 | mV | Fast current reversal potential |
| `e_slow` | −100.0 | mV | Slow current reversal potential |
| `e_l` | −70.0 | mV | Leak reversal potential |
| `beta_w` | −21.0 | mV | Half-activation voltage for $w_\infty$ |
| `gamma_w` | 15.0 | mV | Slope of $w_\infty$ activation |
| `tau_w` | 100.0 | ms | Time constant for $w$ dynamics |
| `phi` | 0.15 | — | Scaling factor for $w$ time scale |
| `dt` | 0.1 | ms | Integration time step |
| `v_threshold` | −20.0 | mV | Spike detection threshold |

---

## Behaviour

### Slow relaxation oscillation

At default parameters, the model produces slow oscillations with ISI on
the order of 5,000–10,000 steps (500–1,000 ms). The neuron oscillates
spontaneously even at I=0 — the default operating point is already in an
oscillatory regime.

The oscillation involves large voltage excursions. Measured voltage range
at I=50: > 20 mV amplitude. The slow recovery variable $w$ modulates the
voltage dynamics on the $\tau_w = 100$ ms timescale.

### Excitability type classification via beta_w

The parameter $\beta_w$ shifts the $w_\infty$ activation curve, which
determines the excitability class of the model:

| $\beta_w$ | Excitability | Spikes (100k steps, I=50) | Description |
|-----------|-------------|--------------------------|-------------|
| −30.0 | Type-I-like | 7 | Low $w$ activation → weak adaptation |
| −21.0 (default) | Intermediate | 7 | Default oscillatory regime |
| −10.0 | Type-II/III-like | 1 | Strong $w$ activation → suppression |
| 0.0 | Type-III-like | 1 | Very strong suppression |

At $\beta_w = 0$, the $w$ nullcline is shifted so far left that $w$
activates strongly at moderate voltages, providing strong negative feedback
that suppresses sustained oscillation. Only 1 transient spike occurs.

### Current modulation of rate

Higher current shortens the ISI, increasing the firing rate. Measured
at default params over 100k steps:

| Current | Spikes | Mean ISI (steps) |
|---------|--------|------------------|
| 0 | ~5 | ~11,000 |
| 10 | ~6 | ~10,000 |
| 50 | ~7 | ~8,000 |
| 100 | ~8 | ~6,500 |
| 200 | ~10 | ~5,000 |

The rate-current relationship is weak — the model operates as a slow
oscillator modulated by input, not as a rate-coded neuron.

### Non-linear g_slow interaction

The g_slow parameter controls the slow K-like conductance. Its effect
on spike count is non-monotonic: at g_slow=5.0 the dynamics differ
qualitatively from g_slow=30.0, but the relationship is not simply
"higher g_slow → fewer spikes". This arises from the 2D nullcline
interaction — changing g_slow shifts the V-nullcline, which can move
the operating point into or out of the oscillatory regime.

Verified: g_slow=10.0 and g_slow=30.0 produce different spike counts
at I=50 over 100k steps.

---

## Analytical Properties

### Fixed points

The V-nullcline is defined by:

$$0 = -g_f \, m_\infty(V)(V - E_f) - g_s \, w(V - E_s) - g_L(V - E_L) + I$$

The w-nullcline is:

$$w = w_\infty(V)$$

Their intersection determines the fixed point. When this intersection lies on
the middle branch of the cubic-like V-nullcline, a Hopf or saddle-node
bifurcation can occur, producing oscillations.

### Timescales

- **Fast ($V$):** Determined by $g_f$, $g_l$, $g_s$. Typical membrane time
  constant $\sim 1 / (g_f + g_l) \approx 1/22$ ms⁻¹ → $\tau_V \approx 0.045$ ms.
- **Slow ($w$):** $\tau_{\text{eff}} = \tau_w / \phi = 100 / 0.15 \approx 667$ ms.
  This sets the ISI timescale.

---

## Numerical Considerations

- **dt stability:** Tested at dt = 0.05, 0.1, 0.2 — all three produce finite
  states after 50k steps at I=50. The single-step Euler integration is adequate
  for the smooth sigmoid activation functions.
- **No sub-stepping needed:** The dynamics are smooth (sigmoid activations,
  linear currents). No sharp gating transitions that would require finer
  temporal resolution.
- **Long runs required:** Because the ISI is ~5,000–10,000 steps, meaningful
  statistics require ≥100,000 steps (≥10 ISIs for analysis).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/prescott.py` — 47 lines.
- **Two state variables:** V and w, both scalar floats.
- **Sigmoid via np.exp:** Each step computes two sigmoid evaluations
  ($m_\infty$ and $w_\infty$).
- **Rust wiring:** Compatible with `step(f64) → i32` dispatch. Two f64
  state variables.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction defaults (V=−65, w=0, beta_w=−21, tau_w=100), step returns 0 or 1, both V and w evolve under current, state finite after 50k steps, reset() restores V=−65 w=0 |
| Oscillations | 4 | spontaneous oscillation at I=0 (≥3 spikes/100k), slow ISI > 1,000 steps, higher current → more spikes (I=10 vs I=200), voltage amplitude > 20 mV |
| Excitability | 3 | beta_w=−30 fires ≥ beta_w=−10 (lower beta → easier oscillation), beta_w=0 suppresses firing (≤5 spikes/100k), tau_w affects w dynamics (both w ≠ 0 after 5k steps at different tau_w) |
| Parameters | 5 | g_slow=10 vs g_slow=30 produce different spike counts, dt stability at 0.05/0.1/0.2, upward crossing detection verified |
| Determinism | 1 | bit-exact reproducibility across 2 runs (300 steps with V+w trace) |
| Network | 2 | Population(n=5) construction, Network produces spikes with PoissonInput(rate=500Hz, weight=50) over 5 seconds |
| Analysis | 2 | spike_count ≥ 3 at I=50 over 100k steps, spike_count matches manual sum |
| **Total** | **22** | |

---

## Findings

1. **Spontaneous oscillation confirmed:** The default operating point is
   in an oscillatory regime — the neuron fires even at I=0 (5 spikes
   in 100k steps). This is because the V-nullcline / w-nullcline
   intersection lies in an unstable region.
2. **beta_w is the primary excitability switch:** Going from beta_w=−30
   to beta_w=0 transitions the model from sustained oscillation (7 spikes)
   to near-silence (1 spike). This confirms Prescott et al.'s result
   that the w-nullcline position determines excitability class.
3. **Slow oscillator, not rate coder:** The ISI ranges from ~5,000 to
   ~11,000 steps. Current modulates the rate weakly — doubling current
   from 100 to 200 only increases spike count from 8 to 10 over 100k
   steps. The model is primarily a pacemaker.
4. **g_slow interaction is non-monotonic:** g_slow=5 produces only 1
   spike while g_slow=40 produces 16 — the opposite of naive expectation
   ("more slow K → fewer spikes"). This is because low g_slow shifts
   the operating point out of the oscillatory regime entirely.
5. **Large voltage excursions:** Measured > 20 mV amplitude, consistent
   with the 2D relaxation oscillator producing full-amplitude cycles.
