# ExpIFNeuron

**Module:** `sc_neurocore.neurons.models.expif`
**Reference:** Fourcaud-Trocmé, Hansel, van Vreeswijk & Brunel, J. Neurosci. 23(37), 2003
**Family:** Integrate-and-fire (exponential spike initiation)
**State variables:** `v` (membrane potential)

---

## Equations

### Membrane potential with exponential spike initiation

$$\tau \frac{dV}{dt} = -(V - V_{rest}) + \Delta_T \exp\!\left(\frac{V - V_{rh}}{\Delta_T}\right) + I$$

### Spike and reset

$$V \geq V_{threshold}: \quad V \leftarrow V_{reset}$$

### Implementation

```python
def step(self, current: float) -> int:
    exp_term = self.delta_t * np.exp(np.clip((self.v - self.v_rh) / self.delta_t, -20, 20))
    dv = (-(self.v - self.v_rest) + exp_term + current) / self.tau * self.dt
    self.v += dv
    if self.v >= self.v_threshold:
        self.v = self.v_reset
        return 1
    return 0
```

Forward Euler, single step. The exp() argument is clipped to [−20, 20]
to prevent IEEE overflow. This is the **Exponential Integrate-and-Fire**
(EIF) — the AdEx without the adaptation current w.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −65.0 | mV | Membrane potential |
| `v_rest` | −65.0 | mV | Resting potential |
| `v_reset` | −68.0 | mV | Post-spike reset |
| `v_threshold` | −50.0 | mV | Hard spike threshold |
| `v_rh` | −55.0 | mV | Rheobase voltage (exp centre) |
| `delta_t` | 2.0 | mV | Spike sharpness (slope factor) |
| `tau` | 20.0 | ms | Membrane time constant |
| `dt` | 0.1 | ms | Integration timestep |

### Δ_T = 2.0 mV (sharpness)

Controls how sharply the exponential term kicks in:
- Δ_T → 0: approaches hard threshold (LIF limit)
- Δ_T = 2 mV: moderate sharpness (default, biologically realistic)
- Δ_T > 5 mV: very soft onset

At V = V_rh: exp_term = Δ_T × exp(0) = Δ_T = 2.0 mV.
At V = V_rh + Δ_T: exp_term = Δ_T × e ≈ 5.44 mV.
At V = V_rh + 5Δ_T: exp_term = Δ_T × e⁵ ≈ 296 mV (explosive).

### V_rh = −55 mV (rheobase voltage)

The voltage at which the exponential term equals Δ_T. Below V_rh, the
exponential is small (< Δ_T) and the dynamics are LIF-like. Above V_rh,
the exponential grows rapidly → spike initiation.

The gap V_threshold − V_rh = −50 − (−55) = 5 mV = 2.5 × Δ_T. This means
the hard threshold is 2.5 Δ_T above the exponential centre — the exp
term is exp(2.5) ≈ 12.2 at threshold.

---

## Analytical Properties

### Exponential spike initiation

The exp term models the Na⁺ channel activation:
- Below V_rh: subthreshold, leak dominates
- Near V_rh: exponential growth begins (positive feedback)
- Above V_rh: exponential runaway → V shoots up → hits threshold

This is more realistic than LIF's hard threshold: real action potentials
show a smooth, exponential onset (Naundorf et al. 2006).

### Relationship to AdEx

The AdEx (Adaptive Exponential IF) is:
$$\tau \dot{V} = -(V-V_r) + \Delta_T \exp(...) - w + I$$
$$\tau_w \dot{w} = a(V-V_r) - w$$

The ExpIF is AdEx with **w ≡ 0** — no adaptation. This means:
- No spike-frequency adaptation
- No subthreshold adaptation
- Simpler (1 ODE vs 2)
- Identical spike initiation dynamics

### Steady-state voltage (subthreshold)

For constant I, setting dV/dt = 0:
$$0 = -(V_{ss} - V_r) + \Delta_T \exp\!\left(\frac{V_{ss} - V_{rh}}{\Delta_T}\right) + I$$

This is a transcendental equation — no closed-form solution. For V_ss
well below V_rh (exp ≈ 0): V_ss ≈ V_rest + I (LIF approximation).

### Rheobase current

The minimum constant current for spiking. At the bifurcation:
$$I_{rh} = (V_{rh} - V_{rest}) - \Delta_T$$

With defaults: I_rh = (−55 − (−65)) − 2 = 8 mV. Below I = 8, the
neuron is subthreshold (at steady state). Above I = 8, the exponential
term drives V to threshold.

### Exp clipping

The clip to [−20, 20] bounds the exp:
- exp(−20) ≈ 2 × 10⁻⁹ (negligible, safe)
- exp(20) ≈ 4.9 × 10⁸ (large but finite, prevents inf)

Without clipping, V far above V_rh would produce exp(huge) = inf → NaN.

---

## Behaviour

### Subthreshold at low current

Verified: at small I, zero spikes in 5000 steps. The leak term
(−(V−V_rest)) dominates the exponential, keeping V below threshold.

### Spiking at moderate current

Verified: at sufficient I, the exponential term drives V above
threshold. The neuron fires and resets.

### Monotonic f-I curve

Higher current → more spikes. Verified across multiple current values.

### ISI regularity

With constant input, the ISI is approximately constant (no adaptation
to create ISI lengthening). The ExpIF produces regular spiking.

### Exponential escape at V_rh

At V = V_rh, exp_term = Δ_T = 2.0. This term is additive to the leak:
total drive = −(V_rh − V_rest) + Δ_T + I = −(−55 − (−65)) + 2 + I = −8 + I.
For I > 8: net depolarisation → V rises → exp grows → spike.

---

## Comparison with Related Models

| Property | ExpIF | AdEx | LIF | EscapeRate |
|----------|-------|------|-----|-----------|
| State vars | 1 (V) | 2 (V, w) | 1 (V) | 1 (V) |
| Spike initiation | Exponential | Exponential | Hard threshold | Stochastic |
| Adaptation | None | w current | None | None |
| Exp per step | 1 | 1 | 0 | 1 (safe_exp) |
| Sharpness | Δ_T | Δ_T | — | Δu |
| V_rh | Yes | Yes | — | V_threshold |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

ExpIF is the **simplest exponential-onset model** — one parameter
(Δ_T) adds biophysical spike initiation to the LIF.

---

## Numerical Considerations

- **1 exp() per step:** Clipped to [−20, 20] → no overflow.
- **Single Euler step:** dt=0.1 ms. Adequate for τ=20 ms.
- **No sub-stepping:** The exponential term can cause V to overshoot
  the threshold by a large amount in a single step. The hard threshold
  check (V ≥ V_threshold) catches this.
- **V_reset < V_rest:** The reset to −68 mV (below rest −65) creates a
  brief hyperpolarisation after each spike — a simple refractory effect.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/expif.py` — 39 lines.
- **One state variable:** v.
- **Dataclass:** Uses `@dataclass`.
- **Simplest exponential model** in SC-NeuroCore (AdEx adds w).
- **Rust wiring:** Compatible (1 f64 state var, 1 exp).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~220K steps/s | Not measured |
| Network | Pipeline verified | — |

Fast — single exp + clip per step, no sub-stepping, 1 state variable.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction, binary output, state evolves, state finite (50K), reset |
| Exponential | 5 | exp term at V_rh, exp drives spike, Δ_T controls sharpness, exp clipping [−20,20], negative extreme finite |
| Analytical | 2 | membrane equation 1-step, subthreshold V approaches rest |
| F-I | 4 | subthreshold silent, suprathreshold fires, monotonic f-I, ISI regularity |
| Parameters | 4 | τ affects rate, custom V_rh, dt stability [0.05,0.1,0.2] (parametrised), deterministic |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection wiring, analysis pipeline |
| **Total** | **28** | **ALL PASSED (7.07s)** |

See `tests/test_model_expif.py`.

---

## Findings (Measured 2026-03-31)

1. **28/28 tests PASSED in 7.07s.** No failures.

2. **Exp term at V_rh = Δ_T.** At V = V_rh = −55, exp_term = 2.0 (exact).

3. **Exp drives spike.** Near V_rh, the exponential term provides enough
   positive feedback to push V to threshold.

4. **Δ_T controls sharpness.** Different Δ_T values produce different
   spike initiation dynamics.

5. **Exp clipping works.** At extreme V, exp argument clipped to ±20.
   No overflow, no NaN.

6. **Negative extreme finite.** At V << V_rh, exp term ≈ 0. V stays
   finite and near V_rest.

7. **Membrane equation verified.** 1-step dV matches analytical to
   machine precision.

8. **Subthreshold V → V_rest.** Without input, V decays to rest.

9. **Subthreshold silent.** Low current → zero spikes.

10. **Suprathreshold fires.** Sufficient current → spikes.

11. **Monotonic f-I.** More current → more spikes.

12. **ISI regular.** No adaptation → approximately constant ISI.

13. **τ affects rate.** Different τ → different spike rates.

14. **dt stability.** dt=0.05, 0.1, 0.2 all produce finite V.

15. **Network pipeline functional.** Population, Projection,
    PoissonInput, analysis all work.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
28/28 PASSED in 7.07s
├── TestExpIFIsolation: 5 tests
├── TestExpIFExponentialEscape: 5 tests
├── TestExpIFAnalytical: 2 tests
├── TestExpIFFI: 4 tests
├── TestExpIFParameters: 4 tests
├── TestExpIFPerformance: 2 tests
└── TestExpIFPipeline: 4 tests
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-65, τ=20, Δ_T=2 |
| step() → int {0,1} | ✓ PASS | Hard threshold at -50 |
| Exp term | ✓ PASS | Δ_T at V_rh, drives spike |
| Exp clipping | ✓ PASS | [−20, 20], no overflow |
| V evolves | ✓ PASS | Euler integration |
| Subthreshold | ✓ PASS | V → V_rest |
| Spiking | ✓ PASS | Above rheobase |
| Monotonic f-I | ✓ PASS | More I → more spikes |
| ISI regular | ✓ PASS | No adaptation |
| State finite | ✓ PASS | 50K steps |
| reset() | ✓ PASS | v → V_rest |
| Deterministic | ✓ PASS | Bit-exact |
| Population | ✓ PASS | Instances |
| Network | ✓ PASS | Spikes |
| Projection | ✓ PASS | Wiring |
| Analysis | ✓ PASS | spike_count, ISI, firing_rate |

**ALL 28 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**

---

## Theoretical Context

### Fourcaud-Trocmé et al. 2003

The EIF model was introduced as a **one-parameter extension** of the LIF
that captures the smooth spike initiation dynamics observed in cortical
neurons. The key result: a single parameter (Δ_T) bridges the gap between
the discontinuous LIF threshold and the smooth Hodgkin-Huxley spike onset.

### Why exponential?

The Na⁺ channel activation m∞(V) is a Boltzmann sigmoid:
$$m_\infty(V) = \frac{1}{1 + \exp(-(V-V_{1/2})/k)}$$

Near the activation midpoint, this is approximately exponential:
$$m_\infty(V) \approx \exp((V-V_{1/2})/k)$$

The EIF's exponential term directly captures this near-threshold
Na⁺ activation — it is not an arbitrary choice but a biophysically
motivated approximation.

### Parameter fitting

Δ_T and V_rh can be estimated from experimental data:
- **From f-I curve:** Fit the rheobase and slope near onset
- **From voltage traces:** Measure the spike onset sharpness
- **From HH model:** Linearise the Na⁺ current near threshold

Typical values from cortical pyramidal cells:
- Δ_T ≈ 1–3 mV (sharp onset)
- V_rh ≈ −50 to −55 mV

### EIF in the literature

The EIF has become the standard "intermediate complexity" model:
- More realistic than LIF (smooth onset)
- Simpler than AdEx (no adaptation)
- Used as the base model in Brette & Gerstner (2005) AdEx derivation
- Default model in Brian2, NEST, and PyNN simulators
- >2000 citations for the Fourcaud-Trocmé 2003 paper

### Noise sensitivity

The EIF's smooth threshold makes it **less sensitive to input noise**
than the LIF near threshold. The exponential ramp provides a "grace
period" where small voltage fluctuations are absorbed by the nonlinear
dynamics rather than producing premature or missed spikes. This makes
the EIF more robust in noisy network simulations.
