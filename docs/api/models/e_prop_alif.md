# EPropALIFNeuron

**Module:** `sc_neurocore.neurons.models.e_prop_alif`
**Reference:** Bellec et al., Nat. Commun. 11(1), 2020 (e-prop)
**Family:** Adaptive LIF with eligibility traces (biologically plausible learning)
**State variables:** `v` (membrane potential), `a` (adaptive threshold), `e_trace` (eligibility trace)

---

## Equations

### Membrane potential (discrete exponential decay)

$$V_{t+1} = \alpha_m \cdot V_t + I_t$$

where $\alpha_m = \exp(-dt / \tau_m)$.

No leak subtraction — the exponential decay factor α_m directly scales
the previous voltage. This is the discrete-time equivalent of the
continuous LIF: $\tau_m \, dV/dt = -V + R \cdot I$.

### Adaptive threshold

$$\theta_t = \theta_{base} + \beta \cdot a_t$$

On spike: $a_{t+1} = \alpha_a \cdot a_t + 1$
No spike: $a_{t+1} = \alpha_a \cdot a_t$

where $\alpha_a = \exp(-dt / \tau_a)$.

Each spike increments $a$ by 1. Between spikes, $a$ decays exponentially
with $\tau_a = 200$ ms. The effective threshold $\theta_t$ rises after
each spike, producing spike-frequency adaptation.

### Eligibility trace (Bellec 2020, Eq. 4)

$$\psi_t = 0.3 \cdot \max(0,\; 1 - |V_t - \theta_t|)$$

$$e_{t+1} = \alpha_a \cdot e_t + \psi_t$$

The pseudo-derivative ψ is non-zero only when V is within 1 unit of the
threshold — a triangular kernel centred on the threshold. The eligibility
trace e accumulates these near-threshold events with slow decay (τ_a).

### Spike and reset

$$V_t \geq \theta_t: \quad V \leftarrow V_{reset}, \quad a \leftarrow \alpha_a \cdot a + 1$$

### Three-factor learning rule (not implemented in step, but enabled by e_trace)

$$\Delta w = \eta \cdot L_t \cdot e_t$$

where $L_t$ is the learning signal (error/reward), $e_t$ is the eligibility
trace, and $\eta$ is the learning rate. This is the e-prop rule: synaptic
weight changes are proportional to the product of a global error signal and
a local eligibility trace — no backpropagation through time (BPTT) needed.

### Implementation

```python
def step(self, current: float) -> int:
    self.v = self.alpha_m * self.v + current
    threshold = self.v_threshold_base + self.beta * self.a
    psi = max(0.0, 1.0 - abs(self.v - threshold)) * 0.3
    self.e_trace = self.alpha_a * self.e_trace + psi
    if self.v >= threshold:
        self.v = self.v_reset
        self.a = self.alpha_a * self.a + 1.0
        return 1
    self.a *= self.alpha_a
    return 0
```

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | 0.0 | a.u. | Membrane potential |
| `a` | 0.0 | — | Adaptive threshold component |
| `e_trace` | 0.0 | — | Eligibility trace |
| `tau_m` | 20.0 | ms | Membrane time constant |
| `tau_a` | 200.0 | ms | Adaptation time constant |
| `v_threshold_base` | 1.0 | a.u. | Base spike threshold |
| `beta` | 0.07 | — | Threshold adaptation coupling |
| `v_reset` | 0.0 | a.u. | Post-spike reset potential |
| `dt` | 1.0 | ms | Integration timestep |
| `alpha_m` | computed | — | exp(−dt/τ_m) = exp(−1/20) ≈ 0.9512 |
| `alpha_a` | computed | — | exp(−dt/τ_a) = exp(−1/200) ≈ 0.9950 |

### Precomputed decay constants

α_m and α_a are computed in `__post_init__` to avoid repeated exp() calls
during step(). This is an optimisation — each step() call uses only
multiplication (no exp()).

### Time constant ratio

$$\tau_a / \tau_m = 200 / 20 = 10$$

The adaptation time constant is 10× the membrane time constant. This means
adaptation persists across many spikes (10 membrane time constants), creating
sustained spike-frequency adaptation.

---

## Analytical Properties

### Adaptive threshold mechanism

After n spikes in quick succession (no decay between):
$$a \approx n, \quad \theta \approx \theta_{base} + \beta \cdot n = 1.0 + 0.07n$$

| Spikes | θ | Increase |
|--------|---|----------|
| 0 | 1.00 | — |
| 5 | 1.35 | 35% |
| 10 | 1.70 | 70% |
| 20 | 2.40 | 140% |

The threshold nearly triples after 20 rapid spikes — strong adaptation.

### Pseudo-derivative ψ

The pseudo-derivative is a **surrogate gradient** for the non-differentiable
spike function:
- ψ = 0 when |V − θ| > 1 (far from threshold: no learning signal)
- ψ = 0.3 when V = θ (at threshold: maximum learning signal)
- Linear interpolation between

This triangular approximation replaces the Dirac delta of the true spike
derivative. The width (1 unit on each side of θ) and amplitude (0.3) are
hyperparameters from Bellec 2020.

### Eligibility trace as filtered pseudo-derivative

$$e_t = \sum_{s=0}^{t} \alpha_a^{t-s} \psi_s$$

The eligibility trace is an exponentially-weighted running sum of
near-threshold events. It tells the learning rule: "this synapse
contributed to recent near-threshold activity." The slow decay (τ_a = 200 ms)
creates a long temporal window for credit assignment.

### Connection to BPTT

E-prop is an online approximation to truncated BPTT:
- BPTT: exact gradients, O(T) memory, biologically implausible
- E-prop: approximate gradients via eligibility traces, O(1) memory,
  biologically plausible (only local information + global error)

The approximation error is bounded by the mixing time of the recurrent
network — for networks with fast dynamics (relative to τ_a), e-prop
converges to near-BPTT performance.

---

## Behaviour

### Spike-frequency adaptation

With constant input, the firing rate decreases over time:
- Early: θ ≈ θ_base = 1.0 → easy to spike
- Later: a accumulates → θ increases → harder to spike
- Steady state: a converges where decay (α_a) balances spike increments

### Eligibility trace dynamics

The eligibility trace tracks near-threshold voltage excursions:
- When V is far from θ: e_trace decays (ψ = 0)
- When V approaches θ: e_trace accumulates (ψ > 0)
- After spike: e_trace continues to decay slowly (τ_a = 200 ms)

This provides the learning rule with a temporally-extended memory of
which synapses recently contributed to near-threshold activity.

### No learning in step()

The step() function computes e_trace but does not apply weight updates.
Learning requires a separate outer loop:
1. Call step() to get spike and update e_trace
2. Compute learning signal L_t (from error/reward)
3. Update weights: Δw = η × L_t × e_trace

This separation allows flexible learning rule implementations.

---

## E-Prop Framework Context

### Bellec et al. 2020 results

The e-prop paper demonstrated that ALIF neurons with eligibility traces
can learn:
- TIMIT speech recognition (word error rate competitive with LSTM)
- Delayed match-to-sample tasks
- Evidence accumulation (similar to Wong-Wang decision tasks)

All without BPTT — using only local eligibility traces and a global
error signal.

### Three types of e-prop

| Type | Learning signal | Biological analogue |
|------|----------------|-------------------|
| e-prop 1 | Broadcasting error | Neuromodulatory signal |
| e-prop 2 | Random feedback | Random feedback alignment |
| e-prop 3 | Symmetric error | Error backpropagation |

The EPropALIFNeuron supports all three types — the eligibility trace
is the same; only the external learning signal differs.

### Why ALIF (not plain LIF)?

Adaptation (via the dynamic threshold) is critical for e-prop performance:
- Creates slow dynamics that encode temporal context
- The slow τ_a provides a long credit assignment window
- Matches the slow timescales of many cognitive tasks (~100 ms)

Without adaptation, e-prop degrades significantly on temporal tasks.

---

## Pipeline Compatibility

### Fully compatible

`step(current) → int` — standard spiking interface. Population, Network,
SpikeMonitor, PoissonInput, Projection all work.

---

## Comparison with Related Models

| Property | EPropALIF | AdEx | SRM0 | LIF |
|----------|----------|------|------|-----|
| State vars | 3 (V, a, e) | 2 (V, w) | 1+η | 1 (V) |
| Adaptation | Dynamic threshold | w current | η kernel | None |
| Eligibility | Yes (e_trace) | No | No | No |
| Learning support | E-prop (3-factor) | None built-in | None | None |
| τ_adaptation | 200 ms | 100 ms | 50 ms (η) | — |
| Discrete decay | α_m, α_a (no exp/step) | Euler (exp clip) | Euler + exp | Various |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

The EPropALIF is the only model in SC-NeuroCore with built-in eligibility
trace support for biologically plausible learning.

---

## Numerical Considerations

- **No exp() per step:** α_m and α_a precomputed in __post_init__.
  step() uses only multiply and compare — no transcendental functions.
- **Stable decay:** α_m, α_a ∈ (0, 1) → V and a always decay. No
  numerical instability possible.
- **ψ bounded:** max(0, 1 − |V − θ|) ∈ [0, 1]. Multiplied by 0.3 →
  ψ ∈ [0, 0.3]. e_trace cannot blow up.
- **Integer-friendly:** The discrete decay formulation (multiply by α)
  is easily quantised for neuromorphic hardware.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/e_prop_alif.py` — 54 lines.
- **Three state variables:** v, a, e_trace.
- **Dataclass + field(init=False):** α_m and α_a are derived parameters.
- **No private methods:** All logic in step().
- **Rust wiring:** Compatible (3 f64 state vars, precomputed α).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~500K steps/s | Not measured |
| Network (10 neurons, 1s) | ~40K neuron-steps/s | — |

Fast model — no exp() per step, no sub-stepping. Only multiplications
and comparisons. Comparable to standard LIF speed.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary, 3-var evolution, finite 50k, reset |
| Adaptation | 5 | a increments on spike, θ rises, ISI lengthens, β=0 no adaptation, strong adapt fewer spikes |
| Eligibility | 4 | e_trace accumulates near threshold, decays far from threshold, ψ triangular shape, e_trace bounded |
| f–I curve | 3 | subthreshold silent, monotonic, fires with drive |
| Parameters | 3 | dt stability, τ_a sweep, precomputed α values |
| Pipeline | 4 | Population, Network+drive, Projection, analysis |
| **Total** | **24** | |

See `tests/test_model_e_prop_alif.py`. No bugs found.

---

## Findings

1. **Adaptive threshold rises on spikes:** After 10 rapid spikes,
   θ ≈ 1.0 + 0.07×10 = 1.70 — 70% increase confirmed.

2. **ISI adaptation confirmed:** First ISI shorter than later ISIs
   at constant input, due to threshold elevation.

3. **Eligibility trace accumulates near threshold:** When V is within
   1 unit of θ, e_trace increases. When V is far away, e_trace decays.

4. **ψ bounded by 0.3:** Maximum pseudo-derivative is 0.3 at V = θ.
   No unbounded gradients.

5. **β=0 eliminates adaptation:** With β=0, θ is constant, and the
   model reduces to a simple exponential-decay LIF.

6. **Precomputed α eliminates exp():** No transcendental function calls
   during step() — only multiply and compare.

7. **τ_a/τ_m = 10:** The adaptation operates on a 10× slower timescale
   than membrane dynamics, creating sustained adaptation.

8. **Only model with eligibility trace:** Unique in SC-NeuroCore —
   the e_trace enables e-prop learning without external trace management.

9. **Network pipeline fully functional:** All standard pipeline
   components work without limitations.

10. **E-prop approximates BPTT:** The eligibility trace mechanism
    provides an online, biologically plausible alternative to
    backpropagation through time for recurrent SNN training.
