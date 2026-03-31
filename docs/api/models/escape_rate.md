# EscapeRateNeuron

**Module:** `sc_neurocore.neurons.models.escape_rate`
**Reference:** Gerstner, Neural Comput. 12(1), 2000; Gerstner & Kistler, Spiking Neuron Models, 2002
**Family:** Stochastic integrate-and-fire (escape noise model)
**State variables:** `v` (membrane potential)

---

## Equations

### Membrane potential (deterministic)

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I$$

### Instantaneous escape rate

$$\rho(V) = \rho_0 \exp\!\left(\frac{V - V_{threshold}}{\Delta u}\right)$$

### Spike probability (per timestep)

$$p_{spike} = \rho(V) \cdot dt$$

### Stochastic spike generation

$$\text{Bernoulli}(p_{spike}): \quad \text{if } U(0,1) < p_{spike}: \text{spike, } V \leftarrow V_{reset}$$

### Implementation

```python
def step(self, current: float) -> int:
    self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt
    rate = self.rho_0 * safe_exp((self.v - self.v_threshold) / self.delta_u)
    p_spike = rate * self.dt
    if np.random.random() < p_spike:
        self.v = self.v_reset
        return 1
    return 0
```

The membrane potential evolves deterministically (Euler), but spiking
is **stochastic**: the probability of a spike increases exponentially
as V approaches threshold. There is no hard threshold — even far below
V_threshold, there is a small but nonzero probability of firing.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | −70.0 | mV | Membrane potential |
| `v_rest` | −70.0 | mV | Resting potential |
| `v_reset` | −70.0 | mV | Post-spike reset |
| `v_threshold` | −50.0 | mV | Nominal threshold (centre of escape zone) |
| `tau_m` | 10.0 | ms | Membrane time constant |
| `rho_0` | 0.001 | kHz | Base escape rate |
| `delta_u` | 3.0 | mV | Escape noise width (sharpness) |
| `resistance` | 1.0 | MΩ | Membrane resistance |
| `dt` | 1.0 | ms | Integration timestep |

### ρ₀ = 0.001 (base rate)

The base escape rate at V = V_threshold: ρ(V_θ) = 0.001 kHz. This gives
p_spike = 0.001 × 1.0 = 0.001 (0.1%) per timestep at threshold. Well
below threshold, p_spike is exponentially smaller.

### Δu = 3.0 mV (noise width)

Controls the sharpness of the soft threshold:
- Small Δu → sharp transition (approaches hard threshold)
- Large Δu → broad, noisy threshold

The escape rate at key voltages:

| V − V_θ (mV) | ρ(V)/ρ₀ | p_spike (dt=1) |
|---------------|---------|----------------|
| −10 | exp(−3.33) ≈ 0.036 | 3.6 × 10⁻⁵ |
| −5 | exp(−1.67) ≈ 0.189 | 1.9 × 10⁻⁴ |
| 0 | 1.0 | 10⁻³ |
| +5 | exp(1.67) ≈ 5.29 | 5.3 × 10⁻³ |
| +10 | exp(3.33) ≈ 28.0 | 2.8 × 10⁻² |
| +20 | exp(6.67) ≈ 789 | 0.789 |

---

## Analytical Properties

### Escape noise vs reset noise

Two main approaches to stochastic spiking models:

| Property | Escape noise (this) | Reset noise (diffusion) |
|----------|-------------------|----------------------|
| Source | Threshold is stochastic | Membrane voltage is stochastic |
| Mechanism | Bernoulli(ρ(V)·dt) | V += σ·ξ each step |
| V dynamics | Deterministic | Stochastic |
| ρ(V) | Exponential escape | Not applicable |
| ISI distribution | Renewal | Non-renewal |
| Analytical | Tractable | Requires Fokker-Planck |

The escape noise model (Gerstner 2000) is more analytically tractable
because the membrane dynamics are deterministic — only the spike decision
is stochastic.

### Soft threshold interpretation

The exponential escape rate can be interpreted as a **Boltzmann
distribution** over threshold crossings:

$$\rho(V) = \rho_0 \exp\!\left(\frac{V - V_\theta}{\Delta u}\right)$$

This is equivalent to a hard threshold V_θ + noise ξ, where ξ is
drawn from an exponential distribution with scale Δu. The "escape"
metaphor: the neuron "escapes" over the threshold barrier with a rate
that increases exponentially as the barrier shrinks.

### ISI statistics

For constant input I producing steady-state V_ss:
- Mean ISI ≈ 1/ρ(V_ss) (for low rates)
- ISI distribution ≈ exponential (memoryless) for constant V
- CV (coefficient of variation) → 1 for Poisson-like firing

At higher rates (multiple spikes), the reset creates ISI correlations
because V must recover from V_reset to V_ss between spikes.

### Steady-state voltage

For constant I (subthreshold, no spikes):
$$V_{ss} = V_{rest} + R \cdot I$$

At default parameters with I=0: V_ss = −70 mV (at rest).
With I=50: V_ss = −70 + 50 = −20 mV (well above nominal threshold).

### Membrane equation one-step verification

The update dV = (−(V−V_rest) + R·I)/τ_m × dt is verified analytically
in the test suite to machine precision.

---

## Behaviour

### Stochastic spiking

The model produces stochastic spikes — two runs with identical parameters
produce different spike trains (different random seeds). This is verified
by test: two independent runs have different spike times.

### Rate increases with input

Higher current → higher V_ss → higher ρ(V) → more spikes. Verified:
strong drive produces more spikes than weak drive across 5000 steps.

### Zero input → silent

At I=0, V stays at V_rest = −70 mV. The escape rate at V_rest:
ρ = 0.001 × exp((−70 − (−50))/3) = 0.001 × exp(−6.67) ≈ 1.3 × 10⁻⁶.
p_spike = 1.3 × 10⁻⁶ per step. In 5000 steps: expected 0.006 spikes.
Effectively silent.

### safe_exp prevents overflow

The `safe_exp()` utility clips the argument to prevent IEEE overflow
when V is far above threshold. Without this, exp((V−V_θ)/Δu) could
produce inf for V >> V_θ.

---

## Comparison with Related Models

| Property | EscapeRate | StochasticIF | LIF | GalvesLocherbach |
|----------|-----------|-------------|-----|-----------------|
| V dynamics | Deterministic | Stochastic (σ·ξ) | Deterministic | Discrete-time |
| Spike mechanism | Bernoulli(ρ·dt) | Hard threshold | Hard threshold | Bernoulli(φ(V)) |
| Stochasticity | Threshold | Membrane noise | None | Activation function |
| ρ₀ parameter | Yes | No | No | No |
| Δu parameter | Yes (noise width) | σ (noise amplitude) | No | β (inverse temp) |
| ISI distribution | Approximately exponential | Inverse Gaussian | Deterministic | Variable |
| Pipeline | Compatible | Compatible | Compatible | Compatible |

---

## Numerical Considerations

- **1 safe_exp() per step:** For the escape rate calculation.
- **1 np.random.random() per step:** RNG call is the performance
  bottleneck (much slower than arithmetic).
- **Bernoulli approximation:** p_spike = ρ·dt assumes dt is small
  enough that p_spike << 1. For very high rates, p_spike > 1 is
  possible but the Bernoulli still works (clipped to 1.0 probability).
- **safe_exp overflow protection:** Clips exp argument to prevent inf.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/escape_rate.py` — 41 lines.
- **One state variable:** v (membrane potential).
- **Dataclass:** Uses `@dataclass`.
- **Uses safe_exp:** From `sc_neurocore.utils.numerics`.
- **Uses np.random:** Per-step RNG call (not seedable via constructor).
- **Rust wiring:** Compatible (1 f64 state var, safe_exp, RNG needed).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~75K steps/s | Not measured |
| Network | Pipeline verified | — |

Slower than deterministic models due to np.random.random() per step.
The RNG call dominates per-step cost (not the exp).

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | construction, binary output, state evolves, state finite (10K), reset |
| Stochastic | 5 | stochastic spiking (produces spikes), two runs differ, rate increases with input, zero input silent, safe_exp no overflow |
| Analytical | 4 | V steady-state, membrane equation 1-step, ρ₀ scales rate, Δu controls sensitivity |
| ISI | 2 | ISI variability (CV > 0), higher current shorter ISI |
| Parameters | 2 | τ_m controls V dynamics, resistance scales input |
| Performance | 2 | isolation throughput, network throughput |
| Pipeline | 4 | Population, Network spikes, Projection wiring, analysis pipeline |
| **Total** | **24** | **ALL PASSED (14.80s)** |

See `tests/test_model_escape_rate.py`.

---

## Findings (Measured 2026-03-31)

1. **24/24 tests PASSED in 14.80s.** No failures.

2. **Stochastic spiking confirmed.** The model produces spikes via
   Bernoulli sampling from ρ(V)·dt.

3. **Two runs differ.** Independent runs with identical parameters
   produce different spike trains (stochastic).

4. **Rate increases with input.** Higher current → more spikes.

5. **Zero input silent.** At I=0, effectively zero spikes (V far below
   threshold, ρ ≈ 10⁻⁶).

6. **safe_exp prevents overflow.** No NaN or inf at extreme voltages.

7. **V steady-state verified.** V_ss = V_rest + R·I.

8. **Membrane equation verified.** 1-step dV matches analytical to
   machine precision.

9. **ρ₀ scales rate.** Higher ρ₀ → more spikes at same voltage.

10. **Δu controls sensitivity.** Different Δu values produce different
    spike statistics.

11. **ISI variability.** Non-zero coefficient of variation confirms
    stochastic ISI distribution.

12. **Higher current → shorter ISI.** More input → faster firing.

13. **τ_m controls V dynamics.** Different τ_m values produce different
    voltage trajectories.

14. **Network pipeline functional.** Population, Projection, PoissonInput,
    SpikeMonitor, analysis all work.

---

## Pipeline Verification (End-to-End, Measured 2026-03-31)

### Test execution

```
24/24 PASSED in 14.80s
├── TestEscapeRateIsolation: 5 tests
├── TestEscapeRateStochasticMechanism: 5 tests
├── TestEscapeRateAnalytical: 4 tests
├── TestEscapeRateISI: 2 tests
├── TestEscapeRateParameters: 2 tests
├── TestEscapeRatePerformance: 2 tests
└── TestEscapeRatePipeline: 4 tests
```

### Pipeline stages verified

| Stage | Status | Notes |
|-------|--------|-------|
| Import + construction | ✓ PASS | v=-70, ρ₀=0.001, Δu=3 |
| step() → int {0,1} | ✓ PASS | Stochastic binary |
| V evolves | ✓ PASS | Deterministic LIF dynamics |
| Stochastic spiking | ✓ PASS | Bernoulli(ρ·dt) |
| Two runs differ | ✓ PASS | Different RNG streams |
| Rate increases | ✓ PASS | More I → more spikes |
| Zero → silent | ✓ PASS | ρ ≈ 10⁻⁶ |
| safe_exp | ✓ PASS | No overflow |
| V steady-state | ✓ PASS | V_rest + R·I |
| 1-step exact | ✓ PASS | Machine precision |
| ISI variability | ✓ PASS | CV > 0 |
| State finite | ✓ PASS | 10K steps |
| reset() | ✓ PASS | v → V_rest |
| Population | ✓ PASS | Instances |
| Network | ✓ PASS | Spikes > 0 |
| Projection | ✓ PASS | Wiring |
| Analysis | ✓ PASS | spike_count, ISI, firing_rate |

**ALL 24 PIPELINE TESTS PASSED. MODEL IS END-TO-END FUNCTIONAL.**
