# SigmoidRateNeuron

**Module:** `sc_neurocore.neurons.models.sigmoid_rate`
**Reference:** Wilson & Cowan, Biophys. J. 12(1), 1972 (general framework)
**Family:** Rate model (continuous, single-unit)
**State variables:** `r` (firing rate)

---

## Equations

### Rate dynamics

$$\tau \frac{dr}{dt} = -r + \sigma(\beta(I - \theta))$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the logistic sigmoid.

### Sigmoid transfer function

$$\sigma(\beta(I - \theta)) = \frac{1}{1 + \exp(-\beta(I - \theta))}$$

- **θ** (theta): Activation threshold (sigmoid midpoint)
- **β** (beta): Gain (steepness). Higher β → sharper transition
- **Range:** (0, 1) for all finite inputs

### Implementation

```python
def step(self, current: float) -> float:
    sigma = 1.0 / (1.0 + np.exp(-self.beta * (current - self.theta)))
    self.r += (-self.r + sigma) / self.tau * self.dt
    return self.r
```

Forward Euler, single step per call. **Returns float (rate), not binary spike.**

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `r` | 0.0 | — | Firing rate (initial) |
| `tau` | 10.0 | ms | Rate time constant |
| `beta` | 1.0 | — | Sigmoid gain (steepness) |
| `theta` | 0.0 | — | Sigmoid threshold (midpoint) |
| `dt` | 0.1 | ms | Integration timestep |

---

## Analytical Properties

### Steady-state rate

Setting $dr/dt = 0$:

$$r_{ss} = \sigma(\beta(I - \theta)) = \frac{1}{1 + \exp(-\beta(I - \theta))}$$

The steady state is just the sigmoid evaluated at the current input.
The time constant τ only controls how fast r approaches r_ss, not the
final value.

### At threshold (I = θ)

$$r_{ss}(\theta) = \sigma(0) = 0.5$$

The rate is exactly 0.5 at the threshold. Verified by test.

### Gain controls steepness

- β = 0.1: very gradual transition (r goes from 0.27 to 0.73 over I ∈ [−10, 10])
- β = 1.0: moderate transition (default)
- β = 10.0: nearly binary on/off (step function in the limit β → ∞)

### Time constant controls response speed

- τ = 1.0: fast approach to steady state (~5 dt to converge)
- τ = 10.0: moderate (default, ~50 dt)
- τ = 100.0: slow (sluggish response, ~500 dt)

The exponential approach: $r(t) = r_{ss}(1 - e^{-t/\tau})$ from r=0.

### Rate bounded in [0, 1]

Since σ ∈ (0, 1) and r follows $dr/dt = (-r + σ)/τ$:
- If r > 1: dr/dt = (-r + σ)/τ < 0 → r decreases
- If r < 0: dr/dt = (-r + σ)/τ > 0 → r increases

The interval [0, 1] is positively invariant. Verified: r stays bounded
after 100,000 steps at any tested input.

### Low input → rate ≈ 0

For I ≪ θ: σ ≈ 0 → r_ss ≈ 0. The neuron is "off".

### High input → rate ≈ 1

For I ≫ θ: σ ≈ 1 → r_ss ≈ 1. The neuron is "on" at maximum rate.

---

## Behaviour

### Simplest possible rate model

The SigmoidRateNeuron is the minimal rate model in SC-NeuroCore:
- 1 state variable (r)
- 1 ODE (first-order linear)
- 1 nonlinearity (sigmoid)
- No E/I interaction (unlike Wilson-Cowan)
- No adaptation (unlike AdEx rate equivalent)
- No noise (unlike StochasticIF)

It serves as the building block for larger rate networks — each node
is a SigmoidRateNeuron, and connectivity is implemented via the input
current.

### Response to step input

Starting from r=0, applying I > θ:
1. σ(β(I−θ)) > 0.5 → dr/dt > 0 → r increases
2. r exponentially approaches σ with time constant τ
3. After ~3τ: r within 5% of steady state
4. After ~5τ: r essentially converged

### Response to oscillatory input

When driven by sinusoidal input I(t) = A·sin(ωt) + θ:
- At low ω (slow input): r tracks the sigmoid of I (rate-following)
- At high ω (fast input): r averages over the oscillation (low-pass filter)
- Cutoff frequency: ω_c ≈ 1/τ

The model acts as a first-order low-pass filter with sigmoid nonlinearity.

---

## Pipeline Compatibility

### Returns float, not int

**Limitation:** `step()` returns `float` (rate r), not `int` (spike).
When placed in a Network, any r > 0 registers as a "spike" — semantically
incorrect for a rate model.

**Recommended use:** Rate-based network simulations where the output
is interpreted as a continuous firing rate, not as individual spikes.

### Population compatible

`Population(SigmoidRateNeuron, n=10, label="sr")` works for construction.

---

## Comparison with Related Models

| Property | SigmoidRate | WilsonCowan | LIF | ThresholdLinearRate |
|----------|-----------|-------------|-----|-------------------|
| Variables | 1 (r) | 2 (E, I) | 1 (V) | 1 (r) |
| Type | Rate | Rate | Spiking | Rate |
| Nonlinearity | Sigmoid | Sigmoid | Hard threshold | ReLU |
| Output | float | float | int | float |
| E/I | No | Yes | No | No |
| Complexity | Minimal | Moderate | Minimal | Minimal |

The SigmoidRateNeuron is the single-unit building block of Wilson-Cowan:
the WC excitatory equation $\tau_E dE/dt = -E + S(...)$ is exactly this
model with S = sigmoid and the input being a weighted combination of E, I,
and external drive.

---

## Numerical Considerations

- **Single Euler step:** No sub-stepping. The linear ODE is unconditionally
  stable for dt < 2τ.
- **dt/τ ratio:** With defaults, dt/τ = 0.01 — well within stability.
  For dt/τ > 2: Euler oscillation. For dt/τ > 1: overshoot possible.
- **Sigmoid overflow:** For β(I−θ) < −700: exp → ∞, σ → 0 (safe).
  For β(I−θ) > 700: exp → 0, σ → 1 (safe). No overflow at float64.
- **No stiffness:** The single-variable linear system has one eigenvalue
  (-1/τ), always stable.

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/sigmoid_rate.py` — 34 lines.
- **One state variable:** r (firing rate).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Simplest model in the library:** 34 lines total, 3 lines of step() logic.
- **Rust wiring:** Compatible but pipeline-limited (float return).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~500K steps/s | Not applicable |
| Network | Limited (float return) | — |

Fastest model in the library — single Euler step, 1 exp() call, no
sub-stepping, no complex logic. The 500K steps/s is dominated by Python
interpreter overhead.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 4 | defaults, float return, r evolves, reset |
| Sigmoid | 4 | σ(θ)=0.5, monotonic, bounded (0,1), β controls steepness |
| Dynamics | 4 | steady state convergence, τ controls speed, high input r→1, low input r→0 |
| Parameters | 3 | dt stability, β sweep, τ sweep |
| Pipeline | 2 | Population creates, float return documented |
| **Total** | **17** | |

See `tests/test_model_sigmoid_rate.py`. No bugs found.

---

## Findings

1. **σ(θ) = 0.5 exact:** The sigmoid midpoint equals the threshold
   parameter to machine precision.

2. **Rate bounded in [0, 1]:** After 100,000 steps at any input, r
   remains in the bounded interval. The positive invariance property holds.

3. **Steady state = σ(input):** At convergence, r equals the sigmoid
   evaluated at the current — confirmed by measuring |r − σ| < 0.001
   after sufficient steps.

4. **τ controls convergence speed:** Larger τ → slower approach to
   steady state, verified by comparing r after 100 steps with τ=1 vs τ=100.

5. **β controls gain:** Higher β → sharper sigmoid transition, verified
   by measuring r at θ±1 for different β values.

6. **Simplest model:** 34 lines total — the minimal possible rate neuron.
   Serves as a pedagogical reference and building block for rate networks.

7. **Low-pass filter behaviour:** The model acts as a first-order
   low-pass filter with cutoff ≈ 1/τ on the input current, followed by
   sigmoid nonlinearity.

8. **Float return limitation:** Rate output prevents standard spiking
   pipeline integration. This is inherent to rate models.

9. **No noise, no adaptation:** The simplest possible dynamics — pure
   exponential relaxation to sigmoid of input.

10. **Foundation for Wilson-Cowan:** The single-unit SigmoidRate is
    exactly one equation of the Wilson-Cowan model — E and I are each
    SigmoidRateNeurons with cross-coupled inputs.

---

## Theoretical Context

### Rate coding vs temporal coding

The SigmoidRateNeuron implements the **rate coding** hypothesis: neural
information is carried by the mean firing rate, not by individual spike
times. This is the classical view (Adrian 1926) and remains dominant in
sensory neuroscience, motor control, and machine learning.

In contrast, spiking models (LIF, HH, etc.) can capture **temporal
coding** — information in spike timing, synchrony, and oscillatory phase.

### Mean-field reduction

For a population of N identical LIF neurons driven by independent noise,
the mean firing rate converges to a sigmoid function of the mean input
as N → ∞ (central limit theorem + Siegert formula). The SigmoidRateNeuron
is therefore the infinite-population limit of a LIF ensemble — a rigorous
mean-field reduction.

### Gradient computation

The sigmoid transfer function is differentiable everywhere:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

Maximum gradient: σ'(0) = 0.25 (at the midpoint). This enables gradient-
based learning in rate networks (backpropagation, BPTT) — the SigmoidRate
is the spiking-network equivalent of a standard artificial neuron.

### Relationship to artificial neural networks

With β=1 and θ=0, the SigmoidRateNeuron is mathematically identical to
a single-layer perceptron with sigmoid activation — the foundational unit
of classical neural networks. The only difference is the temporal dynamics
(τ), which are absent in feedforward ANNs. Setting τ→0 recovers the
instantaneous perceptron.

### Recurrent rate networks

A collection of SigmoidRateNeurons with mutual connectivity implements
a continuous-time recurrent neural network (CTRNN):

$$\tau_i \frac{dr_i}{dt} = -r_i + \sigma\!\left(\sum_j w_{ij} r_j + I_i\right)$$

CTRNNs are universal function approximators (Funahashi & Nakamura 1993)
and can implement any finite-state automaton (Siegelmann & Sontag 1995).
The SigmoidRateNeuron provides the node dynamics for such networks.
