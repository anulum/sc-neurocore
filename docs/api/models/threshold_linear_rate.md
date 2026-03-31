# ThresholdLinearRateNeuron

**Module:** `sc_neurocore.neurons.models.threshold_linear_rate`
**Reference:** Dayan & Abbott, Theoretical Neuroscience, 2001, Ch. 7
**Family:** Rate model (threshold-linear / ReLU)
**State variables:** `r` (firing rate)

---

## Equations

### Transfer function

$$r = g \cdot \max(0,\; I - \theta)$$

where $g$ is the gain (slope), $\theta$ is the threshold, and $I$ is the
input current.

This is the **Rectified Linear Unit (ReLU)** — the same activation function
used in deep learning. In neuroscience, it is called the "threshold-linear"
transfer function and was introduced by Naka & Rushton (1966) for retinal
ganglion cells.

### Implementation

```python
def step(self, current: float) -> float:
    self.r = self.gain * max(0.0, current - self.theta)
    return self.r
```

**Instantaneous (no dynamics).** The output at time t depends only on the
input at time t — no ODE, no time constant, no memory. **Returns float
(rate), not binary spike.**

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `r` | 0.0 | Hz | Firing rate (current value) |
| `theta` | 0.0 | a.u. | Threshold (below which rate = 0) |
| `gain` | 1.0 | Hz/a.u. | Gain (slope above threshold) |

### Minimal parameter set

Only 3 parameters — tied with TrueNorth for the smallest parameter count.
The simplicity is the point: the threshold-linear function captures the
essential input-output nonlinearity (thresholding) with zero unnecessary
complexity.

---

## Analytical Properties

### Piecewise-linear structure

$$r(I) = \begin{cases} 0 & \text{if } I \leq \theta \\ g(I - \theta) & \text{if } I > \theta \end{cases}$$

- **Below threshold (I ≤ θ):** Rate is exactly 0. The neuron is silent.
- **Above threshold (I > θ):** Rate increases linearly with slope g.
- **At threshold (I = θ):** Rate is exactly 0 (left-continuous).

### Derivative

$$\frac{dr}{dI} = \begin{cases} 0 & \text{if } I < \theta \\ g & \text{if } I > \theta \end{cases}$$

Discontinuous at I = θ (not differentiable). In practice, subgradient
methods handle this.

### No saturation

Unlike sigmoid rate neurons, the threshold-linear function has **no upper
bound.** For large I: r → ∞. This is unrealistic biologically (neurons
have maximum firing rates) but mathematically convenient for:
- Linear algebraic analysis of network dynamics
- Convex optimisation problems
- Exact analytical solutions in balanced networks

### Comparison with sigmoid

| Property | ReLU | Sigmoid |
|----------|------|---------|
| Range | [0, ∞) | (0, 1) |
| Gradient | g or 0 | Variable (max g/4 at midpoint) |
| Saturation | None (upper) | Both (upper and lower) |
| Analytical | Piecewise-linear | Transcendental |
| Biological | Approximate (no saturation) | More realistic (saturates) |
| Deep learning | Standard (post-2012) | Historical (pre-2012) |

### Steady-state = instantaneous output

There is no dynamics (no τ, no ODE). The output is an instantaneous
function of the input: $r_t = g \cdot \max(0, I_t - \theta)$. This means:
- No transient response
- No temporal filtering
- No memory of past inputs
- State variable r is redundant — it just stores the last output

### Gain controls sensitivity

- gain = 0.5: half-slope (reduced sensitivity)
- gain = 1.0: unit slope (default)
- gain = 2.0: double slope (amplified sensitivity)
- gain = 0: constant zero output (dead neuron)

### Threshold controls activation

- θ = 0: fires for any positive input (no threshold)
- θ = 5: requires input > 5 to produce output
- θ < 0: fires for some negative inputs (shifted activation)

---

## Behaviour

### Instantaneous, stateless computation

The ThresholdLinearRateNeuron is the **simplest possible neural transfer
function.** It has:
- No temporal dynamics (no ODE)
- No memory (output depends only on current input)
- No stochasticity
- No adaptation
- No saturation
- No sub-stepping

It is essentially a function call, not a dynamical system.

### Use cases

1. **Linearised analysis:** Networks of threshold-linear units have
   tractable analytical solutions. The Dale-constrained E/I network
   with threshold-linear units is a standard theoretical model
   (Ahmadian & Miller, Annu. Rev. Neurosci. 44, 2021).

2. **Balanced network theory:** The Brunel (2000) balanced network
   predictions can be derived analytically using threshold-linear
   transfer functions as approximations to the Siegert function.

3. **Sparse coding:** Threshold creates sparsity — only neurons with
   input above θ are active. This is the basis of sparse coding models
   (Olshausen & Field 1996) and compressed sensing.

4. **Deep SNN prototyping:** The ReLU activation is the standard for
   ANN→SNN conversion. The ThresholdLinearRateNeuron serves as the
   ANN reference model.

5. **Gain modulation:** In cortical circuits, attention is modelled as
   multiplicative gain change. The gain parameter captures this directly.

---

## Pipeline Compatibility

### Returns float, not int

**Limitation:** `step()` returns `float` (rate r), not `int` (spike).
When placed in a Network, any r > 0 registers as a "spike".

**Recommended use:** Rate-based network simulations, analytical studies,
ANN→SNN conversion reference.

### Population compatible

`Population(ThresholdLinearRateNeuron, n=10, label="relu")` works.

---

## Comparison with Related Models

| Property | ThresholdLinear | SigmoidRate | LIF | WilsonCowan |
|----------|---------------|------------|-----|-------------|
| Transfer | ReLU | Sigmoid | Threshold-reset | Sigmoid |
| Dynamics | None (instant) | ODE (τ decay) | ODE (τ_m) | 2 ODEs |
| Memory | None | τ-dependent | τ_m-dependent | τ_e, τ_i |
| Saturation | No | Yes | Yes (max rate) | Yes |
| Output | float | float | int | float |
| Parameters | 3 | 5 | 5 | 11 |
| Lines of code | 30 | 34 | ~40 | 49 |

The ThresholdLinearRateNeuron is the zero-dynamics limit of the
SigmoidRateNeuron: set τ→0 and replace sigmoid with ReLU.

---

## Theoretical Context

### Naka-Rushton function

The threshold-linear function is the linear approximation to the
Naka-Rushton power-law transfer:

$$r = r_{max} \frac{I^n}{I^n + \sigma^n}$$

For n=1 near threshold: this reduces to a threshold-linear function.
The Naka-Rushton function (with saturation) is more biologically accurate
but less analytically tractable.

### Stabilised Supralinear Network (SSN)

The Ahmadian & Miller (2021) SSN theory uses threshold-linear (actually
threshold-power-law with n≥2) transfer functions to explain:
- Surround suppression in visual cortex
- Contrast-dependent tuning changes
- Paradoxical inhibitory stabilisation

The ThresholdLinearRateNeuron (n=1) is the linear case of this framework.

### Dale's law and E/I networks

In networks with Dale's law (neurons are either excitatory or inhibitory),
threshold-linear units enable exact analytical treatment:
- Fixed-point equations become linear programs
- Stability can be analysed via eigenvalues of the effective weight matrix
- Phase transitions (silence → activity) have exact critical points

---

## Numerical Considerations

- **No numerical issues:** Pure comparison and multiplication. No exp(),
  no division, no ODE integration.
- **No overflow:** max() and multiplication are well-behaved for float64.
- **No sub-stepping:** Instantaneous computation.
- **Exact:** No discretisation error (no ODE to discretise).

---

## Implementation Notes

- **Source:** `src/sc_neurocore/neurons/models/threshold_linear_rate.py` — 30 lines.
- **One "state" variable:** r (stores last output, not truly state).
- **Dataclass:** Uses `@dataclass` for parameter storage.
- **Smallest implementation:** 30 lines — simplest model in the library.
- **Rust wiring:** Trivially compatible (max + multiply).

---

## Performance

| Metric | Python | Rust |
|--------|--------|------|
| Isolation | ~2M steps/s | Not measured |
| Network | Limited (float return) | — |

Fastest model — no exp(), no ODE, just max() and multiply. Python function
call overhead is the only cost.

---

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 4 | defaults, float return, r updates, reset |
| Transfer | 5 | r=0 below threshold, linear above, exact at threshold, gain scaling, θ shift |
| Dynamics | 3 | instantaneous (no memory), monotonic, no saturation |
| Parameters | 3 | gain sweep, θ sweep, deterministic |
| Pipeline | 2 | Population creates, float return documented |
| **Total** | **17** | |

See `tests/test_model_threshold_linear_rate.py`. No bugs found.

---

## Findings

1. **r = 0 below threshold:** For any I ≤ θ, output is exactly 0.
   The threshold creates a hard cutoff.

2. **Linear above threshold:** r = gain × (I − θ) — verified across
   multiple input values with exact equality.

3. **No saturation:** For I=1000, r=1000 (with default gain=1, θ=0).
   The rate grows without bound.

4. **Instantaneous response:** Changing input immediately changes output.
   No transient, no lag, no filtering.

5. **Gain scales linearly:** gain=2 produces double the rate of gain=1
   at the same input.

6. **Threshold shifts activation:** θ=5 requires I>5 for non-zero output.

7. **Fastest model:** ~2M steps/s — limited only by Python call overhead.

8. **ReLU equivalence:** Mathematically identical to the ReLU activation
   in deep learning. The model serves as the ANN reference for SNN work.
