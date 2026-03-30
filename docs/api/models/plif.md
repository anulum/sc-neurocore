# ParametricLIFNeuron (PLIF)

**Module:** `sc_neurocore.neurons.models.plif`
**Reference:** Fang et al. 2021
**Family:** Integrate-and-fire (learnable decay)
**State variables:** `v` (voltage), `a` (learnable parameter)

## Equations

$$V(t+1) = \alpha \cdot V(t) \cdot (1 - s(t)) + I(t)$$
$$\alpha = \sigma(a) = \frac{1}{1 + e^{-a}}$$
$$s(t) = \Theta(V(t) - \theta)$$

Return value: $\Theta(V(t+1) - \theta)$ (spike based on **updated** V).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | 0.0 | Membrane voltage |
| `a` | 0.0 | Learnable parameter (alpha = sigmoid(a)) |
| `threshold` | 1.0 | Spike threshold |
| `dt` | 1.0 | Time step |

## Behaviour

- **Learnable decay:** alpha = sigmoid(a) ∈ (0, 1) controls voltage persistence.
  a=0 → alpha=0.5. Higher alpha → more memory → easier to spike.
- **Geometric accumulation:** V converges to V_ss = I/(1-alpha) when V_ss < threshold.
  Error decays as alpha^t (geometric convergence).
- **Critical current:** I_crit = threshold · (1 - alpha). Below this, no spikes.
  Above it, regular firing. At I ≥ threshold, fires every step.
- **Soft reset:** After spike, V = I (not zero). The (1-spike) factor zeros the
  memory term, but current is immediately added.
- **Spike on updated V:** The returned spike is based on V(t+1), not V(t).
  Old V only determines whether reset occurs.

## Analytical Properties

| Property | Formula |
|----------|---------|
| Steady-state V | $V_{ss} = I / (1 - \alpha)$ |
| Critical current | $I_{crit} = \theta (1 - \alpha)$ |
| Convergence rate | Error ∝ $\alpha^t$ |
| Max rate | 1 spike/step (when I ≥ θ) |

## Infrastructure Pipeline

```
ParametricLIFNeuron
├── step(current) → int {0,1} (deterministic)
├── Population: PoissonInput(weight=1.5, rate=500Hz)
├── Verilog: multiply-accumulate + sigmoid LUT, ~40 LUTs
└── Rust: supported (2 f64 state variables)
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 7 | defaults, binary, sigmoid correctness, a=0 midpoint, monotonicity, bounds, saturation |
| Dynamics | 5 | geometric accumulation, steady-state V, convergence rate, alpha≈1 no-leak, alpha≈0 fast-decay |
| Threshold | 5 | spike-on-updated-V, suprathreshold every-step, exact threshold, critical current, soft reset |
| Learnable rate | 8 | alpha effect on rate, 5-point suprathreshold sweep, subcritical (3 alpha values) |
| Edge cases | 4 | zero input, negative input, reset, determinism |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **32** | |

Key finding: spike return is based on **updated** V (post-step), not pre-step V.
The pre-step check only controls the reset mechanism. This is a subtle but
important implementation detail for surrogate gradient training.
