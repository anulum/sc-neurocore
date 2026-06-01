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

The implementation rejects non-physical configurations before integration:
`v` and `a` must be finite, `threshold` must be finite and positive, `dt`
must be finite and positive, and runtime current must be finite before state
mutation. The sigmoid is evaluated in a branch-stable form so very large
negative learnable parameters saturate to `alpha=0` without overflow. Runtime
state is revalidated before every step, and the candidate voltage is computed
before mutation so overflowing finite drives preserve the previous state.

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
└── Go / Julia / Rust safety: candidate-first state preservation
```

## Test Surface

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 24 parametrised/behavioural checks | defaults, binary, sigmoid correctness, a=0 midpoint, monotonicity, bounds, stable saturation, fail-closed parameter/current/runtime/candidate validation |
| Dynamics | 5 | geometric accumulation, steady-state V, convergence rate, alpha≈1 no-leak, alpha≈0 fast-decay |
| Threshold | 5 | spike-on-updated-V, suprathreshold every-step, exact threshold, critical current, soft reset |
| Learnable rate | 8 | alpha effect on rate, 5-point suprathreshold sweep, subcritical (3 alpha values) |
| Edge cases | 4 | zero input, negative input, reset, determinism |
| Network | 2 | population, spikes |
| Analysis | 2 | spike_count, consistency |
| **Total** | **49 Python module checks** | plus Go service, Julia, and Rust safety checks |

Key finding: spike return is based on **updated** V (post-step), not pre-step V.
The pre-step check only controls the reset mechanism. This is a subtle but
important implementation detail for surrogate gradient training.


---

## Measured Performance (2026-06-01)

| Metric | Value |
|--------|-------|
| Python candidate-first step | 777.51111 ns/step median |
| Benchmark command | `PYTHONPATH=src .venv/bin/python benchmarks/bench_model_plif.py` |
| Workload | 200,000 steps × 5 repeats, current = 0.7 |
| Spikes per repeat | 100,000 |
| Accepted ending voltage | `1.0499999999999998` |
| Native safety mirrors | Go / Julia / Rust |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`ParametricLIFNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
10000 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (200,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(ParametricLIFNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Native safety mirrors
Go, Julia, and Rust preserve the previous state when runtime state or candidate
voltage is invalid.

---

## Findings (measured 2026-04-04)

1. Throughput: 777.51111 ns/step median (Python, single-thread)
2. All pipeline stages verified green
3. Native safety surfaces validate the same PLIF update and stable-sigmoid contract
4. Numerical stability confirmed over 200K steps
