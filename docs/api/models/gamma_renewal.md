# GammaRenewalNeuron

**Module:** `sc_neurocore.neurons.models.gamma_renewal`
**Reference:** Keat et al. 2001
**Family:** Statistical (renewal process)
**State variables:** `_time_since_spike` (elapsed time)

## Equations

ISI distribution: $\text{ISI} \sim \text{Gamma}(k, k/\lambda)$ where $\lambda$ = rate_hz.

Hazard function: $h(t) = f(t) / S(t)$ where $f$ = Gamma PDF, $S$ = survival.

$$P(\text{spike in } dt) = h(t) \cdot dt$$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rate_hz` | 50.0 | Mean firing rate (Hz) |
| `shape_k` | 3 | Gamma shape — k=1 = Poisson, higher = more regular |
| `dt_ms` | 1.0 | Time step (ms) |

## Behaviour

- **Renewal process:** Each spike resets the clock. ISI distribution
  is Gamma with shape k and rate k·λ (mean ISI = 1/λ).
- **k=1 → Poisson:** Exponential ISI, constant hazard.
- **k>1 → sub-Poisson:** More regular than Poisson (CV < 1).
  Higher k = less variability.
- **rate_override:** `step(rate_override=...)` replaces base rate
  for time-varying drive.
- **No ODE:** Purely statistical — no membrane voltage dynamics.

## Infrastructure Pipeline

```
GammaRenewalNeuron
├── step(rate_override) → int {0,1} (stochastic)
├── Population: works (no current input needed)
├── Helpers: _log_gamma_int, _gamma_survival (scipy-free)
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 10 | construction, step binary, default rate, rate proportional, rate override, shape k effect (CV), time reset, stability, reset, zero rate |
| Helpers | 3 | log_gamma_int, survival at zero, survival decreasing |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **15** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~27K steps/s |
| Spikes (10K steps, I=5.0) | 53 |
| State stability (20K steps) | PASS |
| Rust parity | N/A |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`GammaRenewalNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
53 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(GammaRenewalNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**N/A** — stochastic model, exact parity not applicable.

---

## Findings (measured 2026-04-04)

1. Throughput: ~27K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: N/A
4. Numerical stability confirmed over 20K steps
