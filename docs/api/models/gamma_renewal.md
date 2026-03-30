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
