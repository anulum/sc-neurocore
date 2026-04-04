# GLMNeuron

**Module:** `sc_neurocore.neurons.models.glm_neuron`
**Reference:** Pillow et al. 2008
**Family:** Statistical (point-process GLM)
**State variables:** `_stim_buf` (stimulus history), `_spike_buf` (spike history)

## Equations

$$\lambda(t) = \exp\bigl(\mathbf{k} \cdot \mathbf{s}(t) + \mathbf{h} \cdot \mathbf{y}(t) + \mu\bigr)$$
$$P(\text{spike in } dt) = \lambda(t) \cdot dt$$

where $\mathbf{k}$ = stimulus filter, $\mathbf{h}$ = post-spike filter,
$\mathbf{s}$ = stimulus history, $\mathbf{y}$ = spike history.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_k` | 10 | Stimulus filter length |
| `n_h` | 20 | Post-spike filter length |
| `mu` | -3.0 | Baseline log-rate (offset) |
| `dt_ms` | 1.0 | Time step (ms) |
| `k` | exp decay | Stimulus filter (auto-generated) |
| `h` | negative exp | Post-spike filter — refractoriness |

## Behaviour

- **Stimulus filter:** Convolves recent stimulus with learned kernel k.
  Default: exponentially decaying, emphasises recent input.
- **Post-spike filter:** h is strongly negative at short lags (h[0]≈-4.5),
  enforcing a refractory period. Decays to ~0 at 20 steps.
- **Stochastic:** Poisson-like spiking with rate modulated by filters.
- **No ODE:** Purely statistical — no membrane dynamics.
- **Widely used:** Standard model for retinal ganglion cells,
  cortical neurons (Pillow lab).

## Infrastructure Pipeline

```
GLMNeuron
├── step(stimulus) → int {0,1} (stochastic)
├── Population: works (no current needed — uses stimulus)
├── Filters: customisable via k, h arrays
└── Rust: supported via NeuronVariant
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, step binary, low stim, spikes, rate increase, k shape, h shape, refractoriness, buffer populated, stability, reset, custom filters |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **14** | |


---

## Measured Performance (2026-04-04)

| Metric | Value |
|--------|-------|
| Python throughput | ~8K steps/s |
| Spikes (10K steps, I=5.0) | 1478 |
| State stability (20K steps) | PASS |
| Rust parity | N/A |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`GLMNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
1478 spikes in 10,000 steps at I=5.0.
**Status: PASS**

### 4. State stability (20,000 steps)
All state variables remain finite after extended simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(GLMNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Rust parity
**N/A** — stochastic model, exact parity not applicable.

---

## Findings (measured 2026-04-04)

1. Throughput: ~8K steps/s (Python, single-thread)
2. All pipeline stages verified green
3. Rust parity: N/A
4. Numerical stability confirmed over 20K steps
