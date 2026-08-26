# GLMNeuron

**Module:** `sc_neurocore.neurons.models.glm_neuron`
**Reference:** Pillow, J.W. et al. (2008). *Nature* 454:995–999 (doi 10.1038/nature07140)
**Family:** Statistical (point-process GLM)
**State variables:** `_stim_buf` (stimulus history), `_spike_buf` (spike history)

The exponential nonlinearity, log-rate clip to [−20, 20], and per-bin
Bernoulli sampling are the repository's discrete-time specialisation of
the paper's point process.

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
├── step(stimulus, uniform=None) → int {0,1}
│     uniform=None  → Bernoulli draw from the seeded generator
│     uniform=u     → deterministic draw from the explicit sample u ∈ [0, 1)
├── Population: works (no current needed — uses stimulus)
├── Filters: customisable via k, h arrays
└── Rust: supported via NeuronVariant
```

### Determinism, Seeding, and Invalid-Input Atomicity

- `GLMNeuron(seed=42)` gives a reproducible generator; `seed=None`
  (default) draws fresh entropy, matching the historical behaviour.
- Passing `uniform` to `step` supplies the Bernoulli sample explicitly
  and bypasses the generator — this is the exact cross-backend parity
  contract (the same explicit samples drive every backend).
- Non-finite `stimulus`, an out-of-domain `uniform` (outside `[0, 1)`),
  a corrupted configuration, or a corrupted history buffer raises
  `ValueError` with the pre-step state preserved exactly — a NaN input
  can no longer poison the stimulus history.

### Backend Inventory

| Surface | Status |
|---------|--------|
| Python reference | `src/sc_neurocore/neurons/models/glm_neuron.py` |
| Production Rust engine | `engine/src/neurons/special/spike_response_models.rs` (`try_step`, reference filters, log-rate clip) |
| PyO3 binding | `engine/src/bindings/stochastic/glm.rs` (typed `ValueError`; `step(stimulus, uniform=None)`, `get_state`) |
| Standalone safety Rust | `src/sc_neurocore/accel/rust/safety/glm_neuron.rs` (explicit-uniform contract) |
| Go service | `src/sc_neurocore/accel/go/services/glm_neuron.go` (`TryStep(stimulus, uniform)`) |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/glm_neuron.jl` (atomic `ArgumentError`) |
| Mojo | not implemented; no kernel exists and no parity is claimed |
| Silicon / RTL | not implemented; no HDL parity claimed |
| Backend parity | engine, safety Rust, Go, Julia vs Python: 64-step spike train and history buffers ≤ 1e-12 under the explicit-uniform contract |

The original engine surface used constant defaults (`k=0.1`, `h=-0.5`).
Those coefficients are a GLM configuration, not a distinct scientific model,
so the canonical constructor now follows the documented reference filters
without growing the catalogue. The historical configuration remains directly
reconstructible through Rust `GLMNeuron::new_legacy_constant_filters` and PyO3
`GLMNeuron.legacy_constant_filters`, with an executed regression test.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Source model | `tests/test_model_glm_neuron_glm_atomicity.py` | defaults, seeding, explicit samples, atomic rejection, reproducibility |
| Backend parity | `tests/test_glm_neuron_backends.py` | engine, legacy engine configuration, safety Rust, Go, Julia, custody claims |
| Hosted gate | `.github/workflows/ci.yml` | exact statement/branch coverage and backend execution |


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
Exact under the explicit-uniform sample contract
(`tests/test_glm_neuron_backends.py`): the engine binding, standalone
safety Rust, Go, and Julia reproduce the Python spike train and history
buffers within 1e-12 when all surfaces consume the same explicit
uniform samples. Free-running (generator-driven) sampling remains
distribution-level only, as the generators differ by design.

---

## Current evidence boundary

The 2026-04-04 throughput and spike-count rows above are retained as historical
local measurements and were not re-benchmarked by this correctness unit. They
are not production performance claims. Current cross-language correctness is
the executed explicit-uniform parity contract described above; free-running
generator speed and distribution parity require separate benchmark evidence.
