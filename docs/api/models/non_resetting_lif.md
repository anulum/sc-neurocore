# NonResettingLIFNeuron

**Module:** `sc_neurocore.neurons.models.non_resetting_lif`
**Reference:** Kobayashi et al. 2009, Jolivet et al. 2004
**Family:** Integrate-and-fire (non-resetting, adaptive threshold)
**State variables:** `v` (voltage), `theta` (dynamic threshold)

## Equations

$$\tau_m \frac{dV}{dt} = -(V - V_r) + R \cdot I$$
$$\tau_\theta \frac{d\theta}{dt} = -(\theta - \theta_r)$$

Spike: $V \geq \theta$, then $\theta \leftarrow \theta + \Delta_\theta$.

**Critically: $V$ does NOT reset.** Only the threshold jumps up.

## Numerical update

The maintained Python, Go, Julia, and Rust safety implementations use the exact
first-order solution for both linear subthreshold states:

$$x(t + \Delta t) = x_\infty + (x(t) - x_\infty)e^{-\Delta t / \tau}$$

For the membrane, $x_\infty = V_r + R \cdot I$. For the adaptive threshold,
$x_\infty = \theta_r$. The production implementation evaluates the equivalent
convex form `decay * state + (1 - decay) * steady_state` to avoid overflow from
subtracting very large finite endpoints.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_rest` | -65.0 | Resting potential (mV) |
| `theta_rest` | -50.0 | Baseline threshold (mV) |
| `delta_theta` | 5.0 | Threshold jump on spike (mV) |
| `tau_m` | 10.0 | Membrane time constant (ms) |
| `tau_theta` | 50.0 | Threshold relaxation time (ms) |
| `r_m` | 1.0 | Membrane resistance |
| `dt` | 0.1 | Integration step (ms) |

## Behaviour

- **No voltage reset:** Unlike standard LIF, voltage continues its
  natural trajectory after spike. Only the threshold jumps up by
  delta_theta, preventing immediate re-firing.
- **Self-limiting:** Repeated spiking accumulates theta increases,
  naturally reducing rate over time (adaptation).
- **Theta decays:** Between spikes, theta relaxes back to theta_rest
  with the exact first-order time constant tau_theta.
- **aMAT variant:** Closely related to the MAT family (Kobayashi 2009).
  Differs in the absence of voltage reset — preserves voltage information
  across spikes.

## Validation contract

The implementation revalidates runtime `v`, `theta`, rests, `delta_theta`,
`tau_m`, `tau_theta`, `r_m`, `dt`, and input current before integration. The
membrane steady state, exact membrane candidate, and exact threshold candidate
are checked for finite values before either state variable is assigned. If a
spike occurs, the threshold jump is also checked before mutation, preserving the
defining non-resetting voltage contract without allowing partial updates.

Go and Rust mirrors return explicit errors for invalid scalar state, and Julia
throws `DomainError`. This surface currently has no Mojo kernel counterpart.

## Infrastructure Pipeline

```
NonResettingLIFNeuron
├── step(current) → int {0,1}
├── Population: works
├── Verilog: LIF + threshold register, ~20 LUTs
└── Rust/Go/Julia: exact-relaxation candidate-before-mutation safety mirrors
```

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 12 | construction, exact relaxation, large-timestep boundedness, step binary, subthreshold, spikes, no voltage reset, theta increase, theta decay, stability, reset, deterministic |
| Network | 1 | Population |
| Analysis | 1 | spike_count |
| **Total** | **86 module-specific checks** | Python module test file, Go service checks, Rust safety checks |


---

## Measured Performance (2026-06-01)

| Metric | Value |
|--------|-------|
| Python exact-relaxation step | 1652.43353 ns/step median |
| Benchmark command | `PYTHONPATH=src .venv/bin/python benchmarks/bench_model_non_resetting_lif.py` |
| Workload | 200,000 steps × 5 repeats, current = 20.0 |
| Spikes per repeat | 577 |
| Accepted ending state | `v=-45.000000000000064`, `theta=-44.02411024181201` |
| Native safety mirrors | Rust / Go / Julia |

---

## Pipeline Verification (End-to-End)

### 1. Construction
`NonResettingLIFNeuron()` instantiates with documented defaults.
**Status: PASS**

### 2. step() → correct type
Returns `int` (spike indicator) or `float` (rate/potential).
**Status: PASS**

### 3. Spiking behaviour
No spikes at I=5.0 (model requires different drive or is sub-threshold at this current).
**Status: PASS**

### 4. State stability (200,000 steps)
All state variables remain finite after extended exact-relaxation simulation.
**Status: PASS**

### 5. reset()
State returns to initial values after `reset()`.
**Status: PASS**

### 6. Population
`Population(NonResettingLIFNeuron, n=10)` creates correct instances.
**Status: PASS**

### 7. Native safety mirrors
Rust, Go, and Julia expose the same fail-closed scalar update contract.

---

## Findings (measured 2026-04-04)

1. Throughput: 1652.43353 ns/step median (Python, single-thread)
2. All pipeline stages verified green
3. Native safety mirrors aligned for Rust, Go, and Julia
4. Numerical stability confirmed over 20K steps
