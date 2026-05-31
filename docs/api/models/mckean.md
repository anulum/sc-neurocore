<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# McKeanNeuron

**Module:** `sc_neurocore.neurons.models.mckean`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::McKeanNeuron`
**Reference:** McKean, H. P. (1970), "Nagumo's equation", *Advances in Mathematics*, 4, 209-223.

`McKeanNeuron` implements the McKean two-state piecewise-linear excitable
system. It is the maintained low-dimensional FitzHugh-Nagumo analogue for
analytical excitability, relaxation oscillation, and fast network experiments
where the voltage nullcline must remain exactly piecewise linear.

## State equation

The state is `(v, w)` with external current `I`:

```text
dv/dt = f(v) - w + I
dw/dt = epsilon * (v - gamma*w)
```

The voltage nonlinearity is the McKean three-branch function:

```text
f(v) = -v        if v < a/2
     = v - a     if a/2 <= v < (1+a)/2
     = 1 - v     if v >= (1+a)/2
```

Default parameters:

| Parameter | Default | Contract |
|-----------|---------|----------|
| `v` | `0.0` | finite |
| `w` | `0.0` | finite |
| `a` | `0.25` | finite, `0 < a < 1` |
| `epsilon` | `0.01` | finite, positive |
| `gamma` | `0.5` | finite, positive |
| `dt` | `0.1` | finite, positive |
| `v_peak` | `0.8` | finite |

A spike is reported only on an upward crossing of `v_peak`. The McKean surface
does not implicitly reset after a spike; callers that need reset dynamics must
call `reset()` explicitly or compose the model inside a reset policy.

## Integration contract

The maintained Python and Rust-engine implementations use candidate-first RK4
for the coupled `(v, w)` ODE. Each RK4 stage evaluates the same piecewise
McKean right-hand side at the stage candidate. State is committed only after the
candidate voltage and recovery variables are both finite.

Invalid runtime contracts are fail-closed:

| Condition | Python public surface | Rust engine / Go / Julia / Rust safety |
|-----------|-----------------------|----------------------------------------|
| Non-finite state | raises before mutation | returns no spike / preserves state |
| Non-finite current | raises before mutation | returns no spike / preserves state |
| Invalid `a`, `epsilon`, `gamma`, `dt` | raises before mutation | returns no spike / preserves state |
| Non-finite RK4 derivative or candidate | raises before mutation | returns no spike / preserves state |

This behavior is intentional: Python exposes explicit numerical errors for
interactive scientific use, while the low-level service/safety surfaces preserve
state and return a no-spike sentinel so foreign runtimes do not unwind across an
FFI or service boundary.

## Public workflow contract

The module-specific behavioural test surface verifies:

| Surface | Contract |
|---------|----------|
| `tests/test_model_mckean.py` | Python RK4 reference, branch equations, dynamics, validation, public network workflow |
| `engine/src/neurons/simple_spiking.rs` | Rust engine RK4 candidate and fail-closed state preservation |
| `src/sc_neurocore/accel/julia/neurons/mckean.jl` | Julia RK4 service parity and invalid-state sentinel behavior |
| `src/sc_neurocore/accel/go/services/mckean.go` | Go RK4 service parity and invalid-state sentinel behavior |
| `src/sc_neurocore/accel/rust/safety/mckean.rs` | Standalone Rust safety RK4 parity and invalid-state sentinel behavior |

The public Python workflow test is named explicitly: McKean public surface
inside the Python simulator. It exercises real `Population`, `Projection`,
`Network`, `SpikeMonitor`, and spike-stat analysis APIs, not synthetic coverage
buckets.

## Example

```python
from sc_neurocore.neurons.models.mckean import McKeanNeuron

neuron = McKeanNeuron()
spikes = []
for step in range(20_000):
    if neuron.step(0.5):
        spikes.append(step)
```

The trajectory is deterministic for a fixed current trace and parameter set.

## Benchmark evidence

Current local benchmark evidence for the maintained Rust engine path:

| Benchmark | Steps | Median | Per step | Artifact |
|-----------|-------|--------|----------|----------|
| `mckean_10k_steps` | 10,000 | 320.24 us | 32.0 ns | `benchmarks/results/local_i5_11600k_criterion_2026-05-31_mckean.json` |

Python reference benchmark evidence:

| Benchmark | Steps | Median per step | Mean per step | Artifact |
|-----------|-------|-----------------|---------------|----------|
| `python_mckean_rk4_reference` | 100,000 | 9.568 us | 9.817 us | `benchmarks/results/local_i5_11600k_python_2026-05-31_mckean.json` |

The benchmark results are evidence for regression tracking, not a target used to
shape tests. Behavioural physics invariants remain the acceptance criteria.

## Scientific scope

The model is useful for:

| Use case | Why McKean is appropriate |
|----------|---------------------------|
| Excitability analysis | Piecewise-linear nullcline gives tractable branch geometry |
| Relaxation oscillations | Slow recovery variable preserves qualitative fast-slow behavior |
| Network experiments | Low state dimension and deterministic RK4 path support reproducible sweeps |
| Cross-runtime parity | Branches and RK4 stages have compact closed-form reference checks |

The implementation does not claim detailed ion-channel biophysics. It is a
validated McKean dynamical system surface with finite-state guards, RK4 parity,
and reproducible public workflow coverage.
