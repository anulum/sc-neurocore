# Traub-Miles Neuron

## Scope

`TraubMilesNeuron` implements the reduced Traub-Miles 1991 CA3 pyramidal conductance model used by SC-NeuroCore for sodium activation/inactivation, delayed-rectifier potassium activation, and leak dynamics. The public state is `(v, m, h, n)` in millivolts and unitless gate fractions.

The model is intended for finite, deterministic conductance simulation. It is not a morphology-resolved compartment model and does not include calcium, M-current, synaptic conductance, or dendritic cable terms.

## Differential system

For external drive `I`, the membrane and gate dynamics are:

```text
dv/dt = -g_na m^3 h (v - e_na) - g_k n^4 (v - e_k) - g_l (v - e_l) + I
dm/dt = alpha_m(v) (1 - m) - beta_m(v) m
dh/dt = alpha_h(v) (1 - h) - beta_h(v) h
dn/dt = alpha_n(v) (1 - n) - beta_n(v) n
```

The rate functions follow the reduced Traub-Miles form:

```text
alpha_m = 0.32 (v + 54) / (1 - exp(-(v + 54) / 4))
beta_m  = 0.28 (v + 27) / (exp((v + 27) / 5) - 1)
alpha_h = 0.128 exp(-(v + 50) / 18)
beta_h  = 4 / (1 + exp(-(v + 27) / 5))
alpha_n = 0.032 (v + 52) / (1 - exp(-(v + 52) / 5))
beta_n  = 0.5 exp(-(v + 57) / 40)
```

Near the removable singularities at `v = -54`, `v = -27`, and `v = -52`, the implementation uses the analytic limiting rates rather than dividing by a near-zero denominator.

## Integration contract

The Python reference and polyglot mirrors now use a candidate-first fixed-step RK4 integrator over the coupled `(v, m, h, n)` system. Each public `step(current)` call splits `dt` into ten RK4 substeps and commits state only after every candidate substep has finite voltage and gate values in `[0, 1]`.

Invalid input and corrupted runtime state fail closed:

```python
from sc_neurocore.neurons.models.traub_miles import TraubMilesNeuron

neuron = TraubMilesNeuron()
spike = neuron.step(5.0)
state = (neuron.v, neuron.m, neuron.h, neuron.n)
```

The spike flag is `1` only when the updated voltage crosses `v_threshold` from below during the public step. Non-finite drive, non-finite parameters, negative conductances, invalid gates, rate overflow, and non-finite current balance raise before mutating committed state.

## Polyglot surfaces

| Surface | File | Status |
| --- | --- | --- |
| Python reference | `src/sc_neurocore/neurons/models/traub_miles.py` | Candidate-first RK4 source of truth |
| Rust engine / PyO3 | `engine/src/neurons/biophysical.rs`, `engine/src/pyo3_neurons.rs` | RK4 parity surface exposed through `get_state()` |
| Go service | `src/sc_neurocore/accel/go/services/traub_miles.go` | RK4 mirror with fail-closed error return |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/traub_miles.jl` | RK4 mirror returning `-1` on invalid candidate |
| Mojo kernel note | `src/sc_neurocore/accel/mojo/kernels/traub_miles.mojo` | RK4 stage contract documented for future compiled kernel work |
| Rust safety surface | `src/sc_neurocore/accel/rust/safety/traub_miles.rs` | Standalone RK4 safety reference |

## Current benchmark evidence

Benchmark command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_traub_miles.py
```

Current local result from `benchmarks/results/bench_traub_miles.json`:

| Backend | Steps/s | Wall ms for 20,000 steps | Speedup |
| --- | ---: | ---: | ---: |
| Python reference | 8,068 | 2,478.78 | 1.00x |
| Rust PyO3 | 347,831 | 57.50 | 43.11x |

Parity evidence: the Python reference and Rust PyO3 class produced measured `max_abs_delta = 0.0` over 2,000 steps at `current = 5.0` on the current build.

## Test evidence

Module-specific evidence is held in `tests/test_model_traub_miles.py`. The suite checks:

- Traub-Miles rate singularity guards.
- RK4 reference separation from the prior gate-first Euler path.
- Finite-state preservation on invalid input, invalid runtime state, rate overflow, and current-balance overflow.
- Spike-train determinism and sustained-drive stability.
- Population, projection, and analysis workflow contracts that use the real production boundary.

Focused verification on 2026-06-01:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_traub_miles.py -q
```

Evidence: 41 module-specific tests passed, and `src/sc_neurocore/neurons/models/traub_miles.py` reported 100% statement exercise in the focused run.
