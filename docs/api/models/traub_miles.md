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

`alpha_m`, `beta_m` and `alpha_n` are `0/0` at `v = -54`, `v = -27` and `v = -52`. Their limits follow from L'Hôpital's rule, `a d / (1 - exp(-d / k)) -> a k`: `alpha_m(-54) = 0.32 * 4 = 1.28`, `beta_m(-27) = 0.28 * 5 = 1.40` and `alpha_n(-52) = 0.032 * 5 = 0.16`. Within `|d| <= 1e-6` of each point, each implementation that evaluates the rates uses the first-order expansion about that limit, `a (k + d / 2)` (and `a (k - d / 2)` for `beta_m`), instead of dividing by a near-zero denominator. Outside that window the closed form is unchanged, so the rates are continuous across the window edge to within the closed form's own rounding (under `1e-9`). The Python reference, the Rust engine and standalone safety crates, and the Go and Julia lanes evaluate the same expression; the Mojo file is a step transcript with no rate code. Before 2026-11-24 the window returned `8.0`, `5.6` and `0.32`, which are 6.25, 4 and 2 times the limits.

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
| Rust engine / PyO3 | `engine/src/neurons/biophysical/traub_miles.rs`, `engine/src/pyo3_neurons.rs` | RK4 parity surface exposed through `get_state()` |
| Go service | `src/sc_neurocore/accel/go/services/traub_miles.go` | RK4 mirror with fail-closed error return |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/traub_miles.jl` | RK4 mirror returning `-1` on invalid candidate |
| Mojo kernel note | `src/sc_neurocore/accel/mojo/kernels/traub_miles.mojo` | RK4 stage contract documented for future compiled kernel work |
| Rust safety surface | `src/sc_neurocore/accel/rust/safety/traub_miles.rs` | Standalone RK4 safety reference |

## Current benchmark evidence

Benchmark command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_traub_miles.py
```

Current local result from `benchmarks/results/bench_traub_miles.json` (2026-11-24, i5-11600K, Python 3.12.16):

| Backend | Steps/s | Wall ms for 20,000 steps | Speedup |
| --- | ---: | ---: | ---: |
| Python reference | 7,837 | 2,551.86 | 1.00x |
| Rust PyO3 | 334,213 | 59.84 | 42.64x |

Across five alternating pairs on that host, the medians were 7,837 (Python) and 349,003 (Rust) steps/s after the fix, against 8,325 and 363,179 before it. These differences are within the run-to-run spread of the loaded host (Python 6,204 to 9,109; Rust 231,848 to 404,178), since the fix changes only code that this run never reaches.

Parity evidence: the Python reference and Rust PyO3 class produced measured `max_abs_delta = 0.0` over 2,000 steps at `current = 5.0` on the 2026-11-24 build, in each of five alternating pairs of runs against the previous build on the same i5-11600K host. The 20,000-step state (244 spikes) is unchanged by the removable-singularity fix, since that trajectory never enters the `1e-6` windows.

## Test evidence

Module-specific evidence is held in the five `tests/test_model_traub_miles_traub_miles_*.py` modules (F-I, Hodgkin-Huxley properties, isolation, parameters and pipeline) and in `tests/test_traub_miles_removable_rate_limits.py`. Together they check:

- the analytic limits at the three removable rate singularities, continuity inside the `1e-6` window and across its edge, the unchanged closed form away from it, and the Julia lane's rates bit for bit against Python's;
- RK4 reference separation from the prior gate-first Euler path;
- finite-state preservation on invalid input, invalid runtime state, rate overflow, and current-balance overflow;
- spike-train determinism and sustained-drive stability;
- population, projection, and analysis workflow contracts that use the real production boundary.

The Rust engine and safety crates (`cargo test traub_miles`) and the Go service (`go test ./services/ -run TraubMiles`) carry the same limit and window-edge checks.

Focused verification on 2026-11-24:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_traub_miles_removable_rate_limits.py tests/test_model_traub_miles_traub_miles_*.py -q
```

Evidence: 73 tests passed, and `src/sc_neurocore/neurons/models/traub_miles.py` reported 100% of its 97 statements and 24 branches in the focused run.
