<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Reference Trace Harness

The reference-trace harness validates schema-driven neuron models against
committed scalar feature contracts. A corpus entry defines the schema model,
runner, deterministic protocol, provenance, expected features, and per-feature
tolerances. The production validator loads those JSON entries from the package,
executes the same `UniversalNeuron` runner used by public schema workflows, and
reports feature-level mismatches without falling back to another trace.

This page documents the WC-A1 deterministic schema corpus. It does not claim
NEST, Brian2, NEURON, or published-figure replay coverage; those remain separate
external-simulator validation surfaces.

## Current Corpus

The committed corpus has one reference entry for every deterministic bundled
schema model. `poisson` and `escape_rate` are excluded from this deterministic
table because their schemas are stochastic.

| Trace | Schema | Runner | Provenance |
|-------|--------|--------|------------|
| `adex_resting_adaptation_doi` | `adex` | `universal_dsl` | Independent explicit-Euler re-derivation of the subthreshold equations from `neurons/model_schemas/adex.toml` with DOI-backed schema provenance |
| `cazelles_map_bursting_doi` | `cazelles_map` | `universal_dsl` | Independent simultaneous clipped logistic fast/slow map iteration (Cazelles, Courbage & Rabinovich 2001, `method="map"`, level `x >= x_threshold` event) from `neurons/model_schemas/cazelles_map.toml` with DOI-backed schema provenance |
| `connor_stevens_driven_spiking_doi` | `connor_stevens` | `universal_dsl` | Independent macro-step RK4 re-derivation of the driven A-current oscillator (100 inner `dt=0.01` sub-steps per 1 ms macro step, no reset, macro-boundary `v >= 0` crossing) from `neurons/model_schemas/connor_stevens.toml` with DOI-backed schema provenance |
| `dpi_neuron_driven_spiking_doi` | `dpi_neuron` | `universal_dsl` | Independent explicit-Euler re-derivation of the current-mode differential-pair-integrator membrane from `neurons/model_schemas/dpi_neuron.toml` with DOI-backed schema provenance |
| `exp_if_resting_exponential_doi` | `exp_if` | `universal_dsl` | Independent explicit-Euler re-derivation of the resting equation from `neurons/model_schemas/exp_if.toml` with DOI-backed schema provenance |
| `fitzhugh_nagumo_driven_oscillation_doi` | `fitzhugh_nagumo` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the driven relaxation oscillator (no reset, rising-edge `v >= 1` crossing) from `neurons/model_schemas/fitzhugh_nagumo.toml` with DOI-backed schema provenance |
| `fitzhugh_rinzel_driven_bursting_doi` | `fitzhugh_rinzel` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the three-state fast-slow bursting flow (no reset, rising-edge `v >= 1` crossing) from `neurons/model_schemas/fitzhugh_rinzel.toml` with DOI-backed schema provenance |
| `glif_constant_current_threshold_adaptation` | `glif` | `universal_dsl` | Analytic linear Euler recurrence from `neurons/model_schemas/glif.toml` with DOI-backed schema provenance |
| `hindmarsh_rose_short_bursting_prefix` | `hindmarsh_rose` | `universal_dsl` | Independent explicit-Euler re-derivation of the short bursting prefix from `neurons/model_schemas/hindmarsh_rose.toml` with DOI-backed schema provenance |
| `hodgkin_huxley_driven_spiking_doi` | `hodgkin_huxley` | `universal_dsl` | Independent macro-step RK4 re-derivation of the driven repetitive-spiking membrane (100 inner `dt=0.01` sub-steps per 1 ms macro step, no reset, macro-boundary `v >= 0` crossing) from `neurons/model_schemas/hodgkin_huxley.toml` with DOI-backed schema provenance |
| `izhikevich_regular_spiking_doi` | `izhikevich` | `universal_dsl` | Independent explicit-Euler re-derivation of the regular-spiking equations from `neurons/model_schemas/izhikevich.toml` with DOI-backed schema provenance |
| `izhikevich2007_regular_spiking_doi` | `izhikevich2007` | `universal_dsl` | Independent explicit-Euler re-derivation of the biophysical quadratic equations from `neurons/model_schemas/izhikevich2007.toml` with DOI-backed schema provenance |
| `lif_constant_current_closed_form` | `lif` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lif.toml` |
| `lapicque_constant_current_closed_form` | `lapicque` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lapicque.toml` |
| `mckean_driven_oscillation_doi` | `mckean` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the piecewise-linear relaxation oscillator (no reset, rising-edge `v >= 0.8` crossing) from `neurons/model_schemas/mckean.toml` with DOI-backed schema provenance |
| `mihalas_niebur_driven_spiking_doi` | `mihalas_niebur` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the four-state adaptive-threshold flow from `neurons/model_schemas/mihalas_niebur.toml` with DOI-backed schema provenance |
| `morris_lecar_driven_oscillation_doi` | `morris_lecar` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the driven calcium-potassium relaxation oscillator (no reset, rising-edge `v >= 0` crossing) from `neurons/model_schemas/morris_lecar.toml` with DOI-backed schema provenance |
| `pernarowski_autonomous_bursting_doi` | `pernarowski` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the autonomous three-state beta-cell bursting flow (no reset, rising-edge `v >= 0.5` crossing) from `neurons/model_schemas/pernarowski.toml` with DOI-backed schema provenance |
| `terman_wang_legion_oscillation_doi` | `terman_wang` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the two-state cubic and `tanh`-gated LEGION relaxation oscillator (no reset, rising-edge `v >= 1.5` crossing) from `neurons/model_schemas/terman_wang.toml` with DOI-backed schema provenance |
| `wilson_hr_driven_spiking_doi` | `wilson_hr` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the two-state polynomial cortical flow (level `v >= 0.4` decision, hard voltage reset preserving recovery) from `neurons/model_schemas/wilson_hr.toml` with DOI-backed schema provenance |
| `perfect_integrator_constant_current_sawtooth` | `perfect_integrator` | `universal_dsl` | Analytic post-reset sawtooth solution from `neurons/model_schemas/perfect_integrator.toml` |
| `quadratic_if_zero_current_analytic` | `quadratic_if` | `universal_dsl` | Analytic zero-current Riccati solution from `neurons/model_schemas/quadratic_if.toml` with DOI-backed schema provenance |
| `resonate_fire_subthreshold_resonance_doi` | `resonate_fire` | `universal_dsl` | Analytic linear Euler recurrence from `neurons/model_schemas/resonate_fire.toml` |
| `rulkov_map_driven_spiking_doi` | `rulkov_map` | `universal_dsl` | Independent piecewise-map iteration (Rulkov 2002, `method="map"`, rising `x >= 0` crossing) from `neurons/model_schemas/rulkov_map.toml` with DOI-backed schema provenance |
| `theta_constant_current_phase_analytic` | `theta` | `universal_dsl` | Analytic tangent half-angle phase solution from `neurons/model_schemas/theta.toml` with DOI-backed schema provenance |
| `wang_buzsaki_driven_spiking_doi` | `wang_buzsaki` | `universal_dsl` | Independent macro-step Gauss-Seidel re-derivation of the driven fast-spiking interneuron (50 inner `dt=0.01` sub-steps per 0.5 ms macro step, gates `h`/`n` updated before `v`, no reset, macro-boundary `v >= v_threshold` crossing) from `neurons/model_schemas/wang_buzsaki.toml` with DOI-backed schema provenance |

All entries record spike count, first spike step, and final/min/max/mean
features for the declared state variables. The tests independently recompute the
LIF, QIF, perfect-integrator, resonate-fire, theta, GLIF, Izhikevich, Cazelles map,
Izhikevich 2007, FitzHugh-Nagumo, FitzHugh-Rinzel, Pernarowski, Terman-Wang, Wilson-HR, McKean, AdEx, exponential-IF,
Hindmarsh-Rose, Morris-Lecar,
Hodgkin-Huxley, Connor-Stevens, Wang-Buzsaki, DPI, and Mihalas-Niebur analytic,
explicit-Euler, sequential Gauss-Seidel, or fourth-order Runge-Kutta solutions — every deterministic
bundled-schema entry — so the committed feature values are not merely copied from
the runner output. The
Hodgkin-Huxley, Connor-Stevens, and Wang-Buzsaki re-derivations reuse the runner's
numpy activation, exponential, and exprel functions so the conductance rate terms
match bit-for-bit. The GLIF entry independently re-derives its four-state
classical-RK4 flow, candidate-level adaptive threshold decision, and
candidate-first reset across a 54-spike driven train; the Izhikevich entry
re-derives the exact regular-spiking explicit-Euler recurrence including its
`v = c`, `u = u + d` reset; the FitzHugh-Nagumo entry re-derives its cubic
relaxation oscillator with the faithful four-stage RK4 step and rising-edge
`v >= 1` crossing detection (no reset — the re-enrolled model is a genuine
relaxation oscillator, not integrate-and-fire); and the DPI entry re-derives its
current-mode leaky-integrator recurrence with the `i_mem = i_reset` reset, its
non-negative drive keeping the source model's `max(i_mem, 0)` rectification inert.
The FitzHugh-Rinzel entry extends that independent cubic RK4 recurrence with the
ultra-slow `y` modulation equation. It advances `v`, `w`, and `y` simultaneously,
uses the same no-reset upward-crossing decision, and reproduces all three state
feature sets plus the eight-crossing `I=0.5` protocol without calling the hand
model or schema runner.
The Pernarowski entry independently advances its cubic fast coordinate, recovery
`w`, and ultra-slow adaptation `z` with simultaneous classical RK4. It uses the
same no-reset upward-crossing decision and reproduces all three state feature sets,
the first-spike step, and the 17-crossing zero-drive protocol without calling the
hand model or schema runner.
The Terman-Wang entry independently advances its cubic fast coordinate and
`tanh`-gated recovery state with simultaneous classical RK4. It applies the
no-reset `v >= 1.5` upward-crossing decision and reproduces both state feature
sets, the first crossing at step 29, and the three-crossing `I=0.5` protocol
without calling the hand model or schema runner. The runner evaluates the
transcendental through NumPy while the independent recurrence uses `math.tanh`,
so the committed feature tolerance captures floating-point library differences;
the spike count and first-spike step remain exact.
The Wilson-HR entry independently advances its polynomial membrane and linear
recovery coordinates with simultaneous classical RK4. It applies level
`v >= 0.4` detection and hard-resets only `v` to `-0.7`, preserving the RK4
candidate `r` state. The re-derivation reproduces both post-step state feature
sets, the first spike at step 2, and the four-spike `I=10.0` protocol without
calling the hand model or schema runner.
The Mihalas-Niebur entry re-derives the exact classical fourth-order Runge-Kutta
recurrence for its four linear states (membrane, adaptive threshold, and two
spike-triggered currents), including the `v = v_reset + b*(v - v_rest)`,
`theta = max(theta, theta_reset)`, `i1 += r1`, `i2 += r2` reset; because
`theta_reset` exceeds `theta_inf` the `max()` threshold floor engages on every
spike, so the state-to-state `v >= theta` comparison is a genuine adaptive
threshold rather than a fixed level.
The McKean entry re-derives the exact classical fourth-order Runge-Kutta recurrence
for its three-branch piecewise-linear membrane `f(v) = min(max(-v, v - a), 1 - v)` and
linear recovery, with rising-edge `v >= v_peak` crossing detection and no reset; at the
enrolled sustained-oscillation regime (`epsilon = 0.2`, `gamma = 0.5`, `I = 0.6`) it is a
robust limit cycle whose sixteen upward crossings survive Q16.16 rounding, so the
min/max branch selection lowers to fixed point without a look-up table.
The Morris-Lecar entry re-derives the exact classical fourth-order Runge-Kutta
recurrence for its calcium-potassium conductance oscillator — sigmoidal `tanh`
calcium activation and `cosh`/`tanh` potassium gating, reusing the runner's numpy
transcendentals — with rising-edge `v >= 0` crossing detection and no reset. At the
enrolled depolarising regime (`I = 100`) it is a robust relaxation oscillator whose
seven upward crossings the Q16.16 cosh/tanh look-up datapath reproduces exactly;
unlike the polynomial FitzHugh-Nagumo and piecewise-linear McKean oscillators the
per-step state is not bit-identical to the hand model (numpy versus `math`
transcendentals through distinct RK4 drivers), so the parity is at the spike-count
level, robust across the whole `I in [90, 110]` band.
The Connor-Stevens entry re-derives the exact **macro-step** RK4 recurrence for its
six-state A-current oscillator: each 1 ms macro step advances 100 inner four-stage RK4
sub-steps of `dt = 0.01`, and the rising-edge `v >= 0` crossing is taken only on the macro
boundary — matching the maintained `ConnorStevensNeuron`, whose `step()` is itself a
100-sub-step macro step. The six-state membrane and Na/K/A-type gating rate functions
(numpy exp/exprel and the cube-root `a`-gate) reproduce the schema runner bit-for-bit, so the
schema counts the same ten action potentials the hand model does (`hand == schema` exact),
which the earlier single-step Euler schema could not. This macro-step integration mode
(`[integration] substeps`) is what lets the schema faithfully replicate a sub-stepping hand
model rather than over-counting one crossing per sub-step.
The Hodgkin-Huxley entry re-derives the same exact **macro-step** RK4 recurrence for the
four-state 1952 membrane: each 1 ms macro step advances 100 inner four-stage RK4 sub-steps of
`dt = 0.01` with a macro-boundary `v >= 0` crossing and no reset, matching
`HodgkinHuxleyNeuron(integrator="rk4")` — whose `step()` is itself a 100-sub-step macro step
over the same simultaneous RK4, not the Gauss-Seidel `baseline_euler` default. The four-state
membrane and Na/K gating rate functions (numpy exp and the exprel-rewritten `alpha_m` /
`alpha_n`) reproduce the schema runner bit-for-bit, so the driven schema counts the same five
action potentials the hand model does (`hand == schema` exact), which the earlier single-step
Euler resting-gate schema could not.
The perfect-integrator, FitzHugh-Nagumo, FitzHugh-Rinzel, Pernarowski, Terman-Wang, Wilson-HR, Rulkov, Cazelles, McKean, Morris-Lecar,
Hodgkin-Huxley, Connor-Stevens,
Izhikevich, Izhikevich 2007, DPI, and Mihalas-Niebur entries are spike-bearing;
they validate reset (or, for FitzHugh-Nagumo, FitzHugh-Rinzel, Pernarowski, Terman-Wang, Rulkov, McKean, Morris-Lecar,
Hodgkin-Huxley, and Connor-Stevens, rising-edge crossing) and first-spike features,
not only quiet trajectories. The
Rulkov entry iterates the Rulkov 2002 piecewise fast/slow map with the
`method = "map"` integration mode (`x_{n+1} = f(x_n, y_n)`, iterated as a discrete
map rather than integrated as an ODE), so the trajectory is bounded and its
committed features are independently re-derived exactly; a driving current
exercises all three fast-map branches (rational subthreshold, spike plateau, hard
reset). Its rising-crossing reference counts ten events rather than the twenty
positive-level steps produced by the superseded schema. The Cazelles entry independently iterates
the simultaneous clipped logistic fast/slow map at `I=0.5`; the 30-step window exercises interior
and lower-clip branches and records two level events. The reference is deliberately bounded because
the `a=3.8` fast map amplifies fixed-point perturbations on longer chaotic trajectories. The QIF and theta tolerances are wider than
machine-epsilon feature precision because the current schema runner declares
explicit Euler integration while those references are continuous analytic
solutions.

## Public API

```python
from sc_neurocore.neurons.reference_traces import validate_all_reference_traces

reports = validate_all_reference_traces()
assert all(report.passed for report in reports)
```

Use `validate_reference_trace(name)` for one committed trace, or
`reference_trace_spec_from_payload(payload)` when reviewing a candidate corpus
entry before committing it. Malformed payloads fail closed on schema version,
runner, schema name, protocol fields, feature values, and tolerance fields.

## Verification

The focused harness selector is:

```bash
PYTHONPATH=src python -m pytest tests/test_reference_traces.py tests/test_reference_trace_payloads.py -q
```

Exact-file coverage for the implementation modules is measured with:

```bash
PYTHONPATH=src python -m coverage run --rcfile=/dev/null --source=src/sc_neurocore/neurons -m pytest tests/test_reference_traces.py tests/test_reference_trace_payloads.py -q
PYTHONPATH=src python -m coverage report --rcfile=/dev/null --include='src/sc_neurocore/neurons/reference_trace*.py' --fail-under=100 -m
```

## External Simulator Boundary

The deterministic bundled-schema corpus is complete for package-local
`UniversalNeuron` validation. External NEST, Brian2, NEURON, and
published-figure replay traces require separate adapters or recorded fixtures
before they can be represented as external-simulator evidence.
