<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Reference Trace Harness

The reference-trace harness validates schema-driven neuron models against
committed scalar feature contracts. A corpus entry defines the schema model,
runner, deterministic protocol, provenance, expected features, and per-feature
tolerances. The production validator loads those JSON entries from the package,
executes the same `UniversalNeuron` runner used by public schema workflows, and
reports feature-level mismatches without falling back to another trace.

This page documents the WC-A1 deterministic schema corpus, the pinned IQIF
source-implementation trace, the stateless McCulloch-Pitts primary-source truth
table, and the separate seeded EscapeRate and Poisson statistical references.
It does not claim NEST, Brian2,
NEURON, or published-figure replay coverage; those remain separate external-
simulator validation surfaces.

## Current Corpus

The committed deterministic corpus has one reference entry for every
deterministic bundled schema model. `escape_rate` and `poisson` are stochastic
and therefore have separate exhaustive seeded references immediately after the
table rather than deterministic feature rows. The generic corpus loader
enumerates only `universal_dsl` scalar-feature traces; it deliberately leaves
the two `hand_and_universal_dsl` statistical artifacts to their dedicated
full-period validators instead of silently coercing their RNG, distribution,
and event-hash fields into the deterministic trace schema.

| Trace | Schema | Runner | Provenance |
|-------|--------|--------|------------|
| `adex_resting_adaptation_doi` | `adex` | `universal_dsl` | Independent explicit-Euler re-derivation of the subthreshold equations from `neurons/model_schemas/adex.toml` with DOI-backed schema provenance |
| `aihara_map_primary` | `aihara_map` | `hand_and_universal_dsl` | Independent literal iteration of Aihara (1989), Eqs. 10–12, using the Figure 4 chaotic parameters and Eq. 12 level waveform shaper; publisher article DOI `10.1016/0375-9601(90)90136-C` |
| `cazelles_map_bursting_doi` | `cazelles_map` | `universal_dsl` | Independent simultaneous clipped logistic fast/slow map iteration (Cazelles, Courbage & Rabinovich 2001, `method="map"`, level `x >= x_threshold` event) from `neurons/model_schemas/cazelles_map.toml` with DOI-backed schema provenance |
| `chialvo_map_doi` | `chialvo_map` | `universal_dsl` | Independent simultaneous iteration of Chialvo (1995), Eq. 1 (`method="map"`), with the maintained upward `x_threshold` observation separated from DOI-sourced dynamics |
| `connor_stevens_driven_spiking_doi` | `connor_stevens` | `universal_dsl` | Independent macro-step RK4 re-derivation of the driven A-current oscillator (100 inner `dt=0.01` sub-steps per 1 ms macro step, no reset, macro-boundary `v >= 0` crossing) from `neurons/model_schemas/connor_stevens.toml` with DOI-backed schema provenance |
| `courage_nekorkin_map_autonomous_doi` | `courage_nekorkin_map` | `universal_dsl` | Independent simultaneous iteration of Courbage, Nekorkin & Vdovin (2007), equations 3–5 (`method="map"`, three fast branches, Heaviside discontinuity, upward `x >= x_threshold` crossing), with DOI-backed schema provenance |
| `dpi_neuron_driven_spiking_doi` | `dpi_neuron` | `universal_dsl` | Independent simultaneous explicit-Euler re-derivation of Indiveri, Stefanini & Chicca (2010), Eqs. (2)–(3): nonlinear membrane feedback, after-hyperpolarisation DPI, threshold reset, and spike-driven refractory pulse |
| `ermentrout_kopell_theta_euler_doi` | `ermentrout_kopell_map_neuron` | `universal_dsl` | Independent forward-Euler iteration of the Ermentrout-Kopell (1986) theta flow with maintained `dt=0.1`, gain, pre-wrap upward `theta=pi` event, and modulo `2*pi`; the implementation conventions are separated from the DOI-sourced continuous equation |
| `exp_if_driven_rk4_doi` | `exp_if` | `universal_dsl` | Independent RK4 re-derivation of the driven Fourcaud-Trocmé EIF equation, fitted parameters, +30 mV finite cutoff, and reset |
| `fitzhugh_nagumo_driven_oscillation_doi` | `fitzhugh_nagumo` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the driven relaxation oscillator (no reset, rising-edge `v >= 1` crossing) from `neurons/model_schemas/fitzhugh_nagumo.toml` with DOI-backed schema provenance |
| `fitzhugh_rinzel_driven_bursting_doi` | `fitzhugh_rinzel` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the three-state fast-slow bursting flow (no reset, rising-edge `v >= 1` crossing) from `neurons/model_schemas/fitzhugh_rinzel.toml` with DOI-backed schema provenance |
| `glif_constant_current_threshold_adaptation` | `glif` | `universal_dsl` | Analytic linear Euler recurrence from `neurons/model_schemas/glif.toml` with DOI-backed schema provenance |
| `hindmarsh_rose_short_bursting_prefix` | `hindmarsh_rose` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the driven three-state bursting flow (no reset, rising-edge `x >= 1` crossing) from `neurons/model_schemas/hindmarsh_rose.toml` with DOI-backed schema provenance |
| `hodgkin_huxley_driven_spiking_doi` | `hodgkin_huxley` | `universal_dsl` | Independent macro-step RK4 re-derivation of the driven repetitive-spiking membrane (100 inner `dt=0.01` sub-steps per 1 ms macro step, no reset, macro-boundary `v >= 0` crossing) from `neurons/model_schemas/hodgkin_huxley.toml` with DOI-backed schema provenance |
| `ibarz_tanaka_map_2007_doi` | `ibarz_tanaka_map` | `universal_dsl` | Independent simultaneous iteration of Ibarz et al. (2007), Eqs. 2–3 (four fast branches, slow `u` update, source reset-branch event), without importing the hand model or schema expressions |
| `iqif_a8752eb_tutorial` | `iqif` | `universal_dsl` | Independent literal iteration of the pinned `twetto/iq-neuron@a8752eba49` C++ tutorial: C++-truncated branch point, Q0.3 arithmetic shift, strict upper event, hard reset, and lower clamp; DOI and both source-file hashes are pinned |
| `izhikevich_regular_spiking_doi` | `izhikevich` | `universal_dsl` | Independent explicit-Euler re-derivation of the regular-spiking equations from `neurons/model_schemas/izhikevich.toml` with DOI-backed schema provenance |
| `izhikevich2007_regular_spiking_doi` | `izhikevich2007` | `universal_dsl` | Independent explicit-Euler re-derivation of the biophysical quadratic equations from `neurons/model_schemas/izhikevich2007.toml` with DOI-backed schema provenance |
| `lif_constant_current_closed_form` | `lif` | `universal_dsl` | Closed-form RC solution from `neurons/model_schemas/lif.toml` |
| `lapicque_constant_current_closed_form` | `lapicque` | `universal_dsl` | Independent exact constant-current RC solution with provenance bound to the 2007 English translation of Lapicque (1907), DOI `10.1007/s00422-007-0189-6` |
| `mcculloch_pitts_1943_truth_table` | `mcculloch_pitts` | `universal_dsl` | Independent all-or-none excitatory-count rule from McCulloch and Pitts (1943): fixed positive threshold, absolute inhibitory veto, no fabricated cell state, and a network-scoped one-synaptic-delay boundary; the eight canonical source rows are SHA-256 pinned |
| `sc_triangular_mckean_project` | `sc_triangular_mckean` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the retained project recurrence (no reset, rising-edge `v_peak` crossing); no paper attribution |
| `medvedev_map_first_return_doi` | `medvedev_map` | `universal_dsl` | Independent scalar iteration of Medvedev (2005) Section 4's three-region slow-calcium first-return construction, with the disclosed global calibration and maintained pre-state event convention separated from the DOI-sourced equations |
| `mihalas_niebur_driven_spiking_doi` | `mihalas_niebur` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the four-state adaptive-threshold flow from `neurons/model_schemas/mihalas_niebur.toml` with DOI-backed schema provenance |
| `sc_scaled_reset_adaptive_if_driven_project` | `sc_scaled_reset_adaptive_if` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the retained candidate-proportional-reset project recurrence; no whole-model publication attribution |
| `morris_lecar_driven_oscillation_doi` | `morris_lecar` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the driven calcium-potassium relaxation oscillator (no reset, rising-edge `v >= 0` crossing) from `neurons/model_schemas/morris_lecar.toml` with DOI-backed schema provenance |
| `pernarowski_autonomous_bursting_doi` | `pernarowski` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the autonomous three-state beta-cell bursting flow (no reset, rising-edge `v >= 0.5` crossing) from `neurons/model_schemas/pernarowski.toml` with DOI-backed schema provenance |
| `terman_wang_legion_oscillation_doi` | `terman_wang` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the two-state cubic and `tanh`-gated LEGION relaxation oscillator (no reset, rising-edge `v >= 1.5` crossing) from `neurons/model_schemas/terman_wang.toml` with DOI-backed schema provenance |
| `wilson_hr_driven_spiking_doi` | `wilson_hr` | `universal_dsl` | Independent fourth-order Runge-Kutta re-derivation of the two-state polynomial cortical flow (level `v >= 0.4` decision, hard voltage reset preserving recovery) from `neurons/model_schemas/wilson_hr.toml` with DOI-backed schema provenance |
| `perfect_integrator_constant_current_sawtooth` | `perfect_integrator` | `universal_dsl` | Analytic post-reset sawtooth solution from `neurons/model_schemas/perfect_integrator.toml` |
| `quadratic_if_zero_current_analytic` | `quadratic_if` | `universal_dsl` | Analytic zero-current Riccati solution from `neurons/model_schemas/quadratic_if.toml` with DOI-backed schema provenance |
| `resonate_fire_subthreshold_resonance_doi` | `resonate_fire` | `universal_dsl` | Independent exact constant-input matrix flow with sampled voltage-coordinate crossing and source reset from `neurons/model_schemas/resonate_fire.toml`, with DOI-backed schema provenance |
| `adaptive_threshold_if_tonic_adaptation_doi` | `adaptive_threshold_if` | `universal_dsl` | Independent exact constant-input relaxations with candidate-crossing reset and fixed post-spike threshold shift from `neurons/model_schemas/adaptive_threshold_if.toml`, with composite Mihalas-Niebur/Platkiewicz-Brette DOI provenance |
| `alpha_dual_synapse_doi` | `alpha` | `universal_dsl` | Independent exact piecewise-constant-input alpha-filter relaxation and alpha-current convolution with somatic-only reset from `neurons/model_schemas/alpha.toml`, with Rall 1967 and Gerstner-Kistler DOI provenance |
| `rulkov_map_driven_spiking_doi` | `rulkov_map` | `universal_dsl` | Independent piecewise-map iteration of Rulkov 2002 Equations 1–2 with `method="map"` and the source pre-update rightmost/reset-branch event, from `neurons/model_schemas/rulkov_map.toml` with DOI-backed provenance |
| `theta_constant_current_phase_analytic` | `theta` | `universal_dsl` | Analytic tangent half-angle phase solution from `neurons/model_schemas/theta.toml` with DOI-backed schema provenance |
| `wang_buzsaki_driven_spiking_doi` | `wang_buzsaki` | `universal_dsl` | Independent macro-step Gauss-Seidel re-derivation of the driven fast-spiking interneuron (50 inner `dt=0.01` sub-steps per 0.5 ms macro step, gates `h`/`n` updated before `v`, no reset, macro-boundary `v >= v_threshold` crossing) from `neurons/model_schemas/wang_buzsaki.toml` with DOI-backed schema provenance |

## Seeded stochastic reference

`escape_rate_lfsr16_statistical_v1.json` is a separate statistical artifact,
validated by `tests/test_reference_escape_rate.py`. Its independent recurrence
does not import the production RNG helper: it re-evaluates the documented
right-shift LFSR16 polynomial, performs eight primitive advances per logical
trial, applies the 17-bit probability threshold, and hashes the resulting event
bytes.

The full-period protocol holds the voltage and escape intensity constant with
`rho*dt=0.25`. Across all 65,535 non-zero LFSR states it records exactly 14,496
events, final state `0xACE1`, mean inter-event interval
`4.520869265263884`, CV `0.8842846076062356`, and event SHA-256
`6f118617f2ecb7a54c5a7ca68ee38a80a68dd15494e361c77aa228397614bfa8`.
The same artifact pins 4,096-step event hashes, counts, and final RNG states for
five seeds: `1`, `42`, `0xACE1`, `0xBEEF`, and `0xFFFF`.

Gerstner (2000), Eqs. (2.13)–(2.15), supplies the conditional intensity,
survival function, and firing-time density; DOI
`10.1162/089976600300015899` anchors that source. The exact RC step,
piecewise-constant finite-step hazard transform, LFSR polynomial and
decimation, comparator quantisation, and default seed are explicitly maintained
SC-NeuroCore conventions. This seeded artifact extends the evidence corpus; it
is not presented as a deterministic `UniversalNeuron` feature trace.

`poisson_lfsr16_statistical_v1.json` separately binds the homogeneous Poisson
source to the same hardware sampler. It independently computes
`p=1-exp(-rate_hz*dt_ms/1000)`, then re-evaluates the LFSR and comparator over
the complete period. At 250 Hz with 1 ms bins and seed `0xACE1`, it records the
same exact 14,496-event vector and distribution features because the interval
hazard is also 0.25. The artifact additionally pins the comparator threshold
14,497, continuous and realised probabilities, first and last event indices,
final RNG state, and the same five-seed corpus. Its test compares the independent
result with the hand `PoissonNeuron`, paired TOML/JSON `UniversalNeuron`
surfaces, and every native backend.

Gerstner, Kistler, Naud, and Paninski (2014), Sections 7.2 and 7.7, supply the
homogeneous process, exponential waiting-time law, and finite-interval event
probability; DOI `10.1017/CBO9781107447615` anchors that source. Binary-bin
collapse, polynomial, decimation, comparator quantisation, and replay seed are
explicit SC-NeuroCore conventions. The artifact is statistical seeded evidence,
not a deterministic scalar-feature trace or an external-simulator claim.

All entries record spike count, first spike step, and final/min/max/mean
features for the declared state variables. The tests independently recompute the
LIF, QIF, IQIF, McCulloch-Pitts, perfect-integrator, resonate-fire, theta, Ermentrout-Kopell theta-Euler, GLIF, Izhikevich, Cazelles map, Chialvo map, Aihara map, Ibarz-Tanaka map, Medvedev map, Courbage-Nekorkin map,
Izhikevich 2007, FitzHugh-Nagumo, FitzHugh-Rinzel, Pernarowski, Terman-Wang, Wilson-HR, McKean, Lapicque, AdEx, exponential-IF,
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
relaxation oscillator, not integrate-and-fire); and the DPI entry independently
advances the coupled membrane and after-hyperpolarisation currents from the 2010
source equations, including the nonlinear feedback gate, threshold reset,
refractory pulse, and all three post-step states over the 13-event driven
protocol.
The FitzHugh-Rinzel entry extends that independent cubic RK4 recurrence with the
ultra-slow `y` modulation equation. It advances `v`, `w`, and `y` simultaneously,
uses the same no-reset upward-crossing decision, and reproduces all three state
feature sets plus the eight-crossing `I=0.5` protocol without calling the hand
model or schema runner.
The Hindmarsh-Rose entry independently advances its cubic fast membrane,
recovery state, and slow adaptation state with simultaneous classical RK4. It
uses the maintained no-reset upward `x >= 1` crossing observation and reproduces
all three feature sets, the first crossing at step 114, and the 26-crossing
`I=3` protocol without calling the hand model or schema runner.
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
recovery coordinates with source `C=0.8` and simultaneous classical RK4. It
observes sampled upward `v=0` crossings without resetting the continuous state.
The re-derivation reproduces both state feature sets, the first crossing at
step 15, and all 46 crossings of the 5,000-step `I=0.1` protocol without
calling the hand model or schema runner. The separately named
`SCResettingWilsonHRNeuron` retains the former unit-capacitance, level-detected,
hard-reset project recurrence under a project-only compatibility anchor rather
than reusing the Wilson publication receipt.
The Mihalas-Niebur entry re-derives equations 2.1–2.2 and Table 1 directly:
two exponentially decaying currents, the capacitance-normalised membrane and
adaptive-threshold flows, and the event map `I_j = R_j*I_j + A_j`, `V = V_r`,
`Theta = max(Theta_r, Theta)`. The separately named
`SCScaledResetAdaptiveIFNeuron` retains the former candidate-proportional
voltage-reset recurrence under a project-only compatibility anchor.
The Benda-Herz entry re-derives equations (8) and (45) with the paper's Figure 8
square-root/linear example and validates the complete deterministic adaptation,
phase, and event receipt. The former stochastic project recurrence is separately
identified as `SCStochasticRateAdaptationNeuron` and carries no paper attribution.

The McKean entry re-derives the exact classical fourth-order Runge-Kutta recurrence
for its three-branch piecewise-linear membrane `f(v) = min(max(-v, v - a), 1 - v)` and
linear recovery, with rising-edge `v >= v_peak` crossing detection and no reset; at the
enrolled sustained-oscillation regime (`epsilon = 0.2`, `gamma = 0.5`, `I = 0.6`) it is a
robust limit cycle whose sixteen upward crossings survive Q16.16 rounding, so the
min/max branch selection lowers to fixed point without a look-up table.
The Lapicque entry independently evaluates
`v(t)=v_inf+(v0-v_inf)*exp(-t/tau)` for its 200-sample subthreshold protocol,
without importing the hand model or schema recurrence. It reproduces every
committed voltage feature within `1e-12`; event count and first-event sentinel
remain exact. The provenance points to the English translation DOI while the
exact-flow discretisation is stated as the maintained implementation contract.
The Medvedev entry independently reconstructs the Section 4 slow-calcium return
from its three source regions and the disclosed SC-NeuroCore global calibration.
Its `I=2`, 100-iteration protocol reproduces the exact four-state cycle and 75
maintained pre-state events without importing the hand model or schema expressions.
The reference explicitly records that non-zero current and event labelling are
maintained conventions rather than equations or spike semantics asserted by the paper.
The Ibarz-Tanaka entry independently evaluates all four fast-map branches and
the simultaneous slow update from Eqs. 2–3. Its zero-current 1,000-iteration
protocol reproduces nine source reset events, the first at step 395, and all
committed `v`/`u` features without importing the hand model or schema runner.
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
The perfect-integrator, Ermentrout-Kopell theta-Euler, FitzHugh-Nagumo, FitzHugh-Rinzel, Pernarowski, Terman-Wang, Wilson-HR, Rulkov, Cazelles, Chialvo, Ibarz-Tanaka, Medvedev, Courbage-Nekorkin, McKean, Morris-Lecar,
Hodgkin-Huxley, Connor-Stevens,
Izhikevich, Izhikevich 2007, DPI, and Mihalas-Niebur entries are spike-bearing;
they validate reset (or, for Ermentrout-Kopell theta-Euler, FitzHugh-Nagumo, FitzHugh-Rinzel, Pernarowski, Terman-Wang, McKean, Morris-Lecar,
Hodgkin-Huxley, and Connor-Stevens, rising-edge crossing) and first-spike features,
not only quiet trajectories. The
Rulkov entry iterates the Rulkov 2002 piecewise fast/slow map with the
`method = "map"` integration mode (`x_{n+1} = f(x_n, y_n)`, iterated as a discrete
map rather than integrated as an ODE), so the trajectory is bounded and its
committed features are independently re-derived exactly; a driving current
exercises all three fast-map branches (rational subthreshold, spike plateau, hard
reset). Its event marks pre-update occupancy of the rightmost branch that commits
the hard reset; it is not a rising crossing or a positive-level count. The former
upward-crossing convention remains a separate count-neutral SC identity with its
own project receipt and is not substituted into this DOI-backed record. The Cazelles entry independently iterates
the simultaneous clipped logistic fast/slow map at `I=0.5`; the 30-step window exercises interior
and lower-clip branches and records two level events. The reference is deliberately bounded because
the `a=3.8` fast map amplifies fixed-point perturbations on longer chaotic trajectories. The
Chialvo entry independently iterates the DOI-sourced exponential two-state map for 100 iterations,
records two maintained upward crossings with the first at iteration 33, and derives both state
feature sets without calling the hand model or schema runner. The threshold observation is
separated explicitly from the paper's recurrence. The Courbage-Nekorkin entry independently iterates the published three-branch fast map and recovery
recurrence for 30 autonomous iterations, including the Heaviside discontinuity and upward event
crossing. It records four events and features for both coordinates without importing the hand model
or schema expressions. The Ermentrout-Kopell entry independently advances the DOI-sourced theta
flow with the maintained forward-Euler parameter, judges the unwrapped candidate against the
pre-step phase, and only then reduces the committed phase modulo `2*pi`. Its `I=0.5`, 2,000-step
protocol records 45 events and the scalar phase feature set without calling the hand model or schema
expressions. The QIF and older `theta` tolerances are wider than
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
PYTHONPATH=src python -m pytest \
    tests/test_reference_traces.py \
    tests/test_reference_trace_payloads.py \
    tests/test_reference_ermentrout_kopell_map_neuron.py \
    tests/test_reference_medvedev_map.py -q
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
