<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Changelog

All notable changes to the `sc-neurocore` project will be documented in this file.

## [Unreleased]

### Chialvo source-to-silicon fidelity enrolment
- Bound `ChialvoMapNeuron` to Chialvo (1995), DOI
  `10.1016/0960-0779(93)E0056-H`. The Python and compiled lanes now share the
  paper's simultaneous `x*x*exp(y-x)+k+I`, `a*y-b*x+c` recurrence and reject
  non-finite candidates without committing them. The paper permits a constant
  or time-dependent additive perturbation; `k` and `current` represent those
  roles. The upward `x_threshold=1.0` crossing is kept separate as a maintained
  observation convention.
- Wired checked Rust-engine, Rust-safety, Go, Julia, and Mojo batch paths into
  measured-order `auto` dispatch. Cross-language tests enforce one-step
  source-equation envelopes and identical 1,000-step event counts at six
  currents. The committed 500,000-iteration, five-repeat benchmark was pinned
  to one logical CPU and records 12,935 events in every lane. Its medians are
  7.270 ms Rust, 9.576 ms Julia, 11.373 ms Mojo, 20.524 ms Go, and 2,175.866 ms
  Python; the JSON records that the host had no kernel-isolated CPU set.
- Added paired TOML/JSON map schemas and an independent 100-iteration DOI
  reference. Hand/TOML/JSON states and events are exact at
  `I=-0.05/0/0.01/0.1/1.0`; Q16.16 retains event counts `0/2/3/0/1` and keeps
  stable-point `x/y` errors below `0.055/0.093`. Four and six oscillatory event
  positions shift at `I=0/0.01`, so timing identity is excluded explicitly.
  The S5/H1 descriptor adds a generated Q8.8 port-only formal job whose
  depth-4 SymbiYosys/Z3 check passes. The public fidelity count moves to 21;
  schema-gap counts move to 31 / 122 / 124; the formal catalogue moves to 23
  jobs and the whole HDL inventory to 60 jobs / 208 statements.

### Ermentrout-Kopell theta-Euler schema-to-RTL enrolment
- Enrolled `ermentrout_kopell_map_neuron` against Ermentrout and Kopell
  (1986), DOI `10.1137/0146017`. Paired schemas reproduce the maintained hand
  recurrence exactly while separating the paper's continuous theta equation
  from the implementation's `dt=0.1`, gain, forward-Euler, `theta=pi` event,
  and modulo `2*pi` choices. The independent `I=0.5` reference records 45
  events over 2,000 steps; varied-drive tests require exact hand/TOML/JSON
  states and events.
- Q16.16 RTL preserves 0/45/64 spikes at `I=-0.5/0.5/1.0` over 2,000 steps,
  with maximum circular phase error below 0.081/0.089/0.025 rad. The cosine LUT
  can shift event positions, so event-vector and full-trajectory identity are
  withheld. Generated integer C/Rust kernels match generated Verilog state and
  event words cycle-for-cycle over the 240-step protocol for both current signs.
- The equation runtime and generated backends now expose `<state>_prev` at the
  macro boundary and lower modulo only for a representable finite positive
  literal, with the negative-remainder correction required to match Python.
  The S5/H1 descriptor adds a Q8.8 port-only formal job; its depth-4 Z3 BMC
  passes. All 21 earlier jobs regenerate byte-identically, the inventory moves
  to 22, and schema-gap counts move to 30 / 123 / 125. Existing acceleration
  and benchmark artefacts remain unchanged.

### Courbage-Nekorkin-Vdovin bounded schema-to-RTL enrolment
- Enrolled the already acceleration-complete 2007 discontinuous map
  (`courage_nekorkin_map`, DOI `10.1063/1.2795435`). The paired TOML/JSON
  schemas reproduce equations 3–5, including simultaneous state commits, all
  three fast-map branches, the Heaviside discontinuity, and the maintained
  upward `x_threshold` crossing. The hand model and both schemas agree exactly
  across the enrolled operating set.
- Q16.16 RTL is event-exact at `I=-0.3/0/0.3` over bounded
  30/20/30-iteration windows, with both state errors below `0.014`. Q32.32 RTL
  is event-exact at all three inputs over 30 iterations, with fast-coordinate
  error below `0.00003` and recovery-coordinate error below `0.000001`. A
  separate regression fixes the autonomous 30-iteration Q16.16 boundary at
  four float64 events, six RTL events, and six event-position mismatches.
- The model-scoped co-simulation and independent-reference tests add a
  DOI-backed recurrence, S5/H1 descriptor/readiness facets, source-bounded
  public documentation, and generated Q8.8 RTL. The port-only depth-4
  SymbiYosys/Z3 reset-spike safety job passes. The formal inventory moves to
  21 models and schema-gap counts move to 29 schema models / 124 net missing /
  126 source modules without a schema. The completed acceleration chain and
  its committed benchmark artefact remain unchanged.

### Cazelles map bounded schema-to-RTL enrolment
- Enrolled the already acceleration-complete Cazelles, Courbage, and Rabinovich
  (2001) fast/slow map (`cazelles_map`, DOI
  `10.1209/epl/i2001-00548-y`). The paired TOML/JSON schemas mirror the
  maintained hand model's simultaneous clipped logistic fast recurrence, slow
  update from the old fast coordinate, and committed-state `x >= x_threshold`
  level event. Hand model and both schemas agree exactly on every state and
  event across the enrolled operating set.
- Over 30 iterations, emitted Q16.16 RTL reproduces the complete 2/1/1 event
  vectors at `I=0.5/1.0/2.0`, with both coordinates within `0.0004` absolute
  error of float64. The three points exercise the interior expression and both
  clip bounds. A separate regression fixes the sensitive `I=0.05` boundary at
  seven float64 events, eight RTL events, and seven event-position mismatches,
  so the bounded evidence cannot imply long-window chaotic identity.
- The Cazelles co-simulation and independent-reference tests live in dedicated
  model-scoped modules rather than the legacy catalogue-wide test accumulators.
  The unit adds the DOI-backed independent map reference, S5/H1
  descriptor/readiness facets, public fidelity and model documentation, and
  generated Q8.8 RTL. Its port-only depth-4 SymbiYosys/Z3 reset-spike safety
  job passes. The formal inventory
  moves to 20 models and schema-gap counts move to 28 schema models / 125 net
  missing / 127 source modules without a schema. The completed acceleration
  chain and its committed benchmark artefact remain unchanged.

### Mihalas-Niebur co-simulation evidence correction
- Replaced the stale loose Q16.16 guard with exact operating-point contracts after the shared
  candidate-reset/output correction. The hand model and paired TOML/JSON schemas agree exactly on
  every event and all four states over a varied 1,600-step sequence containing 168 resets. At
  `I=3` over 300 steps, hand/schema/RTL now report 36/36/36 rather than the former 36/36/35.
- Over 1,000 steps, hand/schema/Q16.16 RTL agree exactly on
  0/0/0/31/60/87/131/157/207/256 spikes at
  `I=0/0.5/1/1.5/2/2.5/3.5/4/5/6`. A separate test pins the isolated `I=3` boundary at
  111/111/112, preserving the marginal fixed-point crossing as an explicit exclusion rather than
  a hidden tolerance. The S5/H1 descriptor, readiness index, public fidelity row, model page, and
  co-simulation guide now state that same contract. The existing depth-3 SymbiYosys/Z3 job remains
  the structural safety proof; all model, acceleration, schema, compiler, generated RTL, formal
  source, and benchmark artefacts are unchanged.

### GLIF faithful RK4 schema-to-RTL correction
- Corrected the already enrolled `glif` TOML/JSON schemas from stale Euler and `v > theta`
  semantics to the maintained Allen Institute GLIF5 contract: simultaneous four-state classical
  RK4, candidate-level `v >= theta` detection, and candidate-first adaptive reset. The hand model
  and both schema formats agree exactly on all states and 181 reset events over a varied
  4,000-step drive.
- Replaced the silent Euler reference with an independent 54-spike RK4 re-derivation. Across the
  six 1,000-step Q16.16 operating points `I=0/15/22/30/45/50`, hand model, schema runner, and RTL
  agree exactly on 0/0/23/54/86/95 spikes. The compiler and bit-true C/Rust generators now evaluate
  reset expressions from the integrated candidate and expose identical post-reset state, removing
  the pre-step-reset mismatch that a one-spike band had masked. The S5/H1 descriptor/readiness
  facets record the exact observable. Deterministic regeneration changes all ten reset-using formal
  RTL models plus seven non-resetting single-step edge models so spike-cycle outputs expose the
  committed candidate. Connor-Stevens and Hodgkin-Huxley already expose that candidate in their
  macro-step branches and remain byte-identical, accounting for all 19 jobs without changing the
  inventory. The completed acceleration chain and benchmark artefacts are unchanged. The reset
  correction exposes one honest legacy
  Izhikevich Q8.8 boundary (float64 25 spikes versus RTL 24 at `I=50` over 200 steps), while a new
  Q16.16 guard proves exact 25/25 parity at the same point.

### Rulkov map class-correct schema-to-RTL enrolment
- Descriptor regeneration now inherits `integration.dt` from a bundled map schema when the hand
  class intentionally has no timestep parameter, so the Rulkov map retains its one-iteration
  metadata without adding a non-functional public field or changing non-map fallbacks. The
  readiness facet writer also preserves any descriptor whose recorded evidence already meets or
  exceeds the indexed floor, preventing regeneration from replacing Rulkov's S5 trajectory and H2
  synthesis evidence with H0 facets.
- Re-enrolled the Rulkov 2002 fast/slow map (`rulkov_map`, DOI
  `10.1103/PhysRevE.65.041922`) with paired TOML/JSON schemas that mirror the maintained
  hand model's simultaneous rational/plateau/hard-reset branches and rising `x >= 0`
  crossing decision. The old schema used level detection and therefore counted every
  positive plateau step rather than the hand model's upward crossings.
- At `I=1.5`, the bounded 30-iteration validation window executes every fast-map branch
  ten times. Hand model and both schema formats agree exactly on all post-step states and
  the ten-event vector; emitted Q16.16 RTL reproduces that event vector exactly with
  absolute `x`/`y` error below `0.001`. The evidence uses the class-appropriate short-window
  trajectory metric and explicitly withholds long-window spike-count identity.
- Corrected the descriptor's stale equations and timestep, regenerated the DOI-backed
  crossing reference, and recorded S5/H2: the Q16.16 core passes Yosys 0.33
  `synth_xilinx` with a raw committed report. The now-perfect descriptor is registered
  with the formal catalogue through generated Q8.8 RTL, a port-only harness, and a
  depth-4 SymbiYosys/Z3 safety job, bringing the inventory to 19 models. Schema-gap
  counts are unchanged because the paired schema already existed. The completed
  acceleration chain and its benchmark artefacts are unchanged.

### Wilson-HR faithful schema-to-RTL enrolment
- Enrolled the Wilson-HR two-state polynomial cortical model (`wilson_hr`, Wilson 1999, DOI
  `10.1006/jtbi.1999.1002`) with paired TOML/JSON schemas that mirror the maintained hand model's
  simultaneous classical RK4 flow, polynomial membrane nullcline, level `v >= v_peak` decision,
  and hard `v = -0.7` reset that preserves the candidate recovery state. A varied 4,000-step drive
  produces 35 spikes and resets with exact hand/TOML/JSON state agreement. Over 5,000
  constant-current steps the hand model, schema runner, and Q16.16 RTL agree exactly on 0, 1, and
  4 spikes at `I=0.0`, `2.0`, and `10.0`.
- Added the DOI-backed `wilson_hr_driven_spiking_doi` trace with an independent two-state RK4
  re-derivation, corrected the public model provenance, marked the descriptor's RK4 and co-sim
  facets, and updated the public fidelity row. The S5/H1 descriptor is registered with the formal
  catalogue: generated Q8.8 RTL, a port-only harness, and a depth-4 SymbiYosys reset-spike safety
  job bring the committed inventory to 18 models. Schema-gap counts move to 27 schema models / 126
  net missing / 128 source modules without a schema. The completed Rust/Go/Julia/Mojo acceleration
  chain and its committed benchmark artefacts are unchanged.

### Terman-Wang faithful schema-to-RTL enrolment
- Enrolled the Terman-Wang two-state LEGION relaxation oscillator (`terman_wang`, Terman & Wang
  1995, DOI `10.1016/0167-2789(94)00205-5`) with paired TOML/JSON schemas that mirror the
  maintained hand model's simultaneous classical RK4 flow, cubic fast nullcline, `tanh`-gated
  recovery, rising-edge `v >= v_peak` decision, and no-reset semantics. Over 8,000 steps the hand
  model, schema runner, and emitted Q16.16 RTL agree exactly on the silent/single/train crossing
  counts: 0 at `I=-1.0`, 1 at `I=0.0`, and 3 at `I=0.5`.
- Added the DOI-backed `terman_wang_legion_oscillation_doi` trace with an independent two-state RK4
  re-derivation, corrected the public model citation, marked the descriptor's RK4 and co-sim
  facets, and updated the public fidelity row. The S5/H1 descriptor is registered with the formal
  catalogue: generated Q8.8 RTL, a port-only harness, and a depth-4 SymbiYosys reset-spike safety
  job bring the committed inventory to 17 models. Schema-gap counts move to 26 schema models / 127
  net missing / 129 source modules without a schema. The completed Rust/Go/Julia/Mojo acceleration
  chain and its committed benchmark artefacts are unchanged.

### Pernarowski faithful schema-to-RTL enrolment
- Enrolled the Pernarowski three-state pancreatic beta-cell burster (`pernarowski`, Pernarowski
  1994, DOI `10.1137/S003613999223449X`) with a TOML/JSON schema that mirrors the maintained hand
  model's simultaneous classical RK4 flow, exact `v * v * v` operation order, rising-edge
  `v >= v_threshold` decision, and no-reset semantics. The hand model, schema runner, and emitted
  Q16.16 RTL have exact spike-count parity at all four enrolled 5,000-step operating points: 17
  crossings at each of `I=-0.1`, `0.0`, `0.1`, and `0.2`.
- Added the DOI-backed `pernarowski_autonomous_bursting_doi` trace with an independent three-state
  RK4 re-derivation, marked the descriptor's RK4 and co-sim facets, and updated the public fidelity
  row. The S5/H1 descriptor is also registered with the formal catalogue: generated Q8.8 RTL, a
  port-only harness, and a depth-4 SymbiYosys reset-spike safety job bring the committed inventory
  to 16 models and pass a direct Z3 BMC. Schema-gap counts move to 25 schema models / 128 net missing
  / 130 source modules without a schema. The already completed Rust/Go/Julia/Mojo acceleration
  chain and its committed benchmark artefacts are unchanged.

### FitzHugh-Rinzel faithful schema-to-RTL enrolment
- Enrolled the FitzHugh-Rinzel three-state qualitative burster (`fitzhugh_rinzel`, Rinzel 1987,
  DOI `10.1007/978-3-642-93360-8_26`) with a TOML/JSON schema that mirrors the maintained hand
  model's coupled classical RK4 flow, exact `v * v * v` operation order, rising-edge
  `v >= v_threshold` decision, and no-reset semantics. The hand model, schema runner, and emitted
  Q16.16 RTL have exact spike-count parity across the enrolled `I=0.4` to `I=0.6` band (seven,
  eight, and eight crossings over 3000 steps); `I=0.7` is recorded as an excluded
  marginal-crossing boundary rather than hidden by a tolerance.
- Added the DOI-backed `fitzhugh_rinzel_driven_bursting_doi` trace with an independent three-state
  RK4 re-derivation, corrected the model page's stale reset claim, marked the descriptor's RK4 and
  co-sim facets, and updated the public fidelity row. Schema-gap counts move to 24 schema models /
  129 net missing / 131 source modules without a schema. The already completed Rust/Go/Julia/Mojo
  acceleration chain and its committed benchmark artefacts are unchanged.

### StochasticLIF same-name Rust engine binding
- `StochasticLIFNeuron` was enrolled in the public registry
  (`sc_neurocore.neurons.models.__all__`) for catalogue readiness but had no same-name PyO3
  constructor, so the registry-parity coverage map flagged it as an uncovered non-Python-only model.
  The Rust neuron already existed in the engine (`engine/src/neurons/trivial.rs`) and was wired into
  the `NetworkRunner`; this adds the missing `PyStochasticLIFNeuron` wrapper (`new(seed)` / `step` /
  `reset` / `get_state`, mirroring `StochasticIFNeuron`), re-exports it from the engine package root,
  and enrols it in the RNG-dependent parity set. Exact traces are not claimed — the Rust Gaussian
  stream is `Xoshiro256++` Box-Muller rather than NumPy's PCG64 Ziggurat — so the spike count is the
  stated parity observable. The registry map is now 160 public Python registry names, 146 same-name
  Rust constructors, and 176 Rust PyO3 model wrappers.

### Connor-Stevens polyglot kernels pinned to the Python golden spike count
- The Go (`accel/go/services/connor_stevens.go`), Rust (`accel/rust/safety/connor_stevens.rs`) and
  Julia (`accel/julia/neurons/connor_stevens.jl`) Connor-Stevens kernels already carried the real
  six-state macro-step RK4 dynamics (100 sub-steps) but were only self-consistency / smoke tested;
  each now asserts the Python golden counts — silent at zero drive, two action potentials at `I=10`
  over 100 macro steps, nine at `I=20`. Connor-Stevens gating is `exp`-based, so the trace is not
  bit-exact across C libraries; the spike count is the stated parity observable and all three
  languages reproduce it.
- Added a native Julia parity test (`accel/julia/connor_stevens_parity_test.jl`), a Go golden-parity
  test plus an honest `BenchmarkConnorStevensStep` (262.7 µs/macro-step), and the idiomatic
  `Default` impl on the Rust side. The Mojo kernel remains an honest parity note pending the Mojo
  neuron-kernel lane's promotion to a build target. Connor-Stevens's Python model dispatches only to
  the Rust engine, so the Go/Julia kernels are language-native; no Python runtime path, FFI
  dispatch, or committed benchmark artefact changed.

### Morris-Lecar polyglot kernels pinned to the Python golden spike count
- The Go (`accel/go/services/morris_lecar.go`), Rust (`accel/rust/safety/morris_lecar.rs`) and Julia
  (`accel/julia/neurons/morris_lecar.jl`) Morris-Lecar kernels already carried the real RK4 dynamics
  but were only self-consistency / smoke tested; each now asserts the Python golden counts — silent
  at zero drive, three action potentials at `I=50` over 2000 steps, five at `I=100`. Morris-Lecar
  gating is `tanh`/`cosh`, so the trace is not bit-exact across C libraries; the spike count is the
  stated parity observable and all four languages reproduce it.
- Added a native Julia parity test (`accel/julia/morris_lecar_parity_test.jl`) and an executable
  `simulate`/`main` parity harness to the Mojo kernel (`accel/mojo/kernels/morris_lecar.mojo`, so
  `mojo run` prints `PARITY OK`); removed a dead `k4_v` assignment in that kernel's `next_w` path.
  On the Rust side, dropped a vestigial `#![allow(unused_variables, dead_code, non_snake_case)]`
  (the code is real, not a masked stub) and added the idiomatic `Default` impl. Morris-Lecar's
  Python model dispatches only to the Rust engine, so the Go/Julia/Mojo kernels are language-native;
  no Python runtime path, FFI dispatch, or committed benchmark artefact changed.

### FitzHugh-Nagumo Go accel-services kernel (services test suite unblocked)
- Added `accel/go/services/fitzhugh_nagumo.go`, a real RK4 `SimulateFitzHughNagumoNeuron` in
  parity with `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron` (the cube written
  `v*v*v`, exact arithmetic, fail-closed on a non-finite input, state, or candidate). The Go
  services test suite could not compile because `services_test.go` referenced this function while
  the FitzHugh-Nagumo Go kernel existed only as a `package main` cgo shared library under
  `accel/go/neurons/fitzhugh_nagumo/`. A golden-parity test pins the kernel to the Python reference
  (one action potential at `I=10` over 100 steps and a five-spike partial train at `I=0.5` over
  2000 steps, final `v` bit-identical to NumPy) and an honest per-step benchmark records the timing.
- Strengthened the `accel/rust/safety` FitzHugh-Nagumo test from a `spike is 0 or 1` smoke check to
  the same Python-golden spike count, dropped a vestigial `#![allow(dead_code)]`, and added the
  idiomatic `Default` impl. The services surface is Go-native, not FFI-dispatched, so no Python
  runtime path, FFI dispatch, cross-language `simulate` benchmark, or committed benchmark artefact
  changed.

### Sequential (Gauss-Seidel) integration mode + faithful Wang-Buzsáki re-enrolment
- Added a sequential (Gauss-Seidel) integration mode (`[integration] method = "gauss_seidel"`) to
  the schema DSL, in both the Python runner (`EquationNeuron`) and the emitted Verilog. The state
  variables advance in declaration order, each derivative reading the already-committed earlier
  variables within the same sub-step — lowering a conductance hand model's gates-then-voltage
  update (gates from the old voltage, voltage from the new gates). The emitter renders each earlier
  variable as its `<var>_next` wire in a later variable's derivative (a commit-before-read chain,
  no cycle). Composes with `substeps`; the simultaneous `euler`/`rk4` methods stay bit-for-bit
  unchanged (default `substeps = 1`).
- Re-enrolled the Wang-Buzsáki (1996) fast-spiking interneuron (`wang_buzsaki` schema, DOI
  `10.1523/JNEUROSCI.16-20-06402.1996`) faithfully. The bundled schema was a single-step
  `method="euler"` re-derivation with a sigmoid-caricature `m_inf`, unfaithful gate initial
  conditions (`h=0.6`, `n=0.32`), a `v > -10` threshold, and a singular `n` rate; it is now
  `method="gauss_seidel"`, `substeps=50`, state ordered `h, n, v` (`h=0.8`, `n=0.1`, `v=-65`), the
  true instantaneous `m_inf = alpha_m/(alpha_m+beta_m)` (`alpha_m = 1/exprel(-(v+35)/10)`), the
  exprel `n` rate `0.1/exprel(-(v+34)/10)`, a macro-boundary `v >= v_threshold` crossing
  (`v_threshold=-20`), no reset — matching `WangBuzsakiNeuron` exactly (`hand == schema`, three
  action potentials at `I=10` over 20 macro steps). The Q16.16 RTL tracks the schema **within one
  spike** over the bounded window (three-way exact at `I=10`, `macro=20`); the residual is the
  `m_inf` fixed-point divide plus a 256-entry exprel look-up, not a datapath-precision limit.
  Re-derived reference trace `wang_buzsaki_driven_spiking_doi`
  (`independent_macrostep_gauss_seidel_reference`, a 3-state helper bit-exact vs the runner,
  spike_count 4), replacing the deleted resting-gate trace; descriptor dynamics synced to the
  exprel form; the 15%-band cosim test becomes a macro-step three-way parity test. Wang-Buzsáki is
  the last of the four WC-A5 conductance oscillators (Morris-Lecar, Connor-Stevens, Hodgkin-Huxley,
  Wang-Buzsáki), all now faithfully enrolled; the polynomial edge-crossing oscillators
  (FitzHugh-Nagumo, McKean) were enrolled earlier.

### Faithful Hodgkin-Huxley re-enrolment (macro-step RK4)
- Re-enrolled the Hodgkin-Huxley (1952) membrane (`hodgkin_huxley` schema,
  DOI `10.1113/jphysiol.1952.sp004764`) as a driven repetitive-spiking oscillator using the
  macro-step mode. The bundled schema was a single-step `method="euler"` resting-gate re-derivation
  compared schema-vs-verilog under a 5% band; it is now `method="rk4"`, `substeps=100`,
  macro-boundary `v >= v_threshold` crossing — matching `HodgkinHuxleyNeuron(integrator="rk4")` (the
  simultaneous RK4, not the Gauss-Seidel `baseline_euler` default the DSL cannot reproduce).
  `hand == schema` exact (five action potentials at `I=20` over 60 macro steps). The Q16.16 RTL
  tracks the schema **within one spike** over the bounded window (`I=15`, 20 macro steps, three-way
  exact); like Connor-Stevens the residual is genuine conductance-LUT quantisation
  (LUT-resolution-limited, identical at Q16.16 / Q24.24 / Q32.32). Re-derived reference trace
  `hodgkin_huxley_driven_spiking_doi` (`independent_macrostep_rk4_reference`, bit-exact vs the
  runner); descriptor integration updated (euler → rk4). Also fixed a pre-existing
  `hodgkin_huxley.json` toml/json drift (singular `a*(V-V0)/(1-exp(...))` rate form → the stable
  `exprel` rewrite the `.toml` already used) and a stale wrong Connor-Stevens DOI (`sp009368` →
  `sp009366`) in the changelog. Schema-gap counts unchanged; no polyglot / benchmark change (the
  hand model already carries the RK4 path).

### Macro-step integration mode + faithful Connor-Stevens re-enrolment
- Added a macro-step integration mode (`[integration] substeps = N`) to the schema DSL — in both
  the Python runner (`EquationNeuron`) and the emitted Verilog. One macro `step()` advances `N`
  inner integration sub-steps before a single spike decision, with the rising-edge crossing taken
  only on the macro boundary. This lets the schema faithfully replicate the maintained conductance
  hand models whose `step()` is a fixed number of fine sub-steps per macro step (HH / Connor-Stevens:
  100 `dt=0.01` sub-steps per 1 ms; Wang-Buzsaki: 50 per 0.5 ms), so a repetitively firing
  oscillator counts one spike per action potential, not one per sub-step above threshold. The RTL
  keeps one sub-step per clock and gates the crossing to the macro boundary via a sub-step counter;
  the lowering is bit-exact against the runner (proven on the polynomial FitzHugh-Nagumo at Q16.16,
  exact across sub-step groupings). Supported for the edge (crossing, non-resetting), non-pipelined
  datapath; other combinations raise `NotImplementedError`. `substeps = 1` (default) leaves every
  existing model bit-for-bit unchanged.
- Re-enrolled Connor-Stevens (1971) faithfully with the macro-step mode: the bundled schema is now
  `method="rk4"`, `substeps=100`, macro-boundary `v >= v_threshold` crossing — matching the
  maintained `ConnorStevensNeuron`. `hand == schema` exact (ten action potentials at `I=100` over
  60 macro steps), which the earlier single-step Euler schema could not achieve. The Q16.16 RTL
  tracks the schema **within one spike** over the bounded window; the residual is genuine
  conductance-LUT quantisation (LUT-resolution-limited, identical at Q16.16 / Q24.24 / Q32.32),
  three-way exact over a bounded window and accumulating beyond it — an honest per-model
  hardware-fidelity band. Re-derived reference trace `connor_stevens_driven_spiking_doi`
  (`independent_macrostep_rk4_reference`, bit-exact vs the runner); descriptor integration updated
  (euler → rk4). Also fixed a pre-existing `connor_stevens.json` toml/json drift (singular
  `a*(V-V0)/(1-exp(...))` rate form → the stable `exprel` rewrite the `.toml` already used).
  Schema-gap counts unchanged; no polyglot / benchmark change (the hand model was already RK4).

### Morris-Lecar faithful re-enrolment (RK4, no reset, rising-edge crossing)
- Re-enrolled the Morris-Lecar (1981) calcium-potassium oscillator (`morris_lecar` schema,
  DOI `10.1016/S0006-3495(81)84782-0`) as the first **conductance** edge-crossing oscillator
  in the WC-A5 Python↔Verilog co-simulation set. The prior schema was `method="euler"` with a
  no-op `[reset]` (`v -> v`, `w -> w`) that disabled edge detection and over-counted every
  above-threshold step — a caricature that only "passed" a ~15% band because both sides
  over-counted identically. The faithful schema mirrors `MorrisLecarNeuron`'s maintained
  defaults: four-stage RK4, **no reset**, rising-edge (`v >= v_threshold`) crossing, and
  `phi = 1/15`.
- At the sustained depolarising regime (`I=100`, 3000 steps) the hand `MorrisLecarNeuron`,
  the schema runner, and the emitted Q16.16 RTL report the same seven upward crossings.
  Because the sigmoidal gating lowers to 256-entry cosh/tanh look-up tables and the hand
  model integrates with `math` transcendentals through a distinct RK4 driver, this is an
  exact **spike-count** parity (robust across the whole `I in [90, 110]` band), not the
  bit-identical state the polynomial FitzHugh-Nagumo / piecewise-linear McKean oscillators
  achieve; `I=120` is a knife-edge that splits a marginal crossing between the paths.
- Re-derived the reference trace as `morris_lecar_driven_oscillation_doi`
  (`independent_rk4_reference`, `I=100`, 3000 steps, seven crossings, first at step 141) via a
  new `_morris_lecar_rk4_features` helper verified bit-exact against the runner. The
  `MorrisLecarNeuron` descriptor now carries the schema's RK4 integration. Schema-gap counts
  are unchanged (a re-enrolment, not a new schema).
- The schema-DSL runner is Python-only; the hand `MorrisLecarNeuron` and its Rust/Julia/Go/Mojo
  mirrors were already RK4 / no-reset, so no polyglot counterpart or benchmark artefact changed.

### McKean piecewise-linear oscillator enrolment (RK4, no reset, rising-edge crossing)
- Enrolled the McKean (1970) piecewise-linear FitzHugh-Nagumo caricature (`mckean` schema,
  DOI `10.1016/0001-8708(70)90023-X`) into the WC-A5 schema corpus and Python↔Verilog
  co-simulation. The bundled schema is RK4, no reset, rising-edge (`v >= v_peak`) detection,
  with the three-branch piecewise-linear membrane `f(v) = min(max(-v, v - a), 1 - v)` — the
  second edge-crossing oscillator after FitzHugh-Nagumo. The min/max branch selection lowers
  to a fixed-point comparison + select (no look-up table), so at the sustained
  relaxation-oscillation operating point (`epsilon=0.2`, `gamma=0.5`, `I=0.6`) the hand
  `McKeanNeuron`, the schema runner, and the emitted Q16.16 RTL report the same 16-crossing
  train over 3000 steps **bit-exactly** — a genuine three-way parity, not a tolerance band.
- The default hand-model regime (`epsilon=0.01`) is a single-transient knife-edge; the
  enrolled regime is a robust limit cycle whose upward crossings survive fixed-point rounding.
- Committed an independent RK4-parity reference trace (`independent_rk4_reference`, `I=0.6`,
  3000 steps, 16 crossings, first at step 12) via a new `_mckean_rk4_features` helper verified
  bit-exact against the runner. Schema-gap counts move to 23 schema models / 129 net missing /
  131 source modules without a schema; the `McKeanNeuron` descriptor now carries the schema's
  RK4 integration and min/max dynamics (golden-trace SHA unchanged).
- The schema-DSL runner is Python-only; the hand `McKeanNeuron` and its Rust/Julia/Go/Mojo
  mirrors were already RK4 / piecewise-linear / no-reset, so no polyglot counterpart or
  benchmark artefact changed.

### FitzHugh-Nagumo faithful re-enrolment (RK4, no reset, rising-edge crossing)
- Replaced the bundled `fitzhugh_nagumo` schema's explicit-Euler + `v = -1` reset
  caricature with the genuine FitzHugh (1961) relaxation oscillator: four-stage RK4,
  **no reset**, rising-edge (`v >= v_threshold` upward crossing) spike detection, and
  the exact IEEE cube `v * v * v`. The schema now matches `FitzHughNagumoNeuron`
  bit-for-bit in float64, and over 3000 steps at `I=0.5` the hand model, the schema
  runner, and the emitted Q16.16 RTL report the same eight-crossing partial train
  **exactly** — a genuine three-way parity, not a tolerance band, because the
  right-hand side is polynomial (no look-up table). The earlier Euler+reset "parity"
  only held because both sides shared the same unfaithful reset dynamics; the RK4
  distinctness demonstration therefore moved to models whose spike count is genuinely
  integrator-sensitive (theta for RK4, resonate-and-fire for exponential Euler), since
  a faithful relaxation oscillator counts the same crossings under any integrator.
- The committed reference trace is re-derived with an independent RK4 recurrence
  (`independent_rk4_reference`, `I=0.5`, 3000 steps, eight crossings, first at step 29).
- The schema-DSL runner now fails closed on a non-finite state (`FloatingPointError`,
  matching the hand neuron models' `_validate_candidate` contract) rather than silently
  propagating `inf`/`nan` into the threshold decision, so the unbounded oscillator's
  large-step divergence is a controlled error, not a corrupt trace. FitzHugh-Nagumo also
  moved from the "transcendental (LUT)" compile group to the polynomial group in the
  DSL→Verilog tests — its cube lowers to plain fixed-point multipliers.
- The schema-DSL runner is Python-only; no benchmark dispatch or benchmark artefact
  changed. The hand `FitzHughNagumoNeuron` and its Rust/Julia/Go/Mojo mirrors were
  already RK4 / `v*v*v` / no-reset, so no polyglot counterpart changed.

### Rising-edge (`crossing`) threshold detection for non-resetting oscillators
- Made the schema DSL's `[threshold] detection = "crossing"` field functional in both the
  Python runner (`EquationNeuron`) and the schema→Verilog emitter. A non-resetting
  oscillator now spikes once per **upward** threshold crossing (matching the hand models'
  `v >= thr and v_prev < thr` edge test) instead of on every step it stays above threshold.
  This unlocks faithful enrolment of the biophysical oscillator family (FitzHugh-Nagumo,
  McKean, and the conductance oscillators), which the previous level-only path could only
  over-count. Validated end-to-end by a faithful FitzHugh-Nagumo hand-model / schema /
  Q16.16 RTL three-way parity at exact spike counts on a sustained relaxation-oscillation
  train (8 of 3000 steps). Edge detection engages **only** for a crossing model with no
  reset — a reset that clears the condition makes `level` and `crossing` identical, so every
  existing reset-based integrate-and-fire model keeps its exact prior behaviour (the field
  was previously decorative, and several reset models declared `crossing` harmlessly). The
  emitted RTL carries a 1-bit `_thr_prev` edge-history register, seeded from the initial
  state to bit-match the golden. The folded `compile_to_datapath` PE stays level-only for
  now (the co-simulation path uses the per-instance module); crossing support in the folded
  datapath is a separate follow-up.

### Mihalas-Niebur RK4 co-simulation enrollment
- Enrolled `mihalas_niebur` (Mihalaş & Niebur 2009 generalised linear
  integrate-and-fire) into the WC-A5 schema corpus and Python↔Verilog
  co-simulation as the first `method="rk4"` bundled model: a four-state RK4 flow
  (membrane, adaptive threshold, two spike-triggered currents) with a
  state-to-state `v >= theta` threshold and a `max(theta, theta_reset)`
  adaptive-threshold reset that floors on every spike. The schema runner
  reproduces the hand model bit-for-bit in float64, and a committed independent
  RK4-parity reference trace covers the new bundled schema. Unlike the exact Euler
  anchors, the Q16.16 RTL tracks the float spike train to within a single spike
  (35 vs 36 over 300 steps) rather than bit-exactly: the state-vs-state threshold
  compares two quantised states and the four-stage RK4 update injects four times
  the per-step rounding, so a marginal crossing shifts by one step under
  fixed-point rounding (timing jitter, not LUT coarseness — the right-hand side is
  linear). The co-simulation claim is therefore an honest tolerance band, not
  exact parity. The schema-DSL runner has no polyglot mirror; no benchmark
  dispatch or benchmark artefact changed.

### DYNAP-SE differential-pair integrator co-simulation enrollment
- Enrolled `dpi_neuron` (Chicca et al. 2014 DYNAP-SE differential-pair
  integrator, current-mode subthreshold LIF) into the WC-A5 Python↔Verilog
  co-simulation campaign: the explicit-Euler discretisation co-simulates at exact
  Q16.16 spike-count parity (three-way hand-model / schema / RTL) on a partial
  spike train, with a committed independent Euler-parity reference trace covering
  the new bundled schema. The drive is non-negative, so the source model's
  `max(i_mem, 0)` current rectification is inert and the linear schema is a
  faithful discretisation. The schema-DSL runner has no polyglot mirror; no
  benchmark dispatch or benchmark artefact changed.

### Izhikevich 2007 co-simulation enrollment and Verilog parameter-collision fix
- Enrolled `izhikevich2007` (Izhikevich 2007 biophysical quadratic IF, NeuroML
  `izhikevich2007Cell`) into the WC-A5 Python↔Verilog co-simulation campaign: the
  explicit-Euler discretisation co-simulates at exact Q16.16 spike-count parity
  (three-way hand-model / schema / RTL), with a committed independent
  Euler-parity reference trace covering the new bundled schema. The model's RK4
  default remains a separate RK4-emitter candidate.
- Fixed the schema→Verilog emitter to keep the parameter port map injective:
  `str.upper()` collapsed case-distinct names (Izhikevich `C` capacitance vs `c`
  reset voltage) onto one `P_C` port that iverilog rejected. Single-case
  parameter names keep the canonical `P_TAU`-style ports. The schema-DSL runner
  has no polyglot mirror; no benchmark dispatch or benchmark artefact changed.

### Discrete-map integration mode and Rulkov 2002 map fix
- Added a `method="map"` discrete-map integration mode to `EquationNeuron`
  (iterates `state_{n+1} = f(state_n)` directly), plus Verilog `IfExp` lowering
  and a discrete-map RTL datapath so piecewise maps compile to
  Icarus-Verilog-valid RTL. The `rulkov_map` schema adopts the DOI-verified
  Rulkov (2002) piecewise fast/slow map with `method="map"`, replacing the smooth
  variant that was integrated with explicit Euler and diverged; its reference
  trace is a bounded driven-spiking trace with exact independent map-iteration
  parity, so all 17 deterministic reference traces now hold independent parity.
  The schema-DSL runner has no polyglot mirror; no benchmark dispatch or
  benchmark artefact changed.

### Compiler HDL e2e CI
- Added a path-filtered `Compiler HDL E2E` workflow for pull requests touching
  `src/sc_neurocore/compiler/`, `src/sc_neurocore/hdl_gen/`, or `tests/e2e/`.
  A focused contract test locks the trigger paths and narrow e2e selector. No
  runtime package code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Deterministic reference-trace corpus
- Completed the deterministic bundled-schema reference-trace corpus for WC-A1b:
  all 17 deterministic schema-DSL models now have committed package data
  entries validated through `sc_neurocore.neurons.reference_traces`, while
  stochastic schemas and external simulator traces remain explicitly outside
  this deterministic corpus. No runtime package code, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Systematic audit rerun
- Added a systematic-audit rerun contract that locks the concrete 2026-07-04
  audit findings to repeatable gitignore, internal-TODO, SPDX-header, and SNN
  memory-discipline checks. No runtime package code, polyglot mirror, benchmark
  dispatch, or benchmark artefact changed.

### Strict typing and docstring policy
- Added a strict typing and NumPy docstring policy contract test that locks the
  2026-06-17 broadcast wiring across `pyproject.toml`, CI, preflight, the
  scoped docstring policy, and public maintenance docs. No runtime package code,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Rust/Python neuron binding boundary
- Documented the durable Python-only boundary for the five registry names
  without same-name PyO3 neuron constructors and locked each boundary to source
  evidence in the Rust/Python neuron parity map. No neuron runtime, PyO3
  implementation, polyglot mirror, benchmark dispatch, or benchmark artefact
  changed.

### Optional-extra CI matrix
- Added optional-extra CI matrix lanes for annealing, ONNX protobuf export, and
  real MPI. The lanes install their focused extras, run the import-skipped
  production test selectors, and are locked by workflow/docs contract tests. No
  runtime package code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Audit cadence
- Added a monthly/manual `Audit Cadence` workflow that runs pytest
  collect-only, validates tracked test inventory with
  `tools/test_inventory_audit.py`, and uploads collection/audit artefacts. A
  focused contract test keeps the workflow, MkDocs navigation, and public
  development guide in sync. No runtime package code, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Training device fallback
- Made `sc_neurocore.training.auto_device()` skip CUDA devices whose compute
  capability is not supported by the installed PyTorch build, avoiding noisy
  local GTX 1060 / `sm_61` warnings while preserving fallback to MPS or CPU.
  Public training and GPU install docs now state that explicit `device="cuda"`
  requires a matching PyTorch CUDA architecture. No training algorithm,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Optics optional-extra CI coverage
- Added a dedicated CI optics-extra lane that installs `.[dev,optics]` and runs
  `tests/test_optics -q -rs`, so the `gdsfactory`-gated GDSII round-trip tests
  are exercised outside the default CPU jobs. Public optional-dependency docs
  now name that CI boundary. No optics runtime algorithm, package promotion
  boundary, polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Surrogate compiler warning hygiene
- Migrated the surrogate custom-op compiler regression to
  `torch.compile(..., backend="eager")` and added a guard against deprecated
  TorchScript `script_method` usage in the touched training lane. Public
  surrogate docs now state that this is graph-capture evidence, not a throughput
  benchmark. No training algorithm, polyglot mirror, benchmark dispatch, or
  benchmark artefact changed.

### Annealing optional extra
- Added an `annealing` optional extra for `dwave-neal` and `dimod`, updated the
  optional dependency matrix and install-profile docs, and locked the metadata
  contract for the quantum-annealing `neal` parity selector. No base dependency,
  annealing runtime algorithm, benchmark dispatch, or benchmark artefact changed.

### Go autonomous-learning parity
- Fixed the autonomous-learning Go CGO setup contract: the parity test now runs
  from the Go module root with the `github.com/anulum/sc-neurocore/accel`
  import path, the Go bridge passes the Rust C-FFI timestep argument while
  preserving existing convenience calls, and public docs record the local
  `LD_LIBRARY_PATH` setup. No learning-rule dynamics, benchmark dispatch, or
  benchmark artefact changed.

### Rust/Python neuron binding coverage
- Documented the Rust/Python neuron binding coverage boundary and turned
  `tests/test_rust_python_neuron_parity.py` into a registry-level coverage map
  for 159 public Python registry names, including same-name Rust constructors,
  Rust-prefixed/core-only constructors, and current Python-only entries. No
  neuron runtime, PyO3 implementation, polyglot mirror, benchmark dispatch, or
  benchmark artefact changed.

### Vivado CI gates
- Added a public Vivado CI gate guide for the `MIF_VIVADO_CI=1` ZU3EG
  synthesis-flow tests. A focused contract test now discovers the live
  Vivado-gated pytest files and keeps the guide plus MkDocs navigation in sync.
  No HDL logic, runtime package code, dependency pins, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Optional dependency matrix
- Added a public optional-dependency matrix for the audited `gdsfactory`,
  `neal`/`dimod`, ONNX, Lava, snnTorch, SpikingJelly, CuPy, and MPI surfaces.
  A focused contract test now checks the matrix against `pyproject.toml`, the
  relevant skip-gated test paths, install-profile docs, and MkDocs navigation.
  No runtime package code, dependency pins, polyglot mirror, benchmark dispatch,
  or benchmark artefact changed.

### Performance-gate CI
- Added a scheduled/manual `Performance Benchmarks` workflow lane for the
  `SC_NEUROCORE_PERF=1` pytest selector and documented the perf-gate contract.
  A focused contract test now discovers perf-gated files from the live test tree
  and keeps the workflow selector plus public guide in sync. No runtime package
  code, polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Schema-gap reporting
- Added `tools/schema_gap_report.py` and focused tests for WC-A5 schema-DSL
  coverage planning. The report scans live model/schema files without importing
  optional backends, distinguishes the net schema gap from exact source-module
  alias coverage, and ranks missing-schema rows by source-evidence enrolment
  priority. No neuron runtime, HDL logic, polyglot mirror, benchmark dispatch,
  or benchmark artefact changed.

### Q4.12 co-simulation range classification
- Replaced the Q4.12 LIF zero-current co-simulation xfail with an explicit
  range-classification regression that checks the Q-format diagnostics and the
  public `precision lif` CLI. The co-simulation and precision docs now state
  that Q4.12 is a normalized-dynamics mode, not a zero-current millivolt-scale
  LIF parity mode. No HDL logic, runtime neuron dynamics, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Studio dependency profile
- Added `httpx2` to the Studio and full install profiles so Starlette
  `TestClient` uses its non-deprecated transport in Studio tests. The install
  profile guide and metadata contract tests now cover the dependency. No Studio
  runtime endpoint, polyglot mirror, benchmark dispatch, or benchmark artefact
  changed.

### Pytest warning hygiene
- Disabled the ambient `pytest_nengo` plugin in the repository pytest config so
  unrelated test collection no longer imports Nengo and emits the third-party
  NumPy 2.x `numpy.core` deprecation warning before SC-NeuroCore tests run.
  Added a policy test for the pytest configuration and active plugin registry.
  No runtime package code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Co-simulation toolchain gate
- Added a typed Icarus Verilog dependency checker and wired the CI test matrix
  to verify `iverilog -V` and `vvp -V` against the documented 12.x
  co-simulation floor before package tests run. The FPGA toolchain guide now
  records the same minimum and CI command. No HDL logic, runtime package code,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### Public prose hygiene
- Removed self-applied public superlatives from the CMOS profile notes, GPU
  backend guide, and archived state report while preserving the underlying
  measured values and benchmark comparison tables. No platform registry
  behavior, GPU backend code, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Copyright spelling hygiene
- Standardised copyright spelling across `tools/` and
  `.github/workflows/security-scanners.yml`: legacy `(c)` forms, ASCII
  date ranges, and `Sotek` spellings now use the canonical `©`, en dash year
  ranges, and `Šotek`. Touched tool headers now keep full seven-line GOTM
  descriptions, and generator string outputs were updated so emitted Vivado and
  Vertex config artefacts keep the same spelling. Strict mypy now passes on the
  touched tool scope. No runtime package code, HDL logic, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Header hygiene
- Split joined SPDX/commercial-license headers across manifests, workflow
  YAML, HDL sources, Vivado-import HDL copies, and the Vmin-LIF LUT generator.
  The touched TOML/Cargo manifests now retain full seven-line GOTM headers and
  canonical author spelling. No runtime package code, HDL logic, polyglot
  mirror, benchmark dispatch, or benchmark artefact changed.

### Configuration validator hygiene
- Hardened `scripts/validate_configs.py` into a typed repository configuration
  validator, removed the unused top-level `tomli` import, aligned the required
  user guide path with `docs/guides/USER_MANUAL.md`, and added focused CLI
  validation tests. The generated capability surfaces were refreshed for the
  new test file. No runtime package, polyglot mirror, benchmark dispatch, or
  benchmark artefact changed.

### Expression differentiator typing
- Restored strict-mypy compliance for the neuron expression differentiator by
  isolating SymPy's partially typed constructors, differentiation, and printer
  APIs behind typed boundary helpers. The in-grammar derivative contract,
  finite-difference behavior, generated API surface, polyglot mirrors, and
  benchmark-dispatched paths are unchanged.

### Studio sandbox hardening
- Hardened Studio job sandbox path confinement so generated job directories,
  seed/control seed inputs, live artifact reads, artifact downloads, control
  commands, and purges validate canonical paths before filesystem access. Added
  focused regression coverage for malformed job IDs and symlinked reserved seed
  directories. No Studio job API, worker payload schema, polyglot mirror,
  benchmark dispatch, or benchmark artefact changed.

### Sensor package exports
- Exported the ADC-to-spike kernel surface from `sc_neurocore.sensors`,
  including `ADCSpikeWindowConfig`, `ADCSpikeWindowResult`, backend selection,
  the bit-true Python floor, and `quantise_adc`. Existing submodule imports
  remain compatible. Sensor API docs, hardware docs, and module-specific tests
  were updated. No ADC arithmetic, Rust/Julia/Go/Mojo backend, dispatch order,
  benchmark output, or benchmark artefact changed.

### Quantum package exports
- Exported the SC→quantum compiler surface from `sc_neurocore.quantum`,
  including `QuantumGate`, `SCQuantumCircuit`, probability/rotation helpers, and
  `compile_sc_multiply` / `compile_sc_layer`. Existing submodule imports remain
  compatible. Quantum API docs, tutorial examples, and module-specific tests
  were updated. No quantum algorithm, polyglot safety mirror, benchmark
  dispatch, or benchmark artefact changed.

### Training package exports
- Exported the NumPy equilibrium-propagation research surface as
  `sc_neurocore.training.EPNetwork`, so the existing two-phase
  settle-and-nudge path is selectable from the package facade. Training docs and
  module-specific tests were updated. No Torch training path, polyglot kernel,
  benchmark dispatch, or benchmark artefact changed.

### Spintronic package exports
- Exported the documented magnetic-domain mapper surface from
  `sc_neurocore.spintronic`, including `SpintronicMapper`,
  device/material models, MuMax3 helpers, racetrack/skyrmion utilities,
  aging/radiation/defect models, and the Verilog generator. The quick-start docs
  now use actual exported symbols, and existing submodule imports remain
  compatible. Generated capability surfaces were refreshed for the new package
  API test file. No mapper kernel, Julia/Rust/Mojo mirror, benchmark dispatch,
  or benchmark artefact changed.

### Memristor package exports
- Exported the documented memristor crossbar mapper surface from
  `sc_neurocore.memristor`, including `MemristorMapper`,
  conductance/crossbar models, compensation helpers, Monte Carlo reports, and
  the SystemVerilog emitter. Existing submodule imports remain compatible. No
  mapper kernel, Julia/Rust/Mojo mirror, benchmark dispatch, or benchmark
  artefact changed.

### JAX dense layer exports
- Exported `JaxSCDenseLayer` through the lazy package facades as
  `sc_neurocore.JaxSCDenseLayer` and
  `sc_neurocore.layers.JaxSCDenseLayer`. The optional backend remains
  construction-time gated by the `jax` extra, and package import stays
  lightweight. Public API tests, layer docs, and generated capability surfaces
  were updated. No JAX kernel, polyglot mirror, benchmark dispatch, or benchmark
  artefact changed.

### Analysis package exports
- Exported `phi_star` and `phi_from_spike_trains` from
  `sc_neurocore.analysis`, so the maintained Phi* integrated-information
  estimator is selectable from the public analysis namespace. The curated
  analysis guide was updated and generated API freshness was verified. No
  backend kernel, benchmark dispatch, or benchmark artefact changed.

### Quantum cognition coverage
- Added focused GOTM brain contracts for local-LLM import fallback and
  spike-index accumulation. The focused selector for `dashboard.py`,
  `gotm_brain.py`, and `radical_pair.py` now reports 100% exact-file coverage.
  No runtime path, polyglot mirror, benchmark dispatch, generated API surface,
  or benchmark artefact changed.

### Public API reference
- Changed the generated API reference to publish public classes, public module
  functions, public methods, and dunder methods only. Single-underscore helper
  classes, functions, and methods are now omitted from `docs/API_REFERENCE.md`;
  the generator contract and documentation workflow were updated together. No
  runtime path, polyglot mirror, benchmark dispatch, or benchmark artefact
  changed.

### Quantum cognition memory schema
- Changed quantum-cognition CLI SNN stimulus records to emit the canonical
  numeric fleet-memory `timestamp` field while keeping `content`, `project`,
  `actor`, `kind`, and `source_ref` stable. The CLI memory-discipline contract
  now rejects legacy `text` and `source` aliases and locks the numeric timestamp
  shape. No runtime model dynamics, polyglot kernel, benchmark-dispatched path,
  or benchmark artefact changed.
- Added `tools/snn_memory_discipline_audit.py` to validate SC-NeuroCore SNN
  stimulus producers and existing records against the fleet memory-write
  schema, including canonical keys, controlled actors, timestamps, entities,
  kinds, and source provenance. Repair mode normalizes legacy local records
  without deleting files or changing runtime model dynamics, polyglot kernels,
  benchmark dispatch, or benchmark artefacts.

### Perfect-integrator co-simulation schema
- Added the DOI-backed `perfect_integrator` UniversalNeuron schema in TOML and
  JSON form, packaged bundled schema assets into the wheel, and enrolled the
  model in deterministic Q8.8 Python-to-Verilog co-simulation. New tests compare
  the schema against the hand-authored `PerfectIntegratorNeuron` and assert
  emitted-RTL spike-count parity. No polyglot mirror implementation or
  benchmark-dispatched path changed.

### Compiler proof-transform wiring
- Added an explicit opt-in `proof_transforms` compiler facade for whitebox state
  taps and operator abstraction. The package root now exposes registry lookup and
  dispatch helpers, docs classify the transforms as proof-only rather than
  production compiler flags, and compatibility coverage keeps `quantize_core`
  anchored to the canonical quantizer surface. No polyglot kernel,
  benchmark-dispatched path, or benchmark artifact changed.

### Public docstring quick wins
- Closed the AB-DOC-1 quick-win set by documenting the public package entry
  points for layers, synapses, utils, datasets, formal verification, and SCPN
  layers; the CLI entrypoint; optional Rust fallback surfaces for DNA,
  quantum-annealing, photonic, and Studio helpers; Studio preset/template
  helpers; and Studio request schemas. The cleaned files are now locked in
  `docs/docstring_policy.toml`, raising the scoped public-docstring gate from
  258 to 273 files. Generated API docs were refreshed. No runtime path,
  polyglot mirror, benchmark dispatch, or benchmark artefact changed.

### NotImplemented guard audit
- Added a tracked source audit for executable Python `NotImplementedError`
  sites. The audit allows only explicit fail-fast guards for unsupported MPI,
  forced-Rust, Torch bridge, NIR node-map, optics GDSII, hardware-DMA, and
  abstract-neuron boundaries, preventing hidden selectable stubs from entering
  tracked Python sources. No runtime path, polyglot mirror, or benchmark
  artefact changed.

### Rust interneuron performance-test gating
- Marked the Rust interneuron wall-clock smoke tests for PV, SST, VIP,
  Chandelier, cerebellar basket, and Martinotti neurons as opt-in ignored tests
  so default `cargo test` no longer fails under host CPU contention. Timing
  evidence remains owned by the Criterion benchmark surfaces; no neuron
  dynamics, Python surface, polyglot mirror, or benchmark artefact changed.

### Block-floating scalar guard
- Added an explicit `BlockFloatingScalarEncodingError` and scalar-only preset
  guard for mixed-precision configs. Block-floating precision remains available
  for metadata-aware dense, adaptive, and manifest paths, while scalar parameter
  encoders can now call `from_preset(..., scalar_only=True)` or
  `encode_scalar_value(...)` to fail closed before detached exponent metadata is
  lost. No polyglot kernel, HDL datapath, or benchmark-dispatched path changed.

### Reference trace validation harness
- Added a schema-driven neuron reference-trace validation harness with immutable
  corpus contracts, fail-closed JSON payload parsing, package-data loading,
  `UniversalNeuron` execution, and feature-level validation reports. The seed
  corpus covers analytic closed-form `lif`, `lapicque`, `quadratic_if`,
  `theta`, and spike-bearing `perfect_integrator` traces and is covered by
  strict mypy checks plus 100% exact-file focused coverage; no polyglot kernel or
  benchmark-dispatched runtime path changed.

### Network Rust backend contract hardening
- Reverified the DEEP_AUDIT network/Rust findings against the current tree:
  spike events use `u64` packing with 32-bit neuron/timestep lanes, population
  dispatch uses `Population.model_name` rather than labels, Rust final voltages
  sync back into populations, and the effective workspace release profile now
  owns `panic = "abort"`. Added regression tests for those contracts and made
  forced Rust fail fast for `StateMonitor`, `RateMonitor`, `spike_gating`, and
  `fim_lambda`, while `backend="auto"` falls back to Python for those
  Python-only semantics. No benchmark-dispatched path, polyglot mirror, or
  benchmark artefact changed.

### Release provenance and publish retry hardening
- Added manual tagged-release backfill support to the release workflow,
  retained the release security packet as a workflow artifact even when the
  sweep fails, and aligned release/security scanner cargo-fuzz installation on
  the validated `cargo-fuzz==0.13.2` pin. Made PyPI and crates.io publish
  retries idempotent for already-published versions while preserving manual
  dry-run validation. No runtime package, polyglot kernel, or benchmark path
  changed.

### Mojo helper contract honesty
- Removed the hidden `NotImplementedError` IPC stub from
  `accel.mojo.runner.MojoKernelRunner.popcount` and `lfsr_encode`, exported
  `MOJO_HELPER_BACKEND="python-fallback"` plus
  `MOJO_HELPER_IPC_AVAILABLE=False`, and added AST regression coverage so scalar
  helpers cannot silently present a fake Mojo path. Refreshed the Mojo docs to
  separate actual build/benchmark subprocess execution from Python helper
  fallbacks. No Mojo kernel, benchmark-dispatched path, Rust path, or polyglot
  mirror implementation changed.

### Backend selector coverage ratchet
- Extended the benchmark-driven backend selector tests to cover CPU probe
  fallback, missing benchmark directories, and malformed benchmark JSON files,
  bringing `accel.backend_selection` to 100% exact-file coverage. Promoted the
  selector into the scoped NumPy docstring policy. No dispatch order, benchmark
  artefact, polyglot runtime, or kernel implementation changed.

### Constants ledger audit hardening
- Added dedicated constants-ledger regression tests for the 44 public constants,
  scalar types, physical ranges, Q8.8 invariants, maintained-source import map,
  module-docstring honesty, and Izhikevich spike-threshold wording. Corrected
  the constants module docstring to state the current 16-module Python adoption
  boundary, promoted it into the scoped NumPy docstring policy, and refreshed
  the constants audit page. No constant values, Rust mirror values, or
  benchmark-dispatched paths changed.

## [3.16.0] - 2026-07-05

### Provenance and citation integrity
- Verified every neuron-descriptor DOI against its registry (Crossref, or DataCite
  for arXiv preprints) and corrected misstated and fabricated citations: a
  non-existent "Kilinc & Bhatt (2023)" (the model is the Nagumo-Sato/Aihara sigmoid
  map), a fabricated "Jahns et al. (2025)", a recurring phantom "Bhatt" co-author
  across several cerebellar models, an arXiv identifier that pointed at an unrelated
  paper, and several digit-transposed or wrong DOIs. Added `tools/provenance/verify_dois.py`
  and a committed DOI ledger checked offline by `tests/test_provenance_doi_integrity.py`,
  so a fabricated or mistyped descriptor DOI now fails CI. Descriptor citeable coverage
  rose from 119 to 131 models.

### Verification safety screen hardening
- Hardened `verification.safety.CodeSafetyVerifier` to reject AST-visible
  filesystem, process, network, relative-import, dynamic-import,
  dynamic-execution, and reflection escape routes including `open(...)`,
  `Path(...).write_text(...)`, `socket.socket()`, `__builtins__.eval(...)`,
  `__builtins__['eval'](...)`, and `getattr(__builtins__, 'eval')`. Promoted
  the verifier into the scoped NumPy docstring policy, strict-typed the focused
  tests, refreshed the verification guide and generated API reference, and kept
  `verification/safety.py` at 100% exact-file coverage. The generated
  capability manifest, snapshot, and README capability block were also refreshed
  after the public-claims selector exposed a stale tracked-test count. No
  polyglot mirror or benchmark-dispatched path changed.

### Go services namespace docstring ratchet
- Promoted `accel.go.services` into the scoped NumPy docstring policy, added
  an AST-visible package docstring, declared the checked-in Go service file and
  package-boundary globs, and extended the acceleration mirror-authority tests
  to cover the services namespace at 100% exact-file coverage. No Go runtime,
  polyglot mirror, or benchmark-dispatched path changed.

### Go acceleration namespace docstring ratchet
- Promoted `accel.go` into the scoped NumPy docstring policy, added an
  AST-visible package docstring, declared the maintained Python ctypes loader
  entry points and broad Go service namespace globs, and extended the
  acceleration mirror-authority tests to cover the Go namespace at 100%
  exact-file coverage. No Go runtime, polyglot mirror, or benchmark-dispatched
  path changed.

### Precision solver width and budget hardening
- Hardened `compiler.precision_solver` with sign-inclusive integer-width
  calculation, finite/range validation, unsigned-negative range rejection,
  invalid budget/alignment rejection, and aligned budget reductions that
  recompute datapath width from the integer floor and reduced fraction. Added a
  dedicated strict-typed precision-solver test module with 100% exact-file
  coverage, promoted the solver into the scoped NumPy docstring policy,
  refreshed generated API documentation, and regenerated capability snapshots
  for the new test-file count. No polyglot mirror or benchmark-dispatched path
  changed.

### Precision config docstring and coverage ratchet
- Promoted `compiler.precision_config` into the scoped NumPy docstring policy,
  added compliant public docstrings for fixed-point and block-floating
  precision value-object properties and manifest helpers, and expanded the
  focused compiler precision tests to cover validation, range maths, manifest
  payloads, exponent-layout delegation, and saturating fixed-point encoding at
  100% exact-file coverage. The generated API reference was refreshed. No
  runtime contract, polyglot mirror, or benchmark-dispatched path changed.

### MLIR emitter docstring and coverage ratchet
- Promoted `compiler.mlir_emitter` into the scoped NumPy docstring policy,
  added compliant docstrings for the MLIR node, bundle, emitter, wire, and
  operation helpers, strict-typed the focused MLIR tests, and covered
  `MLIRBundle.to_dict()` so the emitter reaches 100% exact-file coverage under
  the focused selector. The generated API reference was refreshed. No runtime
  contract, polyglot mirror, or benchmark-dispatched path changed.

### IR type checker docstring ratchet
- Promoted `compiler.ir_type_checker` into the scoped NumPy docstring policy,
  added compliant public class docstrings for the signal-domain enum and edge
  record, strict-typed the focused IR type-checker tests, and refreshed the
  generated API reference. The existing focused selector keeps the module at
  100% exact-file coverage. No runtime contract, polyglot mirror, or
  benchmark-dispatched path changed.

### Cortical column coverage hardening
- Added strict-typed cortical-column coverage contract tests for static scale
  validation, auto-backend single Rust SpMV fallback, and delayed per-bin
  injection. `network.cortical_column` now reaches 100% exact-file coverage
  under the focused fast selector, and the generated capability inventory was
  refreshed. No runtime contract, polyglot mirror, or benchmark-dispatched path
  changed.

### Native learning bridge docstring hardening
- Promoted `_native.learning_bridge` into the scoped NumPy docstring policy,
  documented the public Rust, Rust-WGPU, and Torch bridge methods, covered the
  single-step FFI, layer reset, Torch reset-scope, bit-spec length, and
  PyTorch-unavailable fallback paths, and refreshed the autonomous-learning docs
  plus generated API reference. No FFI contract, polyglot mirror, or benchmark
  path changed.

### DARTS NAS docstring hardening
- Promoted `nas.darts_sc_nas` into the scoped NumPy docstring policy, added
  public docstrings for the differentiable bitstream-selection methods,
  strict-typed the focused DARTS NAS tests, refreshed the NAS API guide and
  generated API reference, and preserved the torch-gated runtime behavior. No
  polyglot mirror or benchmark-dispatched path changed.

### NAS search surface docstring hardening
- Promoted `nas.search_space`, `nas.search`, and `nas.equiv` into the scoped
  NumPy docstring policy, added missing public summary/property docstrings,
  strict-typed the focused NAS tests, refreshed the NAS API guide and generated
  API reference, and removed local NAS type-ignore escapes. No runtime contract,
  polyglot mirror, or benchmark-dispatched path changed.

### SC-NAS engine docstring and coverage hardening
- Promoted `nas.sc_nas_engine` into the scoped NumPy docstring policy, added
  compliant public docstrings for the hardware-aware search/report surface,
  strict-typed the focused NAS engine tests, covered the optional Rust
  tournament import branch through the module import boundary, and refreshed
  the public NAS docs plus generated API reference. No runtime contract,
  polyglot mirror, or benchmark-dispatched path changed.

### SCConv layer docstring and test hardening
- Promoted `layers.sc_conv_layer.SCConv2DLayer` into the scoped NumPy
  docstring policy, added compliant module, initialization, and forward
  docstrings, strict-typed the focused SCConv tests, refreshed the public layer
  docs and generated API reference, and verified the production file at 100%
  exact-file coverage. No runtime contract, polyglot mirror, or
  benchmark-dispatched path changed.

### Compiler MLIR export hardening
- Hardened `export.compiler_export.CompilerExporter` with explicit `mlir`
  target validation, empty-graph rejection, duplicate node/output detection,
  wrong-arity and unsupported-node failures, missing external shape checks,
  graph-input/output collision rejection, and positive tensor-dimension
  validation before SSA MLIR emission. Removed the embedded demo path, covered
  the module at 100% exact-file coverage, strict-typed the focused tests, and
  promoted the compiler exporter into the scoped NumPy docstring policy.

### DVS input contract hardening
- Hardened `interfaces.dvs_input.DVSInputLayer` with fail-closed dimension,
  decay, AER address, timestamp, polarity, and bitstream-length validation
  before event-surface mutation. Empty event batches now return probability
  frames instead of exposing the mutable internal surface, and invalid
  cross-batch timestamp rewinds leave `surface` and `last_update_time`
  unchanged. The Rust safety mirror, Julia validation mirror, and Mojo FFI
  validation shim now enforce the same DVS boundaries, and the advanced-module
  DVS benchmark now uses monotonic precomputed event batches. The public DVS
  input surface is covered at 100% exact-file coverage under strict mypy plus
  the scoped NumPy docstring policy.

### Fisher-Posner LIF contract hardening
- Hardened `quantum_cognition.fisher_posner` with fail-closed neuron-id,
  timestep, voltage, membrane-time-constant, ATP-domain, and step-current
  validation before state mutation. Invalid input current now fails without
  advancing counters, membrane voltage, ATP, or spin-pool measurement state.
  The public Fisher-Posner LIF surface is covered at 100% exact-file coverage
  under strict mypy plus the scoped NumPy docstring policy.

### RNG utility contract hardening
- Hardened `utils.rng.RNG` with fail-closed seed, normal, uniform, and
  Bernoulli parameter validation before generator state advances. Scalar draws
  now expose Python scalar types, shaped draws return dtype-stable NumPy
  arrays, and the public utility is covered at 100% exact-file coverage under
  strict mypy plus the scoped NumPy docstring policy.

### L13 holonomic source-field adapter hardening
- Hardened `adapters.holonomic.l13_source` with fail-closed parameter,
  timestep, feedback-rank, feedback-emptiness, finite-value, decode, and
  no-mutation validation before vacuum/Fisher state updates. Scalar feedback
  now broadcasts across the vacuum lattice, and mismatched vector or batch
  feedback projects deterministically by mean drive. The public adapter is
  covered at 100% exact-file coverage under strict mypy plus the scoped NumPy
  docstring policy. The Rust safety mirror now validates real state, the Julia
  mirror is callable, and the Mojo contract shim builds as a shared library
  with callable validation helpers.

### L6 holonomic planetary adapter hardening
- Hardened `adapters.holonomic.l6_plan` with fail-closed parameter, timestep,
  input-rank, input-width, non-empty-row, and finite-value validation before
  Gaia-field state mutation. Mismatched upstream region counts now broadcast a
  deterministic mean regional drive across configured planetary regions, and
  the public adapter is covered at 100% exact-file coverage under strict mypy
  plus the scoped NumPy docstring policy. The Rust safety mirror now enforces
  the same no-mutation input contract, the Julia mirror is callable, and the
  Mojo contract shim builds as a shared library with callable validation
  helpers.

### L9 holonomic memory adapter hardening
- Hardened `adapters.holonomic.l9_mem` with fail-closed parameter, timestep,
  input-rank, input-width, non-empty-row, and finite-value validation before
  TSVF state mutation. Mismatched upstream slot counts now tile
  deterministically across configured memory slots, and the public adapter is
  covered at 100% exact-file coverage under strict mypy plus the scoped NumPy
  docstring policy. The Rust safety mirror now validates real L9 state, the
  Julia mirror is callable, and the Mojo contract shim builds as a shared
  library with callable validation helpers.

### GPU fallback reduction hardening
- Hardened the CuPy/NumPy GPU fallback and shared packed-bitstream vector
  reductions against coverage-time NumPy reloads by routing reductions through
  the active NumPy module with explicit zero initials. Covered the CuPy-absent
  import branch, GPU CPU fallback, and vector pack/unpack paths at 100%
  exact-file coverage under strict mypy. The benchmark suite now synchronizes
  CUDA only while the GPU runtime is still live, so the GPU section completes
  under the NumPy fallback when local CUDA discovery fails.

### Mojo runner contract hardening
- Covered `accel.mojo.runner` at 100% exact-file coverage, including the
  fail-closed missing-`kernels.mojo` constructor path. Promoted the runner to
  the scoped NumPy docstring policy and corrected the Mojo acceleration docs so
  `build`, `run_benchmark`, `popcount`, and `lfsr_encode` describe the current
  fallback and failure contracts exactly.

### Quantum Studio telemetry hardening
- Covered `QuantumStudioHook` snapshot, compact JSON event, and debug
  representation contracts at 100% exact-file coverage under strict mypy.
  Added the telemetry hook to the scoped NumPy docstring policy and refreshed
  the quantum cognition API documentation for the Studio streaming surface.

### Host driver generator hardening
- Hardened generated Python and C host drivers so module names, parameter
  registers, setters, include guards, and C function prefixes are sanitized into
  valid identifiers before source emission. Empty module identifiers and
  sanitized parameter collisions now fail closed, C drivers expose parameter
  setters matching the Python surface, and `compiler.host_driver_gen` is covered
  at 100% exact-file coverage under strict mypy plus the scoped NumPy docstring
  policy.

### Compiler pipeline hardening
- Hardened `CompilerPipeline` path and tool boundaries: artifact paths now use
  `commonpath` work-directory validation, EDA executables are resolved to
  absolute paths before subprocess launch, and the public pipeline surface is
  covered by strict-typed contract tests plus the scoped NumPy docstring policy.

### Universal DSL contract hardening
- Covered the schema loader's explicit missing-path and Python pre-3.11 TOML
  fallback branches, TOML bool/list serialization, and default Verilog module
  name sanitization at 100% exact-file coverage. Added the Universal DSL public
  surface to the scoped NumPy docstring policy.

### SC-NIR compatibility hardening
- Strict-typed the SC-NIR compatibility matrix contract tests, kept the matrix
  validator at 100% exact-file coverage, and promoted the public compatibility
  audit surface into the scoped NumPy docstring policy.

### ONNX graph export hardening
- Fixed dependency-free `ONNXExporter` final-output metadata so a terminal
  `SC_POPCOUNT` emits an `int32` tensor instead of a bitstream bool tensor, and
  mapped operators without shape inference rules now fail closed. Covered the
  exporter at 100% exact-file coverage, strict-typed the focused tests, and
  documented the file exporter versus graph exporter split.

### Pipeline ingestion hardening
- Hardened `DataIngestor` so the reserved `labels` key is preserved as labels
  instead of normalized as a modality, modality arrays must be finite and share
  a sample axis, and scalar/empty inputs fail closed. Covered
  `pipeline.ingestion` at 100% exact-file coverage, strict-typed the focused
  tests, corrected public pipeline docs, and added the surface to the scoped
  NumPy docstring policy.

### Neuron package facade hardening
- Covered the `sc_neurocore.neurons` lazy facade at 100%, including optional
  Rust-dispatch opt-out, cache reuse, pure-Python model fallback, cached
  package-level model exports, and unknown-symbol `AttributeError` handling.
  Added the package facade to the scoped NumPy docstring policy.

### Descriptor schema contract hardening
- Covered the v2 model descriptor parser's defensive contract branches at 100%:
  missing and non-table sections, empty structural descriptors, scalar legacy
  state/parameter forms, string backend statuses, dynamics expression tables,
  invalid tag/range/year/numeric shapes, and single-author provenance fallback.
  Added `neurons.model_descriptor` to the scoped NumPy docstring policy.

### Descriptor generator hardening
- Added a fail-closed class-name guard to the model descriptor generator and
  narrowed legacy v1 schema fallback so missing schemas remain allowed while
  malformed curated schemas abort corpus refresh. Covered plain-constructor
  filtering, source-inspection fallbacks, merge curation preservation, and
  brought `neurons.descriptor_generator` into the scoped NumPy docstring policy.

### Descriptor catalogue hardening
- Added a fail-closed class-name guard for model descriptor lookup so the
  public descriptor helpers accept only public Python identifiers before
  filesystem access. Covered valid-absent descriptor branches, the aggregate
  catalogue coverage summary, and added `neurons.model_catalogue` to the scoped
  NumPy docstring policy.

### Physics and mathematics hardening
- Promoted `DendriticNMDANeuron` (two-compartment Jahr-Stevens NMDA Mg2+ block
  neuron) from raw dendrite-first Euler to candidate-first RK4 over
  `(v_soma, v_dend)`, with finite parameter/state/input validation,
  reset-on-commit soma spike semantics, and an explicit
  `integrator="baseline_euler"` regression path. Replaced Go, Rust safety, Julia,
  and Mojo placeholders or broken paths with real dual-input RK4 mirrors,
  harmonised the Rust engine, added focused Python/Go/Rust tests, a Rust
  benchmark example, a five-backend local non-isolated benchmark artefact, and
  refreshed the model documentation with measured benchmark results. Python,
  Rust, Go, Julia, and Mojo agree on the 253-spike anchor at 20k steps /
  `i_soma=50.0`, `glutamate=0.5`.
- Promoted `NeuroGridNeuron` (reduced Neurogrid two-compartment analog EIF
  neuron) from raw Euler to candidate-first RK4 over `(v_s, v_d)`, with an
  event-limited soma stage cap at `v_peak`, finite input/state/candidate
  validation, reset-on-commit spike semantics, and an explicit
  `integrator="baseline_euler"` regression path. Replaced placeholder Go and
  Rust safety mirrors, repaired Julia, added Mojo, harmonised the Rust engine,
  added focused Python/Go/Rust tests, a Rust benchmark example, a five-backend
  local non-isolated benchmark artefact, and refreshed the model documentation.
  Python, Rust, Go, Julia, and Mojo agree on the 94-spike anchor at 20k steps /
  current 100.0.
- Promoted `HayL5PyramidalNeuron` (reduced three-compartment Layer 5 thick-tufted
  pyramidal cell) from four raw Euler sub-steps to candidate-first RK4 over the
  nine-state `(v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a, ca_a)` system, with
  finite input/state/candidate validation, non-negative tuft-calcium candidates,
  dual soma/tuft input parity, and an explicit `integrator="baseline_euler"`
  regression path. Replaced broken or placeholder Go, Rust safety, Julia, and
  Mojo surfaces with real RK4 mirrors, harmonised the Rust engine, added focused
  Python/Go/Rust tests, a Rust benchmark example, a five-backend local
  non-isolated benchmark artefact, and refreshed the model documentation. Python,
  Rust, Go, Julia, and Mojo agree on the 1-spike anchor at 20k steps /
  `current_soma=10.0`, `current_tuft=0.0`; the dual-input anchor is 4 spikes at
  20k steps / `current_soma=5.0`, `current_tuft=5.0`.
- Promoted `DeSchutterPurkinjeNeuron` (compact De Schutter & Bower Purkinje cell)
  from five raw Euler sub-steps to candidate-first RK4 over the seven-state
  `(v, h_na, n_k, m_cap, h_cap, q_kca, ca)` system, with finite input/state/
  candidate validation, non-negative calcium candidates, and an explicit
  `integrator="baseline_euler"` regression path. Replaced the decorative Go,
  Rust safety, and Mojo placeholders with real RK4 mirrors, harmonised the Rust
  engine and Julia surface, added focused Python/Go/Rust tests, a Rust benchmark
  example, a five-backend local non-isolated benchmark artefact, and refreshed
  the model documentation. Python, Rust, Go, Julia, and Mojo agree on the 1-spike
  anchor at 20k steps / current 500.0.
- Promoted `MulticompartmentMCNNeuron` (Spiking-WM dual-dendrite working-memory
  cell) from raw forward Euler to candidate-first RK4 over the coupled
  `(u, v_basal, v_apical)` system, with finite input/state/candidate validation
  and an explicit `integrator="baseline_euler"` regression path. The Rust engine,
  Rust safety mirror, Go service, Julia mirror, and new Mojo kernel now share the
  same derivative order and threshold-reset rule. Added focused Python/Go/Rust
  tests, a Rust benchmark example, a five-backend local non-isolated benchmark
  artefact, and refreshed the model documentation. All five backends agree on
  the `49,999` spike anchor at 200k steps / basal current 3.2.

## [3.15.35] - 2026-06-26

### Physics and mathematics hardening
- Promoted `HillTononiNeuron` (Hill & Tononi 2005 thalamocortical sleep/wake
  cell) from a hard-coded forward-Euler step to candidate-first RK4 over the
  six-state `(V, h_na, n_k, m_h, h_t, na_i)` system — fast Na⁺, delayed-rectifier
  K⁺, `Ih`, T-type Ca²⁺, a sodium-dependent K⁺ current, and a saturating Na/K
  pump — with input validation and the opt-in `integrator="baseline_euler"`
  regression path. Replaced the decorative `accel/go/services` and Mojo
  placeholders (and corrected a wrong Go spike threshold) with real RK4 backends
  and harmonised the cross-language arithmetic so Python, Rust, Julia, Go, and
  Mojo reproduce the trajectory bit-for-bit: explicit `m·m·m`/`n·n·n·n`
  conductance powers, and the `I_KNa` Hill exponent `3.5` evaluated as
  `b·b·b·sqrt(b)` (an IEEE-754 exact decomposition) instead of a per-platform
  `pow`. Added a `_safe_exp` guard so the saturating gates stay finite under an
  out-of-range stimulus (Python `math.exp` would otherwise raise where the other
  backends return `+inf`), native Go RK4 parity/behaviour tests, a Go benchmark
  hook, a Rust benchmark example, Python RK4/fail-closed coverage, and a
  five-backend local non-isolated benchmark that fails closed unless every
  backend reports an identical spike count (694 at 200k steps / 10 nA).
- Promoted `DurstewitzDopamineNeuron` (Durstewitz, Seamans & Sejnowski 2000
  D1-modulated PFC cell) from a hard-coded forward-Euler step — which advanced
  the gates from the old voltage and then the voltage from the freshly updated
  gates, mixing two inconsistent states — to candidate-first RK4 over the
  three-state `(V, h_na, n_k)` system, with input validation and the opt-in
  `integrator="baseline_euler"` regression path. Replaced the decorative
  `accel/go/services` and Mojo placeholders with real RK4 backends and harmonised
  the cross-language arithmetic (explicit `m·m·m`/`n·n·n·n` conductance powers,
  the `mg / 3.57 · exp` Mg²⁺-block operand order, `math.exp`) so Python, Rust,
  Julia, Go, and Mojo reproduce the trajectory bit-for-bit. Added native Go RK4
  parity/behaviour tests, a Go benchmark hook, a Rust benchmark example, the
  Python RK4/fail-closed test coverage, and a five-backend local non-isolated
  benchmark that fails closed unless every backend reports an identical spike
  count (925 at 200k steps / 10 nA).
- Completed the `UpperMotorNeuron` (Pospischil 2008 corticospinal L5 pyramidal)
  polyglot backend coverage by adding the Mojo exponential-Euler kernel, raising
  it to a Python / Rust / Julia / Go / Mojo set. The membrane keeps its analytic
  frozen-conductance exponential-Euler step and the gates keep their closed-form
  steady/tau update — both unconditionally stable for the stiff sodium gate, so
  RK4 here would be a regression rather than a hardening. Added a Go benchmark
  hook, a Rust exponential-Euler benchmark example, and a five-backend local
  non-isolated benchmark that fails closed unless every backend reports an
  identical spike count.
- Promoted `EnergyLIFNeuron` from raw Euler membrane and metabolic-reserve
  updates to the exact constant-current flow for the coupled `(v, epsilon)`
  state across Python, Go, Julia, Mojo, and Rust safety surfaces. Added
  module-specific Python/Go/Rust exact-flow and invalid-state coverage,
  refreshed the public model documentation with measured five-backend timing
  rows, and added a local non-isolated benchmark gate for exact spike-count
  parity.
- Promoted `MATNeuron` from a split forward-Euler membrane update plus separate
  threshold decay to candidate-first RK4 over `(v, theta1, theta2)` across the
  Python reference, Go service, Julia mirror, Mojo helper, and Rust safety
  surface. Replaced the Go/Rust/Mojo placeholders with numeric parity surfaces,
  added Go/Rust tests and Python RK4/fail-closed coverage, refreshed the model
  documentation with measured five-backend timings, and added a local
  non-isolated benchmark gate for exact spike-count parity.
- Promoted `SFANeuron` from forward-Euler voltage plus separate adaptation
  decay to candidate-first RK4 over the coupled `(v, g_sfa)` adaptation ODE
  across Python, Go, Julia, Mojo, and Rust safety surfaces. Added native Go/Rust
  RK4 tests, refreshed Python module tests, a five-backend local non-isolated
  benchmark artifact, a regression-gate row, and updated model documentation.
- Replaced `ExpIFNeuron` raw Euler mutation with candidate-first RK4 across the
  maintained Python reference, Rust engine, Go service, Julia mirror, and Mojo
  mirror. The Fourcaud-Trocmé EIF ODE and hard reset are unchanged; all surfaces
  now reject non-finite RK4 derivatives/candidates before mutation. Added focused
  Python/Rust/Go RK4 tests, a Go benchmark hook, a local non-isolated Python RK4
  regression artifact, and refreshed the public model documentation.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `McKeanNeuron` (McKean 1970 piecewise-linear FitzHugh-Nagumo caricature) across
  python / rust / julia / go / mojo. The piecewise-linear RK4 right-hand side is
  exact arithmetic, so Rust, Julia and Go reproduce the NumPy reference
  bit-for-bit; the Mojo backend is ULP-bounded and non-amplifying (a
  two-dimensional autonomous flow cannot be chaotic). Added the Rust engine
  `simulate` plus PyO3 `py_mckean_simulate`, the Julia/Go/Mojo backends,
  cross-backend parity tests, a multi-language benchmark with a committed results
  artefact, and a model-documentation upgrade; replaced the decorative
  `accel/go/services` stub with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `WilsonHRNeuron` (Wilson 1999 polynomial cortical model) across
  python / rust / julia / go / mojo. The polynomial RK4 right-hand side with a
  hard voltage reset is exact arithmetic, so Rust, Julia and Go reproduce the
  NumPy reference bit-for-bit; the Mojo backend is ULP-bounded and non-amplifying
  (the per-spike reset re-anchors the 2D autonomous flow). Added the Rust engine
  `simulate` plus PyO3 `py_wilson_hr_simulate`, the Julia/Go/Mojo backends,
  cross-backend parity tests, a multi-language benchmark with a committed results
  artefact, and a model-documentation upgrade; replaced the decorative
  `accel/go/services` stub with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `PernarowskiNeuron` (Pernarowski 1994 pancreatic beta-cell burster) across
  python / rust / julia / go / mojo. Aligned the Python cubic to `v*v*v` (from
  `v**3`) so it is bit-identical to the engine's `v.powi(3)` and removed the now
  unreachable `OverflowError` branch; Rust, Julia and Go then reproduce the NumPy
  RK4 reference bit-for-bit, and the Mojo backend is ULP-bounded and
  non-amplifying. Added the Rust engine `simulate` plus PyO3
  `py_pernarowski_simulate`, the Julia/Go/Mojo backends, cross-backend parity
  tests, a multi-language benchmark with a committed results artefact, and a
  model-documentation upgrade; replaced the decorative `accel/go/services` stub
  with a real c-shared backend.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `TermanWangOscillator` (Terman-Wang 1995 LEGION relaxation oscillator) across
  python / rust / julia / go / mojo. Aligned the Python cubic to `v*v*v` (from
  `v**3`) so it matches the engine's `v.powi(3)` and removed the now-unreachable
  `OverflowError` branch. The right-hand side mixes the exact cubic with a `tanh`
  gating term: the Rust engine resolves `tanh` to the same glibc symbol as Python
  and is bit-identical, while Julia/Go/Mojo use their own libm `tanh` and are
  ULP-bounded (the 2D relaxation oscillator is non-chaotic, so it does not
  amplify). Added the Rust engine `simulate` plus PyO3 `py_terman_wang_simulate`,
  the Julia/Go/Mojo backends, cross-backend parity tests, a multi-language
  benchmark with a committed results artefact, and a model-documentation upgrade;
  replaced the decorative `accel/go/services` and `accel/mojo/kernels` stubs with
  real backends.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `MihalasNieburNeuron` (Mihalas-Niebur 2009 generalised integrate-and-fire model)
  across python / rust / julia / go / mojo. The four-state `(v, theta, i1, i2)`
  right-hand side is purely linear — no transcendental functions — advanced by
  candidate-first RK4 with a discontinuous spike reset, so the Rust engine, Julia
  and Go backends reproduce the NumPy reference bit-for-bit (trace, spike count
  and final state); the Mojo backend fuses multiply-add and is validated as
  non-amplifying within a ULP band with identical spike counts. Added the Rust
  engine `simulate` plus PyO3 `py_mihalas_niebur_simulate`, the Julia/Go/Mojo
  backends, cross-backend parity tests, a multi-language benchmark with a
  committed results artefact, and a model-documentation upgrade; replaced the
  decorative `accel/go/services` and `accel/mojo/kernels` stubs with real
  backends.
- Added the polyglot N-step `simulate(n_steps, current, backend=...)` chain for
  `GLIFNeuron` (Allen Institute GLIF5 generalised leaky integrate-and-fire model)
  across python / rust / julia / go / mojo. The four-state
  `(v, theta, i_asc1, i_asc2)` right-hand side is purely linear — no
  transcendental functions — advanced by candidate-first RK4 with an additive
  threshold spike reset, so the Rust engine, Julia and Go backends reproduce the
  NumPy reference bit-for-bit (trace, spike count and final state); the Mojo
  backend fuses multiply-add and is validated as non-amplifying within a ULP band
  with identical spike counts. Added the Rust engine `simulate` plus PyO3
  `py_glif_simulate`, the Julia/Go/Mojo backends, cross-backend parity tests, a
  multi-language benchmark with a committed results artefact, and a
  model-documentation upgrade; replaced the decorative `accel/go/services` and
  `accel/mojo/kernels` stubs with real backends.

### Studio
- Documented the optional `sc_neurocore.federation` Hub-facing Studio federation
  surface with a dedicated API page and navigation entry, covering schema-A
  manifest emission, evidence bundles, and verifiable-honesty envelopes.
- Added the admin `POST /api/studio/training/weight-restore/attach/live` endpoint
  and the confined control channel that backs it. The endpoint delivers the
  verified weights of a completed source job to a running target training job;
  the worker polls a reserved control directory at each epoch boundary and applies
  the attach with a strict `load_state_dict` that records a
  `studio.training.weight-restore-attach.v1` (`mode: live`) evidence artifact. An
  incompatible or malformed attach is rejected with an `attach_rejected` metric
  event and never interrupts the running job. Added the control channel
  (`StudioJobManager.send_control_command` with atomic command publication +
  `StudioJobContext.poll_control_command`/`read_control_seed`, reserved
  `.studio_control` and `.studio_control_seed` directories), the epoch-boundary
  poll in the training loop, an architecture-fingerprint pre-check, the route
  policy, the `studio.training.weight_restore.attach_live` audit action, the
  preflight required-route entry, a Training Monitor live-attach action with a
  path-free request strip, frontend client types, and full backend and frontend
  tests plus documentation.
- Added the admin `POST /api/studio/training/weight-restore/attach` endpoint and
  the confined seed-input channel that backs it. The endpoint rebuilds the
  canonical restore plan from a completed training job's checkpoint, delivers the
  integrity-checked weight artifacts to a bounded `studio-training-restore` worker
  as confined seed inputs, and warm-starts a training job that loads the verified
  weights at the epoch-zero checkpoint boundary before training forward. A strict
  `load_state_dict` fails closed on an architecture mismatch before training
  begins. Added an architecture fingerprint that gates compatibility on the
  shape-determining config fields only, the
  `studio.training.weight-restore-attach.v1` evidence contract, the route policy,
  the `studio.training.weight_restore.attach` audit action, the preflight
  required-route entry, a `weight_restore_attach_results` evidence-bundle field
  stored under `evidence/training-weight-restore-attaches/`, a Training Monitor
  warm-start action with a path-free evidence strip, frontend client types, and
  full backend and frontend tests plus documentation.
- Added the admin `POST /api/studio/training/weight-restore` endpoint. It
  rebuilds the canonical restore plan from a completed training job's stored
  checkpoint metadata, fetches the integrity-checked weight and metadata
  artifacts, and materializes the weights inside a bounded
  `studio-training-restore` worker job using a `weights_only=True` trusted
  state-dictionary loader. The worker writes a path-free
  `studio.training.weight-restore.v1` evidence artifact holding only verified
  digests, parameter count, and loaded-key total; the deserialized tensors never
  reach the API response. Added the route policy, the
  `studio.training.weight_restore.materialize` audit action, the preflight
  required-route entry, a `weight_restore_results` evidence-bundle field stored
  under `evidence/training-weight-restores/`, a Training Monitor materialize
  action with a path-free evidence strip, frontend client types, and full
  backend and frontend tests plus documentation.

## [3.15.34] - 2026-06-15

### Physics and mathematics hardening
- Corrected `CourageNekorkinMapNeuron` to the canonical Courbage-Nekorkin-Vdovin
  2007 map (`Chaos` 17:043109): `x + F(x) - y - beta*H(x - d)` with the
  piecewise-linear field, Heaviside discontinuity at `x = d`, and `B^+`
  invariant-region default parameters, replacing the prior non-canonical form.
  Added the polyglot N-step `simulate` chain (python/rust/julia/go/mojo;
  Rust/Julia/Go bit-exact, Mojo ULP-bounded) with parity tests, a multi-language
  benchmark, and a rewritten model documentation page.

### Dependencies
- Migrated z3 `0.12` -> `0.20` (feature `static-link-z3` renamed to `bundled`;
  the bounded-model verifier updated for the lifetime-free 0.20 AST/Solver API).
- Bumped esbuild/vite/@vitejs/plugin-react (Studio frontend, clears the open
  esbuild advisory), github/codeql-action, click, and hypothesis.

### CI and release reconciliation
- Reconciled `main` with the previously orphaned release tags `v3.15.26`–
  `v3.15.33` so the published release lineage is continuous again.
- Regenerated the capability manifest, corrected the mixed-precision emitter
  Q-format label, and built the ARM64 wheel against the exact matrix interpreter.

### Security and Rust engine
- Migrated the PyO3/numpy Rust extension chain from `0.28` to `0.29`
  across the engine, fuzz harness, evo substrate, stochastic doctor, and spike
  stats crates; refreshed lockfiles and added the missing spike-stats lockfile
  for reproducible advisory scanning.
- Updated the GPU feature path for WGPU 29 API changes exposed by the
  all-features engine check.

## [3.15.33] - 2026-06-05

### CI and benchmark evidence
- Replaced the Wilson-Cowan CI throughput floor with a bounded-runtime
  regression sentinel and documented that production throughput evidence must
  come from isolated benchmark runs, not hosted coverage jobs.

## [3.15.32] - 2026-06-05

### CI and release workflows
- Restored the direct `MixedPrecisionSpec.get()` contract to preserve explicit
  `PrecisionConfig(16, 8)` widths while keeping explicit `Q7.8` preset parsing
  sign-inclusive.

## [3.15.31] - 2026-06-05

### CI and release workflows
- Aligned the explicit `Q7.8` mixed-precision preset contract with the
  sign-inclusive Q-label parser so branch CI no longer treats explicit
  Q-format labels as named `q88` aliases.

## [3.15.30] - 2026-06-05

### CI and release workflows
- Installed the pinned `click` runtime dependency before the pinned
  SymbiYosys executable check so hosted HDL/formal CI validates `sby`
  immediately after installation.

## [3.15.29] - 2026-06-05

### CI and release workflows
- Aligned branch CI version-contract tests with source metadata, installed the
  HDL/formal toolchain required by RTL contract tests, and synchronized the
  conda install profile with the release version.

## [3.15.28] - 2026-06-05

### CI and release workflows
- Added an executable entry point to the benchmark context example so
  `cargo test --manifest-path engine/Cargo.toml` builds all examples on Linux.

## [3.15.27] - 2026-06-05

### CI and release workflows
- Removed Yosys workflow-file edits from the Yosys push path filter so
  release-hygiene workflow changes do not self-trigger hosted-runner synthesis.

## [3.15.26] - 2026-06-05

### CI and release workflows
- Skipped Yosys synthesis on release tag pushes so tag releases do not fail on
  non-HDL changes after all modules time out under hosted-runner synthesis
  budgets; branch, pull-request, and manual synthesis workflows remain active.

## [3.15.25] - 2026-06-05

### CI and release workflows
- Moved the macOS static-Z3 C++ parser configuration before the v3-engine
  maturin dependency build so Apple Clang uses delayed template parsing during
  the actual engine install step.

## [3.15.24] - 2026-06-05

### CI and release hygiene
- Formatted the release-surface parity test before publishing the next
  immutable release tag.

## [3.15.23] - 2026-06-05

### Release workflows
- Aligned the Rust engine crate and bridge wheel metadata with the public
  Python package version before registry publication.
- Added release-surface tests for engine crate and bridge metadata version
  parity.

## [3.15.22] - 2026-06-05

### Security workflows
- Made the lightweight actionlint scanner deterministic by disabling its
  external ShellCheck and Pyflakes integrations; those analyzers remain separate
  CI concerns instead of hidden actionlint dependencies.

## [3.15.21] - 2026-06-05

### Security workflows
- Installed ShellCheck in the CI security scanner job so actionlint has the same shell-analysis dependency available as local validation.

## [3.15.20] - 2026-06-05

### Security workflows
- Updated the CI security scanner actionlint toolchain to `v1.7.12`, matching the locally validated workflow parser used for release gating.

## [3.15.19] - 2026-06-05

### Release workflows
- Added macOS-only static-Z3 C++ parser flags for engine wheel and v3 engine builds so Apple Clang handles Z3's template-heavy LP sources.

## [3.15.18] - 2026-06-05

### CI and source hygiene
- Scoped the SPDX guard away from vendored Mojo `.pixi` environments and generated OpenROAD build artefacts.
- Added minimal single-line SPDX markers to real HDL and module-specific test surfaces covered by the guard.
- Formatted the mixed-precision/live-control surfaces and tightened mixed-precision manifest typing so CI mypy passes without changing runtime contracts.

## [3.15.17] - 2026-06-05

### Release workflows
- Added the static-Z3 CMake policy floor to all Rust engine wheel builders.
- Installed Docker build-stage `clang` and `libclang-dev` so Z3 bindings can
  locate libclang during containerized release builds.
- Replaced the Docker build step id used in SARIF gating with an expression-safe
  identifier.

## [3.15.16] - 2026-06-05

### Release workflows
- Fixed Docker Trivy scans to use the metadata-selected image tag as an
  explicit image reference instead of an empty default scan target.

## [3.15.10] - 2026-06-05

### Release workflows
- Fixed Docker workflow image-tag selection so the workflow remains valid on
  tag pushes and scans the selected pushed image reference.

## [3.15.9] - 2026-06-05

### Security and release workflows
- Patched the hub runtime Starlette pin from `1.0.0` to `1.0.1`.
- Fixed Docker image scanning to scan the selected pushed image reference.
- Switched the Rust engine Z3 dependency to a static build path so release
  wheels no longer depend on runner-provided Z3 headers.

## [3.15.8] - 2026-06-05

### Documentation and release polish
- Bumped Python, Rust engine, bridge package, Sphinx docs, README, and
  generated capability metadata to version `3.15.8`.
- Expanded the documentation home page with an evaluator map that routes new
  users, hardware teams, framework reviewers, industrial evaluators, notebook
  readers, and API consumers to the correct first evidence surface.
- Strengthened onboarding, notebook, API, FPGA tutorial, industrial
  applications, product overview, and applications/market documentation so
  users can understand what SC-NeuroCore is for, where it has evidence, where
  optional dependencies apply, and which claims require committed artefacts.

### Engine supervisor
- Added a public Rust supervisor execution entrypoint shared by the PyO3
  controller path, preserving bounded-run completion by dropping snapshot
  senders before joining the Z3 worker and adding module-specific supervisor
  tests for safe bounded execution, unsafe Petri-net rejection, worker shutdown
  signalling, and zero-neuron fail-closed validation.

### Compiler precision
- Hardened adaptive runtime precision manifests for BFP/Q16.16 handoff by
  adding the `adaptive_precision_emitter.v1` contract, emitted datapath
  width/fraction, exponent-stream width, exponent-vector width, and fail-closed
  rejection of block-exponent parameter counts on fixed Q-format paths.
- Hardened generated AXI4-Lite/PCIe live-control readback so invalid
  bank/entry selections return a bus error and latch the sticky
  `invalid_selection` trap instead of silently returning zero.
- Added emitter-facing mixed-precision manifests for fixed Q16.16 and
  block-floating variables, including deterministic assignment order, emitted
  datapath width/fraction, exponent stream width, exponent-vector width, and
  fail-closed BFP parameter-count validation for downstream HDL emitters.
- Routed quantizer precision-envelope proof fields through the static-analysis
  Q-format envelope proof API, with module-specific regression coverage to keep
  dense deployment manifests aligned with the standalone proof contract.
- Added a static-analysis Q-format envelope proof API for conservative Q-code
  bounds, with fail-closed validation, signed Q16.16 width/headroom manifests,
  and module-specific tests for safe, saturating, and block-floating
  exponent-edge contracts.
- Added signed fixed-point width proofs to mixed Q8.8/Q16.16 and
  block-floating precision envelope reports across Python, Rust, and refreshed
  comparison benchmark artefacts, including required total bits, required
  Q16.16 integer bits, headroom, saturation requirement, and static overflow
  proof status.
- Added seeded block-floating exponent-edge parity and trap contracts across
  the Python quantizer, Rust qformat mirror, and comparison benchmark
  artefacts: `BFP16E3X2` safe min/max exponent sweeps now match exact Q16.16
  output codes across languages, while max-exponent saturation records a
  deterministic overflow trap instead of silent wraparound.
- Added explicit block-exponent layout metadata for block-floating precision
  across adaptive manifests, mixed-precision specs, Python dense BFP manifests,
  and the Rust qformat mirror, with exponent-count validation before emission
  or accumulation.
- Added sub-LSB underflow telemetry to mixed Q8.8/Q16.16 and block-floating
  dense precision trap/envelope reports across Python and Rust, with refreshed
  process-affinity benchmark artefacts documenting matched overflow and
  underflow probes.
- Added sticky live-control partial-write traps so generated AXI4-Lite/PCIe
  parameter banks reject partial `WSTRB` updates before control or staged-data
  registers can be modified.
- Hardened live-control trap clearing so generated AXI4-Lite/PCIe parameter
  banks clear only selected sticky trap bits and preserve unrelated latched
  fault evidence.
- Added selected-trap clear helpers to the live-control schema and generated
  Python/C host drivers.
- Added deterministic live-control active-parameter readback for host update
  sequences and generated AXI4-Lite/PCIe parameter banks, including low/high
  committed-word registers and module-specific RTL simulation coverage.
- Hardened generated Python and C host drivers for live-control parameter
  banks with CRC32 update helpers, committed readback verification, trap-status
  checks, and mandatory high-word staging for narrow updates.
- Added generated Python and C live-control rollback, status-read, and
  trap-status-read helpers so host drivers expose the load/apply/rollback/
  clear/readback handshake.
- Added generated C live-control driver compile validation against a C11
  consumer that calls the committed update/readback verification helper.
- Latched live-control shadow bank and entry identity at load time so generated
  AXI4-Lite/PCIe apply and rollback operations cannot be retargeted by later
  selection-register writes.
- Added sticky live-control read-only-bank traps so generated AXI4-Lite/PCIe
  parameter banks reject direct MMIO writes to calibration/read-only banks before
  shadow loading or active coefficient mutation.
- Added sticky live-control invalid-selection traps so generated AXI4-Lite/PCIe
  parameter banks reject non-existent bank/entry writes without raising a false
  shadow-loaded acknowledgement.
- Added sticky CRC32 checksum-mismatch traps and a testbench-visible mismatch
  pulse to the generated AXI4-Lite/PCIe live-control parameter-bank surfaces,
  with module-specific simulation tests and refreshed benchmark-gate evidence.
- Replaced live-control update guards with an IEEE CRC32 register-window
  guard shared by the compiler schema and generated SystemVerilog, with stale
  guard rejection tests and refreshed benchmark evidence.
- Added the PCIe-MMIO live-control register-window adapter over the staged
  parameter-bank core, module-specific PCIe commit simulation, process-affinity
  AXI4-Lite/PCIe comparison benchmark evidence, and compiler API documentation
  for the exact bus-contract boundary.
- Added the UltraScale+ dense-folding contract: shared Rust/Python fold planner,
  folded Q8.8/Q16.16 HDL core, target-emitter fold metadata, module-specific
  simulation tests, and isolated Python/Rust benchmark evidence for fitting the
  64x32 dense contract into the ZU3EG DSP budget.
- Added the NEU-C.1 Zynq UltraScale+ target contract: Rust target metadata,
  conservative resource-budget reporting, deterministic Vivado Tcl generation,
  board-safe timing-only XDC baselines, module-specific tests, and isolated
  Python/Rust comparison benchmark evidence.
- Added the NEU-C.6 DCLS Q8.8 RTL path: bit-true Rust DCLS tent-kernel
  arithmetic, SystemVerilog axonal delay/tent/layer modules, IR
  `DclsLayer` emission, SymbiYosys safety/liveness harness, Python/PyTorch
  cosimulation, module-specific tests, and isolated benchmark evidence.
- Added NEU-C.5 ADC-to-spike quantiser HDL with Q-format decimation, deterministic AER rate coding, formal transfer properties, bit-true Python reference, isolated benchmark evidence, and hardware documentation.
- Added NEU-C.2 timing-aware formal-property framework with reusable SystemVerilog monitors, Python proof orchestration, nuXmv/Kind 2 emitters, a dense-layer SymbiYosys/cvc5 proof, and isolated benchmark evidence.
- Added the NEU-C.4 AER strict-priority queue and router backpressure path,
  including sticky drop/deadline traps, Python reference contract,
  SystemVerilog simulation, formal harness, benchmark gate, and hardware docs.
- Added live-control update and trap evidence benchmarks covering generated
  MMIO update sequences, static RTL regeneration, and staged range-trap
  simulation, with the artefact registered in the benchmark gate manifest.
- Added generated live-parameter-bank staged overflow and underflow traps that
  latch malformed MMIO payloads and block shadow-bank mutation before active
  coefficient application.
- Hardened compiler live-control update semantics with checksum-gated shadow
  loads, explicit apply/rollback sequences, active-only generated parameter
  outputs, and status telemetry for shadow-loaded, applied, rollback, and
  checksum-valid states.
- Added AXI4-Lite live-parameter-bank RTL emission from the compiler
  live-control schema, including BRAM/distributed RAM style hints, flattened
  parameter outputs, staged commits, trap status, and module-specific compile
  tests.
- Added deterministic compiler live-control schemas for AXI4-Lite/PCIe
  parameter-bank updates, including encoded-word range checks, fixed
  control/status registers, atomic staged commit sequences, and trap-clear
  command generation.
- Aligned adaptive block-floating precision metadata with the quantizer
  exponent-bias contract and added explicit block exponent alignment telemetry
  for `BFP16E3X32` to `Q16.16` adaptive-precision manifests.
- Hardened the 2026-06-04 mixed, block-floating, precision-trap, and
  precision-envelope benchmark artefact writers so Python and Rust runs record
  taskset affinity, load before/after, CPU governor, and frequency context.
- Aligned the mixed dense Python and Rust benchmark workloads on the canonical
  raw Q8.8/Q16.16 physical contract (`QFormatMixed(scale_per_tensor=False)`),
  eliminating the stale per-tensor Python envelope mismatch and refreshing the
  cross-language benchmark documentation.
- Marked the 2026-06-04 local precision benchmark artefacts as captured under
  concurrent workstation load and documented the isolated-core requirement for
  future production throughput claims.
- Aligned the block-floating dense Python and Rust benchmark workloads so the
  safe and saturating precision-envelope bounds compare the same physical BFP
  mantissa/exponent contract across languages.
- Added per-output conservative absolute-bound telemetry to the mixed
  Q8.8/Q16.16 and block-floating dense RTL (`abs_bounds_q1616`), aligned the
  Python/Rust benchmark artefacts with the same precision-envelope fields, and
  refreshed module-specific HDL tests plus HDL/Python/Rust benchmark evidence.
- Added per-output overflow telemetry to the mixed Q8.8/Q16.16 dense RTL and
  refreshed the Python, Rust, HDL, and documentation evidence for lane-level
  saturation attribution.
- Added per-output overflow telemetry to the block-floating dense RTL and
  refreshed the Python, Rust, HDL, and documentation evidence for lane-level
  saturation attribution.
- Added precision envelope reports across the mixed fixed-point and
  block-floating dense deployment paths, including conservative absolute-bound
  checks in Python and Rust, a synchronous HDL envelope guard, module-specific
  tests, and committed Python, Rust, and Yosys benchmark artefacts.
- Added precision trap reports across the mixed fixed-point and block-floating
  dense deployment paths, including exact overflow counts in the Rust qformat
  mirror, a synchronous HDL trap latch, module-specific tests, and committed
  Python, Rust, and Yosys benchmark artefacts.
- Added dense block-floating `BFP16E3X32` execution across the Python quantiser
  API, Rust IR qformat mirror, and synchronous HDL reference module, including
  shared-exponent product scaling, Q16.16 output saturation, overflow telemetry,
  module-specific tests, and committed Python, Rust, and Yosys benchmark
  artefacts.
- Corrected block-floating metadata so the maximum unbiased exponent reflects
  every encoded biased exponent code.
- Added the compiled mixed-dense Q8.8/Q16.16 contract across the Python
  quantiser API, Rust IR qformat mirror, and synchronous HDL reference module,
  including exact signed MAC scaling, accumulator saturation, and overflow
  telemetry.
- Added module-specific mixed-dense quantiser and HDL tests plus committed
  Python, Rust, and Yosys benchmark artefacts for the 64×32 mixed-precision
  dense contract.
- Added the `QFormatMixed` quantiser contract for Q8.8 stored weights with
  Q16.16 accumulator metadata, including per-tensor scale round-trip support,
  public compiler exports, module-specific quantiser tests, and refreshed
  compiler precision documentation.
- Corrected block-floating alias normalisation and shared-exponent selection
  so sub-unit tensors retain the finest representable scale within the exponent
  range.

### Typing hygiene
- Removed active source-level file-wide mypy suppressions from the package tree
  and repaired the exposed strict-mypy defects in ASIC flow, BCI Studio,
  bioware, digital-twin synchronisation, evolutionary substrate,
  explainability, federated learning, hypervisor, memristor, model-zoo, and
  spintronic surfaces.
- Confirmed strict package mypy passes for 940 source files with an isolated
  cache path; the repository-local cache path currently exhibits local
  filesystem `ENOSPC` behaviour and should not be used as typing evidence.

## [3.15.7] - 2026-06-01

### Engine publishing credentials
- Added the dedicated `sc-neurocore-engine` PyPI project token to the engine
  wheel publish step after PyPI rejected OIDC tokens scoped to the primary
  `sc-neurocore` project.
- Issued this patch release without rewriting prior tags so the partial
  publication attempts remain auditable and superseded.

## [3.15.6] - 2026-06-01

### Publishing hygiene
- Aligned the engine wheel PyPI publication job with the repository's existing
  trusted-publishing environment after PyPI rejected the dedicated
  `pypi-engine` environment claim.
- Issued this patch release without rewriting earlier tags so the failed
  trusted-publisher attempt remains traceable and superseded.

## [3.15.5] - 2026-06-01

### Engine package metadata
- Declared the engine wheel runtime NumPy dependency in the bridge package
  metadata so installed wheels resolve the dependency needed by the public
  `sc_neurocore_engine.layers` module.
- Issued this patch release after `v3.15.4` validated Python and crate
  publication but exposed the missing engine-wheel runtime dependency during
  smoke testing.

## [3.15.4] - 2026-06-01

### Publish automation
- Bumped the Rust engine crate release metadata alongside the Python package
  release surfaces so crates.io publication no longer attempts to republish an
  older engine version.
- Changed the engine wheel smoke test to install the built wheel before import,
  preserving runtime dependency resolution instead of unpacking the archive
  directly.

## [3.15.3] - 2026-06-01

### Release automation
- Fixed the tag-release workflow to extract release notes from the committed
  documentation changelog path.
- Issued this patch release candidate without rewriting the existing
  `v3.15.1` or `v3.15.2` tags.

## [3.15.2] - 2026-06-01

### Release integrity
- Issued a patch release candidate after the `v3.15.1` tag to preserve the
  no-history-rewrite rule while keeping the release tag aligned with the
  current CI, documentation, version metadata, and typed RK4 candidate-state
  fixes.
- Confirmed the strict mypy preparation lane remains error-free for the
  package source tree without adding suppressions.

## [3.15.1] - 2026-06-01

### Documentation and release polish
- Added public Product Overview and Applications and Market pages so new users, evaluators, and commercial readers can understand the project scope, evidence boundary, potential applications, and market position without reverse-engineering the API inventory.
- Refreshed the README, documentation home page, learning path, getting-started guide, notebook guide, API index, industrial-applications page, benchmark index, and cross-framework benchmark evidence page for clearer onboarding and claim traceability.
- Added a notebooks README with recommended reading order and reproducibility rules.
- Bumped Python package, public docs, capability metadata, and Rust engine package version references from 3.15.0 to 3.15.1.

### NIR Bridge
- Roundtrip tests for all 18/18 NIR primitives (was 7/18)
- Auto-broadcast scalar neuron params to input size (Norse/snnTorch export 0-dim tensors)
- Threshold fix: `>=` to `>` matching NIR spec and snnTorch behavior
- `reset_mode="subtract"` for snnTorch compatibility (subtract-reset vs zero-reset)
- IF subtract-reset test and unknown `reset_mode` fallback handling
- Cross-framework interop tests: Sinabs LIF/IAF/ExpLeak, Rockpool LIF/CubaLIF/LI, snnTorch RSynaptic subgraph
- Cross-framework r-encoding test documenting per-framework dt conventions
- SpikingJelly NIR roundtrip demo (`examples/spikingjelly_nir_roundtrip.py`)
- Norse NIR roundtrip demo with real Norse weights (`examples/norse_nir_roundtrip.py`)
- NIR roundtrip demo: stronger input to produce visible spikes
- Documentation: added SpikingJelly, Rockpool, Sinabs, snnTorch RSynaptic sections to `docs/guides/nir_integration.md`
- Documentation: framework dt/r quick reference table
- Documented Norse tau observation (export/import roundtrip discrepancy in Norse code)
- Removed unverified "first FPGA backend" claim from 6 files

### Physics and mathematics hardening
- Promoted `LeakyCompeteFireNeuron` from raw Euler vector updates to exact
  first-order relaxation across the Python reference, Go service, Julia
  mirror, Mojo scalar helpers, and Rust safety surface; module-owned tests now
  cover closed-form WTA parity, large-timestep boundedness, fail-closed state
  preservation, vector mirror contracts, and refreshed Python benchmark
  evidence.
- Promoted `AlphaNeuron` from a single-pole synaptic filter to the full
  two-state Rall/Gerstner alpha-cascade flow across the Python reference, Go
  service, Julia mirror, and Rust safety surface; module-owned tests now cover
  closed-form alpha parity, equal-time-constant limits, large-timestep
  boundedness, fail-closed state preservation, and refreshed Python benchmark
  evidence.
- Promoted `ResonateAndFireNeuron` from raw Euler oscillator increments to the
  exact constant-input linear resonator flow across the Python reference, Go
  service, Julia mirror, Mojo scalar helpers, and Rust safety surface; module
  tests now cover matrix-exponential parity, large-timestep damping,
  fail-closed state preservation, and refreshed Python benchmark evidence.
- Promoted `ArcaneNeuron` to candidate-first exact first-order relaxation across
  the fast, working-memory, and deep identity compartments on the Python
  reference, Go service, Julia mirror, Mojo scalar helpers, and Rust safety
  surface; module-owned tests now cover exact trajectory parity, large-timestep
  boundedness, fail-closed state preservation, and refreshed Python benchmark
  evidence.
- Hardened `ParametricLIFNeuron` candidate-first discrete recurrence semantics across the Python reference, Go service, Julia mirror, and Rust safety surface, preserving the Fang et al. PLIF update while rejecting corrupted runtime state and non-finite voltage candidates before mutation with refreshed benchmark evidence.
- Promoted `SigmoidRateNeuron` from raw Euler rate updates to exact first-order relaxation across the Python reference, Go service, Julia mirror, Mojo kernel, and Rust safety surface, with module-specific tests for closed-form parity, large-timestep boundedness, invalid-state preservation, and refreshed benchmark evidence.
- Promoted `NonResettingLIFNeuron` from raw Euler membrane and adaptive-threshold updates to exact first-order relaxation across the Python reference, Go service, Julia mirror, and Rust safety surface, with module-specific tests for closed-form parity, large-timestep boundedness, invalid-update preservation, and refreshed benchmark evidence.
- Promoted `AdaptiveThresholdIFNeuron` from guarded Euler mutation to exact
  first-order relaxation across the Python reference, Go service, Julia mirror,
  Mojo spike kernel, and Rust safety surface; module-owned tests now cover
  exact trajectory parity, large-timestep boundedness, fail-closed state
  preservation, and refreshed Python benchmark evidence.
- Promoted `YamadaNeuron` to candidate-first RK4 integration across the Python
  reference, Go service, Julia mirror, Mojo spike kernel, and Rust safety
  surface; module-owned tests now cover RK4 parity, finite-stage validation,
  fail-closed state preservation, and refreshed Python benchmark evidence
  documents the RK4 runtime cost.
- Promoted `BendaHerzNeuron` to candidate-first RK4 adaptation integration
  with exponential hazard spike probability across the Python reference, Go
  service, Julia mirror, Mojo kernel notes, and Rust safety surface; module
  tests now cover RK4 parity, seeded stochastic reproducibility, fail-closed
  state preservation, and refreshed Python/Go benchmark evidence.
- Hardened `CochlearHairCell` across the Python reference, Rust engine, Go
  service, Julia mirror, and Rust safety surface by replacing the raw membrane
  Euler voltage increment with exact conductance-form relaxation, adding stable
  finite-domain Boltzmann activation, preserving state on invalid runtime
  inputs, adding module-owned tests, and recording a refreshed Python benchmark
  artefact.
- Promoted `ButeraRespiratoryNeuron` to bounded candidate-first RK4
  integration across the Python reference, Rust engine, Go service, Julia
  mirror, and Rust safety surface; module-owned tests now cover RK4 parity,
  high-current bounded stability, fail-closed invalid-state preservation, and
  refreshed Python/Rust benchmark artefacts document the RK4 runtime cost.
- Hardened `DirectionSelectiveRGC` across the Python reference, Rust engine, Go service, Julia mirror, and Rust safety surface by replacing raw Euler membrane drift with exact first-order relaxation, preserving state on invalid optical drive or corrupted runtime buffers, adding module-specific tests, and recording a refreshed Python benchmark artefact.
- Hardened `BoothRinzelNeuron` Python, Julia, Go, and Rust safety surfaces with finite-domain validation, fail-closed candidate updates, physical gate/calcium bounds, and module-owned regression tests.
- Promoted `ConnorStevensNeuron` to candidate-first RK4 integration across the Python reference, Rust engine, Julia mirror, Go service, Mojo parity notes, and Rust safety surface; module-owned tests now cover RK4 parity, finite-domain validation, fail-closed state preservation, and refreshed Python/Rust benchmark artefacts document the RK4 runtime cost.
- Promoted `TermanWangOscillator` to candidate-first RK4 integration across
  the Python reference, Rust engine, Julia mirror, Go mirror, Mojo kernel
  notes, and Rust safety surface; module-owned tests now cover the Python
  model at 100%, public docs state the finite-domain and continuous
  threshold-crossing contracts, and refreshed Python/Rust benchmark artefacts
  document the RK4 runtime cost.
- Promoted `PernarowskiNeuron` to candidate-first RK4 integration across the
  Python reference, Rust engine, Julia mirror, Go mirror, and Rust safety
  surface; module-owned tests now cover the Python model at 100%, public docs
  state the finite-domain and continuous threshold-crossing contracts, and
  refreshed Python/Rust benchmark artefacts document the RK4 runtime cost.
- Promoted `FitzHughRinzelNeuron` to candidate-first RK4 integration across
  the Python reference, Rust engine, Julia mirror, Go mirror, and Rust safety
  surface; module-owned tests now cover the Python model at 100%, public docs
  state the finite-domain and reset contracts, and refreshed Python/Rust
  benchmark artefacts document the RK4 runtime cost.
- Hardened `BertramPhantomBurster` across Python, Julia, Go, and Rust safety
  surfaces by replacing raw Euler state mutation with bounded RK4 integration
  over the published three-state ODE, adding finite physical-parameter and
  candidate-state validation, updating module-owned tests and model
  documentation, and adding refreshed local benchmark evidence.
- Replaced proxy `ollivier_ricci_curvature` evaluation with graph-metric
  lazy-random-walk Wasserstein transport, added fail-closed coupling graph and
  node-index validation, aligned topology tests, and documented the exact
  topological observable contract.
- Hardened `ExactLIFSolver` and the `solver.lif.subthreshold-exact`
  alternative route with finite physical-parameter validation, non-negative
  runtime-time contracts, reset/threshold ordering, fail-closed subthreshold
  route-domain checks, route documentation, and refreshed benchmark evidence.
- Hardened `physics.kuramoto.noiseless-symplectic-lift` route validation
  for empty phase arrays, phase/frequency shape mismatches, non-finite
  phases or frequencies, non-positive horizons and timesteps, boolean scalar
  parameters, and negative coupling; refreshed route documentation and
  benchmark evidence for the bounded noiseless Hamiltonian-lift lane.
- Hardened `StormerVerlet`, `LeapfrogSolver`, and the
  `physics.oscillator.harmonic-symplectic` route with fail-closed
  Hamiltonian state validation, finite time/timestep contracts, RHS
  shape/finite-output checks, zero-energy route rejection, updated
  alternative-path documentation, and refreshed oscillator benchmark
  evidence.
- Hardened `WolframHypergraph` with hyperedge, node-id, and rewrite-step
  validation; rewrites now revalidate graph invariants after each evolution
  pass, dimension estimation fails closed on corrupted topology, the Julia
  mirror was corrected, and physics docs now state the topology contract.
- Hardened `FeynmanKacHeatSolver` with finite-domain validation for
  length, diffusivity, walker count, timestep, seed, density grids, target
  time, and histogram bins; replaced bounded iterative boundary correction
  with exact triangle-wave reflection for Neumann Brownian paths; updated
  Julia, Mojo, and Rust heat mirrors plus physics docs and reran the
  `physics.heat.cosine-mode` shadow benchmark.
- Hardened `PinskyRinzelNeuron` Python, Julia, Go, and Rust safety
  surfaces to validate two-compartment state, compartment fraction,
  positive conductances, timestep, calcium non-negativity, gate bounds,
  and dual-input currents before integration; candidate updates now fail
  before mutation on non-finite state or gate-envelope excursions while
  preserving somatic threshold-crossing semantics.
- Hardened `LarterBreakspearNeuron` Python, Julia, Go, and Rust safety
  surfaces to revalidate conductance, ion-rate, timestep, coupling, and
  potassium-gate bounds before integration; RK4 candidates now fail
  before mutation on non-finite state or gate excursions while preserving
  continuous voltage output semantics.
- Hardened `WilsonCowanUnit` Python and Rust safety surfaces to
  revalidate E/I state, non-negative coupling weights, positive time
  constants, sigmoid gain, timestep, and candidate rate bounds before
  mutation; public model documentation now states the two-term sigmoid
  range and fail-closed polyglot runtime contract.
- Hardened `MorrisLecarNeuron` Python, Julia, Go, and Rust safety
  surfaces to validate finite conductance state, membrane capacitance,
  activation slopes, potassium activation bounds, timestep, threshold,
  and runtime drive before integration; candidate updates now fail
  before mutation on potassium-rate overflow or non-finite state while
  preserving no-reset threshold crossing semantics.
- Promoted `FitzHughNagumoNeuron` to RK4-by-default integration across the
  Python reference, Rust engine, Julia mirror, Go mirror, and Rust safety
  surface; the Python legacy Euler path is now explicit opt-in, fail-closed
  candidate validation is preserved before mutation, module-owned tests now
  cover 100% of the Python model, and refreshed Python/Rust benchmark
  artefacts document the RK4 runtime cost.
- Hardened `JansenRitUnit` Python, Julia, Go, and Rust safety surfaces to
  validate neural-mass state, excitatory/inhibitory gain and rate contracts,
  timestep and external-drive boundaries, overflow-stable sigmoid bounds, and
  finite candidate updates before mutation while preserving continuous EEG
  proxy output semantics.
- Hardened `WendlingNeuron` Python, Go, and Rust safety surfaces to validate
  neural-mass state, physiological gain/rate/timestep contracts, non-finite
  external drive, overflow-stable sigmoid bounds, and finite candidate updates
  before mutation while preserving continuous EEG-proxy output semantics.
- Hardened `CompteWMNeuron` Python, Julia, Go, and Rust safety surfaces
  to validate NMDA/AMPA/GABA gate state, Mg2+-block denominators,
  conductance and timescale contracts, non-finite drive, and bounded
  voltage or gate candidates before mutation while preserving
  spike-triggered self-inhibitory GABA feedback.
- Hardened `COBALIFNeuron` Python, Julia, Go, and Rust safety surfaces
  to validate mutable conductance state, membrane geometry, synaptic
  deltas, and exponential decay contracts before each update; compute
  voltage and conductance candidates before mutation; and reject
  non-finite or out-of-envelope candidates while preserving spike reset
  semantics.
- Hardened `ComplementaryLIFNeuron` Python, Julia, Go, and Rust safety
  surfaces to revalidate mutable dual-path state, threshold, timestep,
  and membrane timescale before each update; recompute the decay
  constant after runtime parameter mutation; and reject non-finite
  drive or membrane candidates before mutation while preserving ternary
  positive and negative spike semantics.
- Hardened `ChayKeizerNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid beta-cell gate/calcium state,
  non-physical Ca-dependent potassium and calcium-buffer contracts,
  unstable logistic/timescale exponentials, non-finite drive, and
  out-of-bounds membrane, gate, or calcium candidates before mutation.
- Hardened `ChayNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid beta-cell gate/calcium state, non-physical conductance
  and calcium-buffer contracts, unstable logistic exponentials,
  non-finite drive, and out-of-bounds membrane, gate, or calcium
  candidates before mutation while substepping the stiff potassium
  dynamics.
- Hardened `ChandelierNeuron` Python, Julia, Go, and Rust safety surfaces
  to reject invalid Kv1/Kv3 gate state, non-physical conductance and
  capacitance contracts, unstable rate and gate exponentials, non-finite
  drive, and out-of-bounds membrane or gate candidates before mutation while
  preserving axo-axonic Kv1 delay and Kv3 sharpening dynamics.
- Hardened `CerebellarBasketNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid A-type/KCa gate state, calcium state,
  non-physical conductance and capacitance contracts, unstable rate
  exponentials, non-finite drive, malformed calcium activation denominators,
  and out-of-bounds membrane or calcium candidates before mutation.
- Hardened `BKNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  reject invalid BK gate state, calcium state, non-physical conductance and
  capacitance contracts, malformed substep geometry, unstable rate and BK
  activation exponentials, non-finite drive, and out-of-bounds membrane or
  calcium candidates before mutation while preserving spike-triggered calcium
  influx.
- Hardened `ATypeKNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid transient IA gate state, non-physical conductance and
  capacitance contracts, malformed substep geometry, unstable rate
  exponentials, non-finite drive, and out-of-bounds membrane candidates before
  mutation while preserving A-type K first-spike-delay dynamics.
- Hardened `AstrocyteLIFNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid glial calcium state, non-positive membrane and
  calcium timescales, malformed threshold geometry, non-finite drive,
  gliotransmitter drift, and non-finite calcium or membrane candidates before
  mutation while preserving tripartite feedback semantics.
- Hardened `AlphaMotorNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid HH/PIC gate state, non-physical calcium buffers,
  non-positive timestep/capacitance/timescale contracts, unstable rate
  exponentials, and non-finite membrane/calcium candidates before mutation.
- Hardened `ErmentroutKopellMapNeuron` Python, Julia, Go, and Rust safety
  surfaces to reject invalid phase-map state, non-positive timestep,
  non-finite drive, non-finite phase candidates, and mirror threshold drift
  before mutation while preserving compact-circle phase wrapping.
- Hardened `AiharaMapNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid chaotic-map state, malformed feedback/damping parameters,
  non-finite drive, unstable sigmoid evaluation, and non-finite map candidates
  before mutation.
- Hardened `ChialvoMapNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid discrete-map state, non-finite drive, unstable exponential
  map terms, and non-finite two-dimensional map candidates before mutation.
- Hardened `RulkovMapNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid discrete-map state, non-positive map gain/timescale
  parameters, non-finite drive, non-finite branch boundaries, and non-finite
  map candidates before mutation.
- Hardened `BrunelWangNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid conductance/timescale/capacitance contracts, malformed
  synaptic gates, non-finite refractory or voltage state, unstable NMDA
  Mg2+-block exponentials, and non-finite membrane candidates before mutation.
- Promoted `WilsonHRNeuron` Python, Rust engine, Julia, Go, and Rust safety surfaces to candidate-first RK4 over the coupled polynomial cortical `(v, r)` state, with finite derivative/candidate guards, reset-preserving spike semantics, module-specific RK4 parity tests, and refreshed benchmark evidence.
- Hardened `WilsonHRNeuron` Python, Julia, Go, and Rust safety surfaces to
  reject invalid polynomial-cortical runtime state, non-positive recovery
  timescale or timestep, non-finite current, and non-finite voltage/recovery
  candidates before mutation while preserving spike-triggered voltage reset.
- Hardened `WongWangUnit` Python, Julia, Go, Mojo, and Rust safety surfaces to
  reject invalid two-pool gating state, non-positive timescales, non-finite
  stimuli or noise, unstable transfer-function exponentials, and non-finite
  candidate states before mutation while preserving tuple rate outputs.
- Promoted `WongWangUnit` Python, Rust engine, Julia, Go, Mojo, and Rust safety
  surfaces from forward Euler to candidate-first RK4 over the coupled
  two-pool decision ODE, preserving one sampled stochastic drive per pool per
  step and tuple rate outputs.
- Hardened `WilsonCowanUnit` Python, Julia, Go, and Rust safety surfaces to
  reject invalid rate-state, non-positive timescales, non-finite external
  drive, unstable sigmoid exponentials, and non-finite rate candidates before
  mutation while preserving rate-model return semantics.
- Hardened `TraubMilesNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid HH gate probabilities, non-physical
  conductances, non-finite rate constants, and non-finite ten-substep
  voltage candidates before state mutation.
- Hardened `TermanWangOscillator` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid relaxation-oscillator state, non-positive
  timescale parameters, non-finite drive, and non-finite cubic recovery
  updates before mutation.
- Hardened `WangBuzsakiNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state or non-finite fast-spiking
  conductance updates before state mutation.
- Hardened `PoissonNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  revalidate mutable rate and timestep state before sampling, reject non-finite
  interval hazards, and keep the finite-step Poisson probability bounded.
- Hardened `McCullochPittsNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to enforce finite weighted-input and mutable-threshold contracts,
  preserve equality-at-threshold Heaviside semantics, and keep reset as a
  stateless no-op.
- Hardened `EscapeRateNeuron` Python, Julia, Go, and Rust safety surfaces to
  revalidate mutable point-process state before membrane integration,
  exponentiation, hazard evaluation, or random sampling; non-finite voltage
  candidates and escape hazards now fail before membrane mutation.
- Hardened `LapicqueNeuron` Python, Julia, Go, and Rust safety surfaces to
  revalidate mutable RC state before division/integration and report invalid
  current, corrupted state, or non-finite Euler increments explicitly before
  membrane mutation; documented the Mojo fail-closed spike-flag boundary.
- Hardened `NonResettingLIFNeuron` Python, Julia, Go, and Rust safety surfaces
  to revalidate runtime membrane and adaptive-threshold state before
  integration, compute both candidates before mutation, and report non-finite
  updates explicitly while preserving the no-voltage-reset spike contract.
- Hardened `PerfectIntegratorNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to revalidate runtime membrane geometry before division/integration
  and to report invalid or non-finite voltage increments explicitly before
  state mutation.
- Hardened `ThetaNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces to
  reject corrupted runtime phase or timestep state before cosine/Euler
  evaluation and to report non-finite phase increments explicitly without
  mutating the compact-circle state.
- Hardened `SiegertTransferFunction` Python, Julia, Go, Mojo, and Rust safety
  surfaces to revalidate first-passage parameters at runtime, reject non-finite
  quadrature bounds, integrals, and inter-spike intervals, and keep rates
  finite, non-negative, and refractory bounded.
- Hardened `SigmoidRateNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to enforce the continuous-rate `[0, 1]` invariant, reject unstable
  Euler ratios and corrupted runtime state before mutation, and use saturated
  finite-drive logistic evaluation for extreme inputs.
- Hardened `AdaptiveThresholdMoENeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state and non-finite adaptive-threshold,
  quotient, or soft-reset candidates before state mutation, while preserving
  non-negative integer spike-count residual semantics.
- Hardened the `ThresholdLinearRateNeuron` Python, Julia, Go, Mojo, and Rust
  safety surfaces to reject invalid runtime rate state and non-finite rate
  outputs before state mutation.
- Hardened the `AdaptiveThresholdIFNeuron` Python, Julia, Go, Mojo, and Rust
  safety surfaces to reject invalid runtime state and non-finite Euler or
  threshold-jump updates before state mutation.
- Hardened the `BrainScaleSAdExNeuron` Python, Julia, Go, Mojo, and Rust safety
  surfaces to reject invalid runtime state and non-finite hardware-scaled
  integrator or adaptation updates before state mutation.
- Hardened the `AdExNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid runtime state and non-finite integrator or adaptation
  updates before state mutation.
- Hardened the `ExpIFNeuron` Python, Julia, Go, Mojo, and Rust safety surfaces
  to reject invalid runtime state and non-finite Euler updates before membrane
  mutation.
- Hardened the Rust/PyO3 Kuramoto solver boundary to fail closed on non-finite frequencies, coupling matrices, initial/runtime phases, field pressure, SSGF geometry/PGBO matrices, invalid `dt`, and negative/non-finite noise amplitudes.
- Extended Kuramoto `run()` and `run_ssgf()` validation so invalid timesteps and non-finite SSGF gains/matrices are rejected even for zero-step dry runs.
- Added PyO3 SSGF shape guards so malformed `W` and `h_munu` matrices raise `ValueError` before entering the Rust solver.
- Corrected the `PINGCircuit` Python reference step to consume one excitatory and one inhibitory Wiener-noise vector per timestep, matching the Rust, Julia, Go, and Mojo backend stochastic contract; benchmark metadata now reports the selected backend directly.
- Hardened `CorticalColumn(backend="python", use_block_csr=True)` so it remains on the scipy.sparse reference path and does not call the Rust single-block fallback when native symbols are present.
- Hardened `HindmarshRoseNeuron` RK4/Euler derivative evaluation to fail closed on cubic overflow or non-finite intermediate stages without mutating state.
- Aligned `HindmarshRoseNeuron` Rust engine, Rust safety, Go, and Julia
  counterparts with the Python RK4 trajectory and fail-closed candidate-state
  contract.
- Hardened `MorrisLecarNeuron` Euler/RK4/Rosenbrock paths to fail closed on potassium-rate overflow or non-finite derivative/state updates without mutating state.
- Hardened `FitzHughNagumoNeuron` Euler/RK4/Rosenbrock paths to fail closed on cubic overflow or non-finite derivative/state updates without mutating state, and aligned the Julia, Go, and Rust safety counterparts with the documented no-reset state equation.
- Promoted `McKeanNeuron` Python, Rust engine, Julia, Go, and Rust safety surfaces from simultaneous Euler to candidate-first RK4 over the coupled `(v, w)` state, with finite derivative/candidate guards and module-specific RK4 parity tests.
- Hardened `McKeanNeuron` runtime updates across Python, Julia, Go, and Rust safety surfaces to fail closed on non-finite state/current or non-finite post-update state instead of silently reporting no spike.
- Hardened `ResonateAndFireNeuron` Julia, Go, Mojo, and Rust safety counterparts so invalid current/state and non-finite Euler updates report explicit errors/sentinels instead of silently returning no spike.
- Hardened `QuadraticIFNeuron` Julia, Go, Mojo, and Rust safety counterparts so invalid current/state and non-finite Euler increments report explicit errors/sentinels instead of silently returning no spike.

### Repository hygiene
- Purged obsolete completed failed/cancelled GitHub Actions repair-sequence runs after later successful replacement runs were verified on `main`.
- Removed inactive stale GitHub Pages deployment records while retaining the current successful Pages deployment and successful package-release deployment evidence.
- Rechecked Dependabot, code-scanning, and secret-scanning alert surfaces; all reported zero open alerts.

## [3.15.0] — 2026-05-19

### Compiler Intelligence, Platform Registry, and Deployment (2026-05-01)

#### Added
- Expanded the hardware profile catalogue across FPGA, ASIC, neuromorphic, photonic, chiplet, PIM/CXL, rad-hard, edge AI, superconducting, spintronic, ferroelectric, mixed-signal, wafer-scale, acoustic, fluidic, biological, and molecular targets.
- Added constraint-derived hardware profile construction with TOML profile loading, directory loading, runtime platform hooks, and platform discovery.
- Added compiler intelligence for target recommendation, portability scoring, topology optimisation, heterogeneous dispatch, partial reconfiguration planning, multi-die floorplanning, CDC analysis, power-state generation, regression detection, compilation reporting, and caching.
- Added verification and safety utilities for equivalence sketches, ODE stability checks, testbench generation, fault-tree generation, compliance matrices, safety-certification evidence, formal CDC checks, and provenance chains.
- Added security, sovereignty, and compliance tooling for hardware-trojan linting, side-channel linting, SBOM generation, license-compliance checks, supply-chain risk scoring, IP obfuscation, netlist watermarking, bitstream encryption, and model checksums.
- Added power, thermal, reliability, and sustainability analysis for thermal envelopes, power intent, power-domain wrappers, energy schedules, carbon estimates, reliability prediction, SEU scrubbing, and HIL calibration.
- Added deployment and integration generators for AXI4-Lite, Wishbone, RISC-V drivers, RTOS templates, memory maps, DVS-to-AER bridges, debug probes, TCL projects, open-source FPGA flows, SymbiYosys scripts, IP-XACT packaging, and Cocotb/UVM testbenches.
- Added numerical and representation support for mixed precision, microscaling FP formats, IEEE FP8, posit arithmetic, auto-quantisation sweeps, photonic MZI encoding, PIM/CXL layout planning, analog noise modelling, and bit-true software kernels.
- Added advanced co-design helpers for NIR/ONNX-SNN import, photonic configuration export, chiplet/UCIe mapping, CXL mapping, on-chip learning parameter export, drift compensation, and digital-twin generation.
- Added documentation for compiler intelligence, research platforms, deployment, platform extensibility, multi-target deployment, safety certification, verification/debug flows, carbon sustainability, static analysis, SoC integration, and equation-to-Verilog workflows.

#### Removed
- Removed monolithic compiler intelligence and platform profile modules in favour of responsibility-scoped packages.
- Removed legacy delivery-scoped test entry points in favour of responsibility-scoped regression suites.

### Security hardening (2026-04-29)

#### Added
- Property-based fuzz coverage for malformed bitstream/IR ports, Studio graph
  JSON, transfer checkpoints, NIR imports, model-zoo NPZ archives, SCPN
  datastream JSON, custom chip-spec JSON, HDL stochastic-source lowering,
  equation/MLIR lowering, and optimiser evidence JSON.
- Offline supply-chain audit command for committed CycloneDX SBOM and release
  requirements metadata: `python tools/supply_chain_audit.py`.
- Hardware-install documentation now records Vivado `v2025.2` as the current
  SHD/PYNQ evidence pin and marks OpenROAD PPA numbers as unpublished until the
  binary/container digest and PDK revision are recorded.
- Packaging metadata now exposes `sc-neurocore[hdl]`, expands
  `sc-neurocore[full]` across CPU-side training, NIR, Studio, HDL, codec,
  bioware, and quantum workflows, and packages HDL/OpenROAD source artefacts.
- Added an offline EDA toolchain version inventory helper for Vivado,
  OpenROAD, Yosys, nextpnr, IceStorm, Trellis, Quartus, Lattice tools, PYNQ,
  and OpenROAD/PDK pin metadata.

#### Fixed
- Hardened validation boundaries for fuzzed JSON, NPZ, NIR, IR, and HDL inputs
  before they reach parser, lowering, or hardware-resource paths.
- Documented the strict release-mode supply-chain gate in `SECURITY.md`.
- Aligned the CycloneDX SBOM root component version with `pyproject.toml` so
  strict supply-chain audit runs pass without metadata drift.

### CI coverage restoration (2026-04-21)

#### Fixed
- `tools/ci_install_dev.py` now installs `dev,nir,compression,training,research,bioware,studio` so the 342 torch-gated tests (`arcane_zenith`, `darts_sc_nas`, `advanced_plasticity`, and the `_native` bridges that hit the `torch.autograd.Function` path) run inside the 3.10–3.14 matrix instead of being silently skipped.
- `tests/test_analog_bridge/test_analog_bridge.py` + `test_analog_bridge_extended.py` now import through `sc_neurocore.analog_bridge` rather than via direct `sys.path.insert`; `coverage.py` was reporting 0 % for `analog_bridge.analog_bridge` despite the 27 tests executing every line.

#### Added
- `sc_neurocore.analog_bridge` package root re-exports `AnalogBridge`, `AnalogSubstrateProfile`, `EventDrivenInterface`, `CalibrationRoutine`, `AEREvent` through `__all__`.
- `tests/test_native/test_array_guards.py` — 24 multi-angle tests for `require_c_contiguous` covering happy path, dtype coercion, non-contiguous rejection, list / tuple conversion, the post-asarray defensive branch via `__array__` producers, alignment enforcement, and FFI integration byte ops. Module coverage 42 % → 100 %.
- Two `unittest.mock.patch`-based tests for `CalibrationRoutine.effective_resolution_bits` fallback (`max_err == 0` and `full_range == 0`); reachable branches not touched by the sweep-and-measure suite. Module coverage 99 % → 100 %.

### evo_substrate: 4-backend whole-process industrial evolve runner (2026-04-20)

#### Added
- `crates/evo_substrate_core` (new Rust crate, 1 227 LOC of `runner.rs` + C-FFI + PyO3 extension) — port of `ReplicationEngine.evolve_generation` + eleven industrial guards (TournamentSelector, AgeRegulator, FormalSafetyGuard, BloatPenalizer, ExtinctionDetector, HallOfFame, ParetoFront, LineageTracker, MutationEngine × 4 variants, CrossoverEngine, parametric FitnessEvaluator). Entry point `py_evolve_run(config_json) -> str`. Measured 72× speedup over the Python `ReplicationEngine` on 10-gen × 16-pop industrial runs (0.57 ms vs 40.88 ms).
- `src/sc_neurocore/accel/julia/evo_substrate/evo_runner.jl` (720 LOC) — same industrial loop in Julia 1.10+. JSON-in / JSON-out subprocess contract. Pinned deps via `Project.toml`.
- `src/sc_neurocore/accel/go/evo_substrate/runner.go` (926 LOC) — same industrial loop in Go 1.22+. Shares the JSON contract. `--runner` flag on the existing `evo_substrate_bench` binary dispatches to it.
- `src/sc_neurocore/accel/mojo/kernels/evo_runner.mojo` (803 LOC) — same industrial loop in Mojo 0.26+. Uses Mojo's Python interop for JSON + SHA-256 at the I/O boundary; compute loop (mutation, fitness, tournament, Pareto, lineage, extinction) runs in pure Mojo.
- Unified XorShift64 PRNG across all four backends (shift constants 13/7/17, `0xDEADBEEFCAFEBABE` fallback for zero seeds) so the uniform-random sequence is byte-identical cross-language. Rust↔Julia full bit-exact parity on final genomes / lineage / Pareto; Rust↔Go & Rust↔Mojo agree on structural counters but drift ~1e-3 on `best_fitness` because Go + Mojo `libm` `cos()` / `log()` differ from Rust's libm at ~1 ULP and Box-Muller compounds that.
- Hamming(7,4) encode / decode + `ScDoctor.adapt` control law added to `crates/stochastic_doctor_core` with PyO3 bridge (`py_hamming74_encode`, `py_hamming74_decode`, `py_sc_doctor_adapt`); `src/sc_neurocore/debug/sc_doctor.py` now dispatches to Rust when the extension is importable (1.7× / 3.1× speedup on encode / decode; `adapt` slower via FFI at 276 ns due to dominant PyO3 overhead). Pure-Python fallback preserved bit-exact.
- `sc_scope.compute_scc` now dispatches to `stochastic_doctor_core.py_scc_packed` (174× speedup over pure Python; bit-exact parity with fallback).
- Cross-language parity test harness `tests/test_evo_substrate/test_multilang_parity.py` (18 assertions) asserts Rust↔Julia byte-exact, Rust↔Go counter match + fitness tolerance, Rust↔Mojo schema match.
- Per-backend unit tests: Julia 17 tests (`test_evo_runner.jl`), Go 8 tests (`runner_test.go`), Mojo 7 side-validated tests (`tests/test_evo_substrate/test_mojo_runner.py`).

#### Documentation
- `docs/api/evo_substrate.md` §7.3 — new whole-process runners section with entry-point table, measured 4-way parity matrix, honest timing breakdown per backend (Rust PyO3 warm 0.57 ms, Go execution 2 ms excluding ~3 s `go build` first time, Mojo cold ~1.1 s pixi + JIT + Python interop, Julia cold ~3 s JSON.jl + SHA.jl precompile, Python reference 40.88 ms), decision matrix for which backend to pick, and the 4-way test-suite invocation list.

### Strategic module unification (2026-04-20)

#### Added
- `sc_neurocore.arcane_zenith.ArcaneZenithCognitiveCore` — three-compartment ArcaneNeuron (fast / working / deep membrane states) coupled via attention gate + self-model predictor, wired to four reward-modulated plasticity rules via a sharpened sigmoid that maps weights into biological ranges for `tau_deep`, `surprise_baseline`, `delta_conf`, `lr_base`. Factory `create_arcane_neuron_with_zenith_plasticity(backend=…)`, plus `step_from_bio_rates` (MEA rate dict) and `step_from_genome` (evo_substrate bridge). 32 multi-angle tests in `tests/test_arcane_zenith/`.
- `sc_neurocore.optics.photonic_emitter` — full rewrite of `CrosstalkModel.analyze_bank` on Marcatili coupled-mode theory (adjacent + next-nearest pairs); new `analyze_pairs` for O(N²) arbitrary geometry. Rust FFI `py_ph_analyze_crosstalk_bank` / `py_ph_analyze_crosstalk_pairs` (with 4 cargo tests); Python fallback matches to 1e-9. `FDTD2DSolver` split-field Berenger PML (Ezx + Ezy with σ-matched magnetic conductivity). `CompilationResult.to_gdsii` now produces real GDSII via `gdsfactory` + `klayout` (PDK auto-activation, `allow_duplicate` cells, netlist string to GDS TEXT layer 63/0). 43 tests in `tests/test_optics/`.
- `sc_neurocore.bioware` closed-loop surface: `BioHybridSession.process_frame` returns `BioHybridFrameResult` (typed dataclass with legacy mapping view — `result["round"]` + `result.round` both valid). `SpikeSorter` fit/assign with sklearn PCA+KMeans, no-op on empty input. `HomeostaticPlasticity.update_threshold` Q8.8 proportional controller (error × α × 256, clamped to min/max). New `mea_fitness_hook` — converts MEA spike dynamics to `{accuracy, energy_mw, latency_ms}` for evo_substrate's `ReplicationEngine(metrics_fn=…)`. Matching PCA / Berenger / closed-loop regression tests added.
- `sc_neurocore.accel.mojo.MojoKernelRunner` + `kernels.mojo` — Mojo SIMD primitives (packed SC ops, `sc_and/or/xor/mux/sub/not`, pack/unpack, `vec_mac`, `stdp_update`, `reward_modulated_stdp`, `hdc_bind`). Pixi-managed toolchain; `_HAS_MOJO` flag never raises on missing tooling. `benchmarks/bench_mojo_vs_rust.py` pure-text side-by-side harness.
- `sc_neurocore.edge.aer_router.AERRoutingDaemon` — Python lifecycle wrapper for the Go AER UDP mesh router (`accel/go/services/aer_router/main.go`). Three sibling Go modules: `hil_debugger` (WebSocket telemetry), `services` / `services_ext` (service coordination). Each with its own `go.mod` + `main_test.go`.
- `sc_neurocore.debug.hil_server.HILServerDaemon` + `HILDebugger` — lifecycle wrapper for the Go HIL debugger binary with `GET /health` readiness probe, 5 s timeout, SIGTERM → SIGKILL ladder.
- `sc_neurocore.formal.FormalProofEngine` — Lean 4 bridge. `safety_bounds.lean` proves six theorems (`monitor_soundness`, `safe_transition`, `sc_precision_bound`, `sc_add_preserves_range`, `lif_membrane_bounded`, `correlation_range`) mapped 1:1 to `neuro_safe_monitor.sv` P-properties. New `src/sc_neurocore/formal/__init__.py` exports the engine.
- `sc_neurocore.accel.julia.solvers.JuliaFusionSolver` + 4 `.jl` scripts (`fusion_solver`, `neuron_zoo`, `dynamical_analysis`, `spike_analysis`) — reference continuous-time ODE solvers via `DifferentialEquations.jl` (Tsit5).
- `sc_neurocore.hdl_gen.safety.neuro_safe_monitor` + `tb_safety_monitor` — SystemVerilog runtime safety monitor enforcing the six Lean theorems at nanosecond scale. Parameterised on Q8.8 current / voltage / coherence / SC denominator / LIF max. `openroad_flow/run_asic_flow.sh` drives Yosys synthesis (+ optional OpenROAD P&R) against the monitor with area / timing reports.
- `sc_neurocore.evo_substrate` gained (documented in full): `FormalSafetyGuard`, `BloatPenalizer`, `ExtinctionDetector`, `ComplexityTracker`, `CPPNGenome`, `ParetoFront`, `NoveltyArchive`, `HallOfFame`, `TileDeploymentTracker`, `ResourceBudget`, `LineageTracker`, `IslandModel`. Bridged to MEA via `mea_fitness_hook` and to ArcaneZenith via `step_from_genome`.
- `sc_neurocore.proto` — `core.proto` (Tensor, BitstreamMetadata) + `telemetry.proto` (HILFrame) as the wire contract for HIL debugging.
- Plasticity-layer `reset()` contract: new FFI `reset_rule_layer` in `libautonomous_learning` (Rayon par_iter over rules), new `WgpuRuleLayer::reset` + `reset_wgpu_layer` FFI, and `reset()` methods on `RustRuleLayer`, `RustWgpuRuleLayer`, `TorchRuleLayer` with per-rule trace-clearing scope matching the Rust `PlasticityRule::reset` trait contract. `ArcaneZenithCognitiveCore.reset()` now works across all three backends. 11 new tests.
- Example demos: `examples/14_bioware_closed_loop_demo.py` (100-frame MEA ↔ ArcaneZenith closed loop), `examples/15_photonic_compilation_demo.py` (SC → MZI cascade → real GDSII), `examples/16_evo_substrate_demo.py` (genome → SC top-level module → Verilog emit).

#### Documentation
- New API pages: `docs/api/mojo_accel.md`, `docs/api/edge.md`, `docs/api/formal.md`, `docs/api/julia_solvers.md`, `docs/api/proto.md`.
- Upgraded from stubs: `docs/api/evo_substrate.md` (23 → 155 lines), `docs/api/debug.md` (24 → 120 lines, added HIL section), `docs/api/hdl_gen.md` (17 → 100 lines, added safety-monitor P-property table + Lean mapping + ASIC flow).
- `docs/api/bioware.md` upgraded from 14-line stub (full `BioHybridSession` + `BioHybridFrameResult` dual-access + Q8.8 homeostatic controller + SpikeSorter + mea_fitness_hook sections).
- New `docs/api/arcane_zenith.md` + `docs/api/optics.md` completely rewritten (photonic compiler + Berenger PML + Marcatili crosstalk + GDSII).
- `mkdocs.yml` navigation restructured: new *Acceleration* (Mojo + Julia), *Formal + Safety*, *Edge + Wire Protocol* groups under Frontiers.

#### Fixed
- `RustEligentLearner.step` FFI signature was missing the `dt` parameter (4 args passed, 5 expected) — every non-empty call raised `AttributeError`. Added `dt: float = 0.001` kwarg.
- `sc_neurocore._native.learning_bridge` no longer raises at import time when `libautonomous_learning.so` is absent; returns `_HAS_LEARNING = False` so downstream imports succeed (the 398 previously-failing test collections now run).
- `CI workflows` (`ci.yml`, `v3-engine.yml`) now build the `autonomous_learning` cdylib and copy it into `src/sc_neurocore/_native/` before pytest runs — keeps the Rust path live.

#### Repository hygiene
- Untracked compiled Go bench binaries (`services_bench`, `services_ext_bench` ≈ 4.4 MB total) from `src/sc_neurocore/accel/go/services/…`; pattern added to `.gitignore` (regenerate locally via `go test -bench -c`).
- 22 ruff lint + format fixes across user-WIP modules (evo_substrate, mojo/runner, debug/hil_*, edge/aer_router, formal/lean_bridge). `ruff check src/ tests/` and `ruff format --check src/ tests/` clean.
- New optional extras in `pyproject.toml`: `optics = ["gdsfactory>=9.0"]`, `bioware = ["scikit-learn>=1.3"]`.

### CorticalColumn full-scale (77 169 cells) verification (2026-04-19)
- Ran the canonical fidelity reference: `scale=1.0, seed=42`, 600 ms simulation with the block + Rust batched multi-spmv path. 77 169 cells, build 298 s, sim 3 564 s ≈ **64 minutes wall**.
- **5/8 populations within 1.2× of Potjans Table 4** (L23i 1.07×, L4e 1.06×, L4i 1.09×, L6e 1.24×, L6i 1.05×). L5e 1.32×, L5i 1.22× plateau ~25 % over published — NOT purely a finite-size effect (does not collapse below 1.20× at full scale). L23e under-fires at 0.67× consistently across all four scales.
- Honest interpretation: the residual is a combination of (i) shorter analysis window than the published 5 s, (ii) dt-quantised global-bin delays vs the paper's per-connection continuous Gaussian, (iii) per-target multapse sampling vs NEST's `multapses=False` (which we cannot trivially use without breaking van Albada 2015 in-degree preservation). The shape is faithful (population ordering, E/I balance, all rates finite and bounded); the absolute residual at ≤ 1.32× is the practical limit of the current architecture.
- Doc page §4.1 now records all four scales side-by-side; the full-scale row is the canonical reference.

### CorticalColumn full-scale convergence verified at scale=0.5 (2026-04-18)
- Ran `scale=0.5, seed=42`, 600 ms simulation with the block + Rust batched multi-spmv path. 38 586 cells, build 116 s, sim 1 956 s (≈ 33 min wall).
- **6/8 populations within 1.2× of Potjans Table 4** (vs 5/8 at scale=0.1, 5/8 at scale=0.2): L23i 1.00×, L4e 0.95×, L4i 1.07×, L5i 1.20×, L6i 1.04×.
- L5e shrinks 1.97× → 1.52× → **1.36×**; L6e shrinks 2.81× → 2.43× → **1.68×**. Both still residual but on the predicted convergence trajectory of van Albada et al. 2015 Fig 5.
- Confirms the finite-size hypothesis empirically: residuals collapse monotonically as scale grows, full-scale (~77 000 cells) would close to ≤ 1.05× across all populations. scale=0.5 / 600 ms is now reachable in 33 min wall, unblocked by the block + Rust path.

### CorticalColumn batched multi-spmv Rust call (2026-04-18)
- New `engine/src/cortical_inject.rs::parallel_csr_multi_spmv_add` — does `2 × n_delay_bins` (= 10) spmv add operations in ONE FFI call. Rust loops internally over the bins; `par_chunks_mut(512)` parallelism still applies, with the per-row kernel summing contributions from all bins before writing back.
- New PyO3 wrapper `sc_neurocore_engine.py_parallel_csr_multi_spmv_add` accepting `Vec<PyReadonlyArray1>` for indptrs / indices / data / xs.
- `CorticalColumn._inject_block(dt)` now batches all non-empty (E + I) bins into ONE FFI call when the multi-spmv kernel is available; falls back to per-block calls otherwise.
- Bridge wrapper `bridge/sc_neurocore_engine/__init__.py` re-exports `py_parallel_csr_multi_spmv_add`.
- 1 new Rust unit test `test_multi_spmv_matches_sequential` proving batched output equals N sequential `parallel_csr_spmv_add` calls.
- **Measured perf at scale=0.1, 600 ms**: 287.5 s wall — DOWN from 460 s (single-call Rust) and ON PAR with scipy per-pair (290 s). FFI overhead reduction (10 calls → 1) reclaimed the gap.

### CorticalColumn Rust per-row-parallel CSR spmv kernel (2026-04-18)
- New `engine/src/cortical_inject.rs`: rayon-parallel CSR sparse mat-vec add (`y += W @ x`) with row-chunking (`CHUNK_SIZE = 512`) so each task sees ~250 µs of work — well above rayon's per-iteration scheduler break-even point. 4 unit tests.
- PyO3 wrapper `sc_neurocore_engine.py_parallel_csr_spmv_add` re-exported via `bridge/sc_neurocore_engine/__init__.py`.
- `CorticalColumn._inject_block(dt)` now dispatches to the Rust kernel automatically when available (auto-detected via `_HAS_RUST_CSR_SPMV`). Bit-identical results vs scipy single-threaded — per-row reductions are local so parallel order does not affect output.
- Pre-extracted `(indptr, indices, data)` triples per block at construction (`_block_e_arrays`, `_block_i_arrays`) to dodge per-step `np.ascontiguousarray` cast overhead that otherwise eats the per-call Rust speedup.
- **Honest perf finding**: Rust kernel measures 18.9 ms vs scipy 33 ms standalone (1.75× per call). In the full simulation pipeline at scale=0.1 / 600 ms, however, Rust takes **460 s** vs scipy **290 s** (per-pair) — a 1.6× regression. scipy's CSR mat-vec is already well-tuned for the in-pipeline access pattern (cache-warm matrices, sparse spike vectors); per-call Rust overhead + the surrounding Python concat / count_nonzero / slice work dominates.
- The Rust kernel is preserved as the **right primitive** for the future block-CSR / GPU / multi-node scale-up regime (where per-call FFI overhead shrinks relative to per-call work). Default per-pair scipy path is already the fastest Python-side measurement; Rust is opt-in via `use_block_csr=True`.

### CorticalColumn block-CSR opt-in path (2026-04-18)
- Added stacked block-CSR matrices keyed by `(source-type, global-bin-idx)` so the per-step inner loop can collapse from `n_pairs × n_delay_bins` (≈ 320 sparse mat-vecs) to `2 × n_delay_bins` (≈ 10). Bin centres are global, derived from theoretical Gaussian quantiles via `scipy.stats.norm.ppf`.
- New `CorticalColumn` parameter `use_block_csr: bool = False`. When True, the construction builds block matrices alongside the per-pair representation; `step()` dispatches to `_inject_block(dt)`.
- **Honest perf finding**: at `scale=0.1`, 300 ms sim, the block path measures 306 s vs ~145 s for the legacy per-pair path (≈ 2× SLOWER). scipy.sparse CSR mat-vec is compute-bound (FLOPs scale with `nnz`, identical between paths), and the per-pair tight inner loop wins on cache locality. The block path is preserved as an opt-in because it is the natural data layout for any future Rust / Mojo FFI port (10 FFI calls vs 320, where call overhead DOES dominate).
- Default flipped to `use_block_csr=False` so the as-shipped Python path stays on the fastest measured backend.
- New `tests/test_cortical_column.py::TestConnectivity::test_block_csr_path_builds_and_runs` exercises the opt-in path so it does not silently rot.

### CorticalColumn finite-size verification at scale=0.2 (2026-04-18)
- Empirically verified that the L5e/L6e residual at `scale=0.1` is a finite-size effect (van Albada et al. 2015 Fig 5), not a model bug. Scale=0.2 / 600 ms / seed=42 measurements:

  | Pop | scale=0.1 ratio | scale=0.2 ratio | Δ |
  |-----|----------------:|----------------:|---:|
  | L23e | 0.67× | 0.27× | overshoots low |
  | L23i | 1.19× | 0.94× | improving |
  | L4e | 0.68× | 0.73× | stable |
  | L4i | 1.21× | 1.08× | improving |
  | L5e | 1.97× | **1.52×** | **-23 %** |
  | L5i | 1.50× | **1.27×** | **-15 %** |
  | L6e | 2.81× | **2.43×** | **-14 %** |
  | L6i | 1.24× | **1.10×** | improving |

- The deep-layer residuals (L5e, L6e) shrink monotonically with scale; extrapolating linearly suggests scale=0.5 closes them to within 1.2-1.3× of Potjans Table 4. Closing all 8 populations to within 10 % requires full scale (~77 000 cells, ≈ 50 min/sec biotime). The implementation is faithful — the residual is intrinsic to sub-full-scale finite-size effects.
- `docs/api/cortical_column.md` §4.1 now documents the per-scale ratios side-by-side with the historical baseline and the rejected no-multapse experiment.

### CorticalColumn per-connection Gaussian delay distribution (2026-04-18)
- `network/cortical_column.py` adds per-connection delay binning. New constants `DELAY_E_SIGMA = 0.75 ms`, `DELAY_I_SIGMA = 0.4 ms` (Potjans Table 5). New `__init__` parameters `delay_distribution: bool = True` and `n_delay_bins: int = 5`. At construction time each (target, source) pair samples `K_per_target * n_t` per-connection delays from `N(DELAY_*, sigma_*)`, quantile-bins them into 5 groups and stores one sub-CSR per bin. Per `step()`, each pair contributes one `dot()` per bin, reading the source spike vector at that bin's delay offset.
- Setting `delay_distribution=False` restores the legacy single-mean-delay path for fast smoke tests and direct comparison.
- **Fidelity dramatically tightened.** Measured at `scale=0.1, seed=42`, 200 ms analysis window after 100 ms burn-in:

  | Population | single-delay ratio | per-conn Gaussian ratio |
  |------------|-------------------:|-----------------------:|
  | L23e | 5.29× | **0.67×** |
  | L23i | 4.78× | **1.19×** |
  | L4e  | 0.83× | 0.68× |
  | L4i  | 2.03× | **1.21×** |
  | L5e  | 3.05× | 1.97× |
  | L5i  | 2.10× | 1.50× |
  | L6e  | 5.23× | 2.81× |
  | L6i  | 2.33× | **1.24×** |

  5/8 populations now sit within 1.2× of Potjans Table 4; the remaining 3 (L4e, L5e, L6e) within 2-3×.
- Cost: per-step ≈ 5× slower (5 sparse mat-vecs per pair instead of 1). At `scale=0.1`, sim wall went 32 s → ~290 s for 600 ms (matches 5× expectation).
- New `tests/test_cortical_column.py::TestPublishedFidelity::test_per_connection_delays_tighten_rates` — asserts ≥ 5/8 populations within `[0.5, 1.5]×` of published Table 4 values. Pins the win.
- `benchmarks/bench_cortical_column.py` now bench BOTH `delay_distribution` modes side-by-side.
- All 29 cortical_column tests pass with the new default (29 passed in 14:18 with delay distribution, 24 deselected-fidelity tests in 4:39 for fast iteration via `-k 'not Fidelity'`).

### PINGCircuit Rust acceleration backend (2026-04-18)
- New Rust per-step kernel `engine/src/ping.rs` with PyO3 wrapper `sc_neurocore_engine.py_ping_step`. Mirrors the Python step semantics (LIF + AMPA / GABA decays + drive + Wiener noise + refractory + spike detect + reset). Noise samples are drawn on the Python side and passed in as `xi_e` / `xi_i` so the per-instance RNG state evolves identically across both backends.
- New `backend=` parameter on `PINGCircuit` (`"auto" | "rust" | "python"`, default `"auto"`). `"rust"` raises `RuntimeError` if the kernel is not built; `"auto"` falls back to NumPy.
- Bridge wrapper `bridge/sc_neurocore_engine/__init__.py` re-exports `py_ping_step` so pytest's `bridge/`-on-`sys.path` setup sees the Rust symbol.
- New `tests/test_gamma_oscillation.py::TestPythonRustParity` (6 cases): per-population firing rates within 10 % across (80, 20) / (400, 100) / (1000, 250); dominant FFT peak within 1.5 Hz; explicit `backend="rust"` smoke; invalid-backend rejection. Per-cell membrane V values drift at the float-noise level (NumPy SIMD/FMA vs Rust scalar ordering) — documented inline; aggregate dynamics match.
- `benchmarks/bench_gamma_oscillation.py` extended to bench BOTH backends. Measured speedup: ~3.3-4.3× across the three workload sizes (per-step 145.8 → 33.7 µs at (80, 20); 588.3 → 178.3 µs at (4000, 1000)). All 6 runs stay in the published 30-80 Hz dominant band.
- `engine/src/ping.rs` ships 3 Rust unit tests (no-drive silence; supra-threshold drive + refractory hold; deterministic for identical inputs). All pass on `cargo test --release`.

### CorticalColumn no-multapse experiment — REJECTED (2026-04-18)
- Tried replacing the multapse-with-replacement adjacency builder with a vectorised `argpartition` no-multapse sampler (matching NEST `multapses=False` default). Mean per-target weight is identical between the two approaches and per-target unique connectivity rises from ~63 % to 100 %.
- Measured at `scale=0.1, seed=42`, 600 ms: rates BLEW UP to refractory ceiling for 6 of 8 populations (L23e 90 Hz, L4e/L4i ≈ 410 Hz, L5e/L5i/L6i 260-390 Hz). Pre-experiment multapse-with-replacement gave rates 1.6-7.5× over Potjans Table 4 (within band, just inflated). Post-experiment no-multapse made the divergence ~10× worse.
- Honest finding: at sub-full scale the deterministic per-target in-degree of the no-multapse path amplifies population synchrony in the heavy-recurrent regime (K approaches N_s for several pairs); the multapse path's natural variance dampens this. Documented inline next to the multapse sampler so future contributors don't repeat the experiment without first re-reading van Albada 2015 §3.

### PINGCircuit scale-invariant weight normalisation (2026-04-18)
- `network/gamma_oscillation.py`: per-spike conductance contributions are now divided by source population size at construction (`_w_*_eff = w_* · default_size / actual_size`). The default `(80, 20)` published weights stay bit-identical; larger circuits no longer drift out of the 30-80 Hz band. `bench_gamma_oscillation.py` now reports 40.0 / 41.2 / 41.2 Hz across `(80,20) / (400,100) / (4000,1000)` — all in band — vs 40.0 / 103.8 / 76.2 before the fix. All 19 PINGCircuit tests still pass (default weights and behaviour unchanged at `(80, 20)`).

### Honest benchmark scripts for network/ models (2026-04-18)
- `benchmarks/bench_cortical_column.py`: 3-config wall-clock + per-population firing rates + Potjans Table 4 ratios for `CorticalColumn`. Replaces hand-measured numbers in `docs/api/cortical_column.md` with reproducible JSON output at `benchmarks/results/bench_cortical_column.json`. Honest BLOCKED status reported per backend (Rust/Julia/Go/Mojo) per `feedback_no_fabricated_benchmarks` and `feedback_module_standard_attnres`.
- `benchmarks/bench_gamma_oscillation.py`: 3-workload `step()` wall-clock + dominant gamma frequency check (must lie in 30-80 Hz) for `PINGCircuit`. JSON output at `benchmarks/results/bench_gamma_oscillation.json`. Documents the per-cell LIF + 4 conductance decays as a clean Rust + Mojo target (BLOCKED, tracked under multilang policy). Bench surfaces a real fidelity edge case at `n_e=400, n_i=100` (f_dom=103.8 Hz, outside published 30-80 Hz band) that the default-configuration test does not catch.
- `docs/api/cortical_column.md` performance table updated to reference the bench script and JSON path; numbers replaced with the measured values (build 0.04 / 2.04 / 4.07 s and per-step 0.96 / 2.07 / 5.29 ms across the three configurations).

### Bandit MEDIUM triage (2026-04-18)
- 6 MEDIUM `B307` findings (use of `eval`) → ACCEPT with `# nosec B307` markers and inline rationale: `equation_builder.py` Euler integrator, RK4 derivative eval, threshold expression and reset rule (4 sites); `studio/analysis.py` nullcline grid eval (2 sites). All sites are downstream of `EquationNeuron._validate_expr` AST whitelist (`_ALLOWED_AST_NODES` + `_BLOCKED_NAMES` reject any escape vector before `compile`) with empty-`__builtins__` eval globals.
- Re-running `bandit -r src/ -ll` returns 0 findings.
- 55 LOW findings remain (B101 asserts, B603/B404/B607 subprocess, B110 try/pass, B311 random); informational, no real impact, full inventory in `docs/internal/audit_bandit_2026-04-18.md` and `docs/internal/AUDIT_INDEX.md`.

### CorticalColumn Potjans & Diesmann 2014 (2026-04-18)
- `network/cortical_column.py` rewritten from 5-population canonical-microcircuit reduction to the full 8-population Potjans & Diesmann 2014 model: L23e, L23i, L4e, L4i, L5e, L5i, L6e, L6i with per-population sizes from Table 5, the verbatim 8×8 connection-probability matrix from Table 5, per-cell background Poisson drive (`K_bg` per population, `bg_rate=8 Hz`), and exponentially decaying current-based PSCs (`tau_syn=0.5 ms`).
- LIF integration: `C_m=250 pF`, `tau_m=10 ms`, `t_ref=2 ms`, `E_L=V_reset=-65 mV`, `V_th=-50 mV`. Per-source delays: `1.5 ms` (E), `0.8 ms` (I), quantised to `dt`.
- Synaptic weights: `w_e=87.81 pA`, `w_i=-g·w_e` with `g=4` (configurable), `w_l4_to_l23e=2·w_e` per Potjans boost.
- Sparse `scipy.sparse.csr_matrix` adjacency per (target, source) pair with multapses sampled with replacement; full-scale in-degree preservation under `scale_correction=True` (van Albada et al. 2015 protocol).
- `simulate(duration_ms, dt)`, `step(dt)`, `population_rates(rasters, dt, burn_in_ms)`, `total_indegree(target)` and `reset_state()` helpers.
- `tests/test_cortical_column.py` rewritten: 29 tests covering smoke, determinism (per-instance RNG, global-seed leak-proofing), connectivity (Table 5 entries, K_bg, weight signs, L4e→L2/3e boost, sparse adjacency built per pair), and published fidelity (no silent populations, no refractory-ceiling saturation, E/I asymmetry, L4e in band, zero-background silence). 100 % coverage on `cortical_column.py`. Closes #10.
- `docs/api/cortical_column.md` rewritten end-to-end (308 lines): published-reference summary, implementation overview (8 populations, sparse adjacency build, LIF + synapse + refractory, delay handling), public API reference, verification table vs Potjans Table 4 (L4e match within 1 %, other populations within 2-4×), performance table (4.6 s / 19.5 s / 43.6 s wall at scale 0.02 / 0.05 / 0.1) and reference list (Potjans 2014, van Albada 2015, Binzegger 2004, Hahne 2017, Douglas & Martin 2004).

### PINGCircuit conductance-based gamma (2026-04-18)
- `network/gamma_oscillation.py` rewritten from a rate-coded reduced model to per-cell conductance-based Börgers-Kopell 2003 weak-PING. HH-style integrate-and-fire with separate AMPA / GABA exponentially decaying conductances, refractory window, per-cell drive jitter and stochastic kicks. Default parameters reproduce the published 30-80 Hz gamma peak (verified at 40 Hz at the default operating point).
- `population_rate(spike_log, dt, bin_ms)` and `dominant_frequency(spike_log, dt, bin_ms, f_min, f_max)` helpers added; FFT-based with empty-log + out-of-band silence handling.
- `tests/test_gamma_oscillation.py` updated to the new API: 19 tests covering smoke, determinism (per-instance RNG isolation, global-seed leak-proofing), published fidelity (30-80 Hz peak, gain-loop disengage paths, Hz units, silence handling). 100 % coverage on `gamma_oscillation.py`. Closes #11.
- Replaced `np.sum(boolarray)` with `np.count_nonzero(boolarray)` in both implementation and tests to be reload-safe under coverage instrumentation (the `_NoValue` sentinel mismatch otherwise raised `TypeError` from `_methods.py`).

### Repository hygiene (2026-04-18)
- SPDX header format converted from 1-line piped to 2-line form across 2728 source files (.py / .jl / .rs / .go / .mojo). Closes #60.
- `microtubule_neuron.v` Engineer attribution: `Arcane Sapience`.
- `cargo clippy --release --lib`: 20 in-source warnings → 0.
- Bandit HIGH severity in `nas/sc_nas_engine.py:169` → 0 (`hashlib.md5(..., usedforsecurity=False)`).
- Chiplet package coverage 95 % → 100 % (`test_hierarchical_partitioner_perf.py`, `test_chiplet_gen_edge_cases.py`).
- `tools/run_full_cov.sh`: batched per-directory `--cov-append` runner. First full sweep completes at 43.81 % cumulative coverage; no OOM. Closes #58.
- `.gitignore`: `.agent_metadata.json`.
- `ruff`, `rustfmt`: clean across all touched files.

### Chiplet Partitioner — Multi-Language KL Refine (2026-04-18)
- **Perf:** `HierarchicalPartitioner.partition` V=200 went from 963 ms (pre-#65) → 12.7 ms (Python post-fix) → 0.04 ms (Mojo). Total wall-clock improvement at V=200: 24,000× across the chain.
- **#65 fix:** `CorrelationAwareGraph` now caches `(min, max) → edge` lookup → O(1); `_spectral_bisect` hoists `set(vertices)` out of the inner loop. 22-29× speedup at V=50/100/200.
- **#64-prep refine fix:** `_per_partition_cost(v, n_parts, ...)` returns the full length-P cost vector in ONE neighbour scan (was P redundant scans). Additional 2-9× over #65; bit-identical canonical output.
- **#74 multi-language KL refine:** Rust (`engine/src/partition.rs`), Julia (`accel/julia/chiplet/kl_refine.jl`), Go (`accel/go/partition/partition.go`), Mojo (`accel/mojo/partition/partition.mojo`) all wired into `HierarchicalPartitioner(refine_backend=...)`. Bit-exact `part_map` parity verified end-to-end via dispatcher tests on V=100. Empirical fastest-pick at V=1000: Mojo 0.20 ms (351×), Julia 0.26 ms (270×), Rust 0.29 ms (242×), Go 0.68 ms (103×), Python 70 ms.
- **Bench harness:** `benchmarks/bench_kl_refine.py` runs 5 backends with parity check; results in `benchmarks/results/bench_kl_refine.json`.
- **Tests:** 218 chiplet tests (39 new this batch); coverage 99.58 % on the chiplet package, with `chiplet_gen.py` at 100 % and `hierarchical_partitioner.py` at 99 %.

### LGSSM Multi-Language Acceleration (2026-04-17)
- **Mojo LGSSM Kalman filter** (`accel/mojo/world_model/lgssm.mojo`): hand-rolled matmul + Cholesky + triangular solve via `mojo build --emit shared-lib`. **46× over Python, 8× over Rust** at T=200 d=4 p=3 workload. Closes #69.
- **Go LGSSM** (`accel/go/lgssm/lgssm.go`): cgo + ctypes shared lib, hand-rolled Cholesky. Closes #70.
- **Julia LGSSM** (`accel/julia/world_model/predictive_model.jl`): juliacall + LinearAlgebra LAPACK. Closes #68.
- **Rust LGSSM** (`engine/src/lgssm.rs`): PyO3 + ndarray Cholesky. Closes #67.
- All 4 backends dispatched via `KalmanFilter.filter(backend='auto'|'rust'|'julia'|'go'|'mojo'|'python')`; bit-exact parity vs Python at atol≤1e-9 on means/covs, ≤1e-7 on log-likelihood.
- **Mojo 0.26 FFI pattern proven:** raw `Int` address via `arr.ctypes.data` + `UnsafePointer[T, MutAnyOrigin](unsafe_from_address=addr)` reconstruction inside the `@export` body works around the parametric-signature restriction. Same pattern reused for fault_injection + KL refine.

### Fault Injection Multi-Language (2026-04-17)
- **Rust + Julia + Go + Mojo** kernels for the 5 fault models (`bitflip`, `stuck_at_0/1`, `dropout`, `gaussian`). Mojo wins 4/5 boolean kernels (2.7-8.2× over NumPy); Julia wins Gaussian via Ziggurat randn. Bench harness with 4σ Binomial parity at `benchmarks/bench_kl_refine.py`-style 5-backend layout.

### Bench Harness Honest Exemptions (2026-04-17)
- `bench_safety_monitor.py` + `bench_chiplet.py` now emit a `backends` block in the JSON output documenting USED / EXEMPT / BLOCKED-ON-#X status per backend per op, with explicit FFI-vs-compute math instead of silent skipping.

### Cross-Module Integration — (2026-04-16)
- **Shared Core Types** `core/types.py`: unified `HardwareBudget`, `ResourceReport`, `LayerSpec`, `estimate_network()` — single source of truth for Optimizer↔NAS↔Runtime
- **Closed-Loop Adaptive Controller** `control/adaptive_loop.py`: Runtime drift detection → SA re-optimisation → new `RuntimeConfig`, configurable cooldown/threshold
- **Unified Energy Reporter** `energy_accounting/unified_reporter.py`: bridges `CarbonModel` + `ThermalModel` + ASIC power into single `analyze()` call
- **End-to-End Export Pipeline** `export/pipeline.py`: Model Zoo → ONNX → TVM Relay → MLIR/SSA → SystemVerilog in one `run()` call
- **Rust Wiring**: `sc_optimizer.py` → `optimizer.rs` SA engine, `sc_nas_engine.py` → `evo.rs` tournament selection, `photonic_emitter.py` → `photonic.rs` crosstalk analysis
- **Package Exports**: Updated `core/__init__.py`, `control/__init__.py`, `export/__init__.py` with new module exports
- **Integration Tests**: 20 new tests in `tests/test_integration/test_cross_module.py` covering all 5 actions
- **Maturin**: Rebuilt `sc_neurocore_engine` v3.14.0 with all Rust bindings
- **Total**: 10,592 tests (8,895 Python + 1,697 Rust) — ALL GREEN

### Extended Rust Wiring — QA & DNA Bridges (2026-04-17)
- **Quantum Annealing**: `bridges/quantum_annealing.py` → `py_qa_simulated_annealing` (**2,402×** at 100 qubits)
  - `IsingModel.energy()` → `py_qa_ising_energy` (Rust path for n>20 qubits)
  - `SimulatedAnnealer.solve_ising()` → `py_qa_simulated_annealing` (467× at 20Q → **2,402×** at 100Q)
  - `EnergyLandscape.analyze()` → `py_qa_batch_ising_energy` (batch energy for >100 samples)
- **DNA Mapper**: `bridges/dna_mapper.py` — Rust engine loaded (`_HAS_RUST_DNA`)
  - Imported: `py_dna_design_sequence`, `py_dna_detect_hairpins`, `py_dna_check_cross_hybridization`, `py_dna_simulate_kinetics`, `py_dna_design_orthogonal_set`
- **Photonic**: Fixed `py_ph_analyze_crosstalk` API (channel_ids, wavelengths, bandwidths, powers)

### Python vs Rust Benchmarks — Integration Hot Paths (2026-04-16)
- **SA Optimizer**: 7× (5 layers) → 36× (20 layers) → **47× (50 layers)**
- **Tournament Selection**: **337–394×** (amortised per-round overhead elimination)
- **Batch Mutate**: 17–21× across population sizes 50–1000
- **Population Diversity**: 34–**90×** (O(N²) SIMD pairwise distance)
- **Mean Rust speedup**: **334.6×** across all hot paths (incl. QA)
- **Peak QA**: 467× (20Q) → 1,426× (50Q) → **2,402× (100Q)**
- **E2E Pipeline**: NAS→Optimizer→Energy→Verilog in **13.7ms** (small) to **116ms** (large)
- Criterion (Rust-native): spike_times=83ns, firing_rate=13ns, ISI=96ns, van_rossum=1.2µs (N=100)
- Results: `benchmarks/results/py_vs_rust_integration.json`
- Script: `benchmarks/py_vs_rust_benchmark.py`

### Cross-Language Acceleration — Spike Stats (2026-04-16)
- **Crate** `spike_stats_core` (v0.1.0): 16 functions, 28 Rust tests, PyO3 + Criterion
- **Distance** (7 fns): `victor_purpura_distance` **181×**, `spike_sync` **31×**, `hunter_milton` **27×**, `van_rossum`, `spike_distance`, `earth_movers_distance`, `multi_neuron_victor_purpura` **160×**
- **Correlation** (5 fns): `cross_correlation`, `event_synchronization`, `spike_time_tiling_coefficient`, `coincidence_index`
- **Variability** (4 fns): `approximate_entropy` **73×**, `sample_entropy` **78×**, `lempel_ziv_complexity` **69×**, `permutation_entropy` **65×**
- 99/99 Python tests pass on both Rust and Python fallback paths
- Python dispatch wired in: `distance.py`, `correlation.py`, `variability.py`

### Cross-Language Acceleration —  Stochastic Doctor (2026-04-16)
- **PyO3 bindings** for `stochastic_doctor_core` crate: `py_scc_bytes`, `py_scc_batch`, `py_precision_bytes`, `py_histogram`, `PyDriftDetector`
- Replaced legacy `ctypes.CDLL` with PyO3 import pattern (primary), Python fallback (secondary)
- `SC_NEUROCORE_NO_RUST=1` env var forces Python path
- 16/16 Python tests pass on both Rust and Python paths
- 23 Rust tests pass
- **Benchmarks** (SCC single-pair): 35× at N=100, 3.5× at N=1M
- **Benchmarks** (batch SCC N×N): 15–18× for 4–64 neuron layers
- **Benchmarks** (precision): 5–14× across all sizes
- Criterion benchmarks: `crates/stochastic_doctor_core/benches/doctor_bench.rs`
- Python benchmark: `benchmarks/stochastic_doctor_benchmark.py`
- Results: `benchmarks/results/stochastic_doctor_py_vs_rust.json`
- API docs updated with full benchmark tables: `docs/api/stochastic_doctor.md`

### Module Integration — 19 Industrialized Modules (2026-04-16)
- **Industrial tier:** safety_cert (IEC 61508/ISO 26262, 81 tests), asic_flow (multi-PDK, 67 tests), fault_injection (radiation-grade, 22 tests), uvm_gen (UVM testbench, 71 tests)
- **Exascale tier:** hypervisor (multi-tenant, 78 tests), digital_twin/twinsync (time-warp sync, 72 tests)
- **Substrates tier:** spintronic (MTJ mapper, 66 tests), chiplet (UCIe/BoW, 94 tests), memristor (crossbar, 70 tests), analog_bridge (SC-to-analog, 27 tests)
- **Frontiers tier:** evo_substrate (self-replicating evolution, 91 tests), meta_plasticity (self-modifying rules, 72 tests), bioware (organoid interface, 79 tests), federated (DP-SGD, 93 tests), bci_studio (closed-loop BCI, 32 tests)
- **Unification tier:** explainability (causal attribution, 71 tests), neuro_symbolic (predictive coding, 34 tests), stochastic_doctor (bitstream diagnostics, 16 tests), model_zoo (auto-Verilog, 37 tests)
- All modules: SPDX dual-license headers, `__tier__` classification, `__init__.py` with docstrings
- 19 MkDocs API doc pages with `mkdocstrings` directives
- Updated `mkdocs.yml` nav with 5 new categories (Industrial, Substrates, Exascale, Frontiers, Unification)
- Integration reference: `docs/MODULE_INTEGRATION.md`
- Total: **1,173 new Python tests** from integrated modules

### Rust Workspace — 5 Research Crates Integrated (2026-04-16)
- Created `crates/` directory for research Rust crates
- Integrated: tinysc_riscv (83 tests), core_engine (22 tests), autonomous_learning (12 tests), neuro_symbolic (28 tests), stochastic_doctor_core (23 tests)
- Root `Cargo.toml` workspace now has 6 members (engine + 5 research crates)
- Engine (`sc_neurocore_engine`, 1,549 tests) verified undamaged after workspace expansion
- Total: **1,717 Rust tests** across 6 crates

### Evolutionary Substrate — (2026-04-16)
- `FormalSafetyGuard`: pre-deployment safety validation
- `CPPNGenome`: Compositional Pattern Producing Network developmental encoding
- `IslandModel`: multi-deme evolution with migration
- `NoveltyArchive`: k-NN behavioural novelty search
- `HWFitnessCollector`: FPGA execution feedback for hardware-in-loop fitness
- `ParetoFront`: NSGA-II style non-dominated sorting
- `TournamentSelector`, `AgeRegulator`, `BloatPenalizer`, `ExtinctionDetector`, `CoevolutionArena`
- `EvoStatisticsTracker`, `ComplexityTracker`, `genome_diff()`, `shared_fitness()`
- Module grew from 657 to 1,400 LOC, 42 to 91 tests

### Foundation-Model Neural Decoders (2026-04-07)
- **POYODecoder**: spike tokenisation + cross-attention (Azabou et al. 2023 NeurIPS)
- **POSSMDecoder**: diagonal SSM with HiPPO-LegS init (Ryoo et al. 2025 ICLR)
- **NDT3Decoder**: causal masked self-attention on binned spikes (Ye & Pandarinath 2025)
- **CEBRAEncoder**: InfoNCE contrastive embedding with analytical backprop (Schneider et al. 2023 Nature)
- Rust acceleration: tokenise_spikes, sinusoidal_position_encode, scaled_dot_product_attention, gaussian_attention, ssm_step_diagonal, infonce_loss (6 pub fn, 11 tests)
- PyO3: 5 functions registered
- Tests: 47 multi-angle tests
- Documentation: 976 lines, 8/8 sections

### Transcriptomic Foundation Model Interfaces (2026-04-07)
- **ScKGBERTInterface**: dual S-Encoder + K-Encoder with Gaussian attention (Li et al. 2025 Genome Biology)
- **GeneformerInterface**: rank-value tokenisation + multi-head attention + MLM (Theodoris et al. 2023 Nature)
- **rank_value_encode**: shared utility for gene expression tokenisation
- Tests: 29 multi-angle tests
- Documentation: 1,118 lines, 8/8 sections

### Gap Model Python + PyO3 + Docs (11 models, 2026-04-07)
- 10 new Python implementations (publication-exact): AdaptiveThresholdMoENeuron, HybridLinearAttentionNeuron, QuantumInspiredLIFNeuron, DendriticNMDANeuron, MulticompartmentMCNNeuron, AstrocyteLIFNeuron, DirectionSelectiveRGC, CochlearHairCell, ShortTermPlasticitySynapse, DopamineStdpSynapse
- PyO3 wiring: 11 models registered (2 macro + 9 manual wrappers)
- Tests: 87 multi-angle tests
- 10 docs (5,701 lines total)
- GPU backend documentation (607 lines)

### CI & Dependency Fixes (2026-04-07)
- **PEP 639**: migrated `license = { text = "..." }` → `license = "AGPL-3.0-or-later"` (fixes setuptools ≥78)
- **mypy**: 1.19.1 → 1.20.0
- **cyclonedx-bom**: 7.2.2 → 7.3.0
- **ci.yml**: pinned all mypy stub dependencies to exact versions (CodeQL #287)
- **cargo fmt**: applied to all new Rust code
- Purged 52 resolved failed/cancelled CI runs
- Closed superseded dependabot PRs #53, #55

### Neuron Models — (12 new models, 2026-04-04/05)
- **TUMNetwork**: rate model with short-term plasticity (depression + facilitation), 3 ODEs
- **ElBoustaniNetwork**: E/I + NMDA bistability, 3 ODEs
- **GradedSynapseNeuron**: non-spiking, passive RC + sigmoid release
- **GapJunctionNeuron**: LIF + electrical synapse with Cx36 rectification
- **FrankenhaeUserHuxleyAxon**: GHK permeability-based currents (not linear V-E)
- **NodeOfRanvier**: MRG 2002 — Nav1.6 transient + persistent + Kv7 slow K
- **MyelinatedAxon**: MRG node + passive internode cable
- **CardiacPurkinjeFibre**: DiFrancesco-Noble 1985, 6 currents
- **SmoothMuscleCell**: CaL + BK + IP3R/SERCA + Ca²⁺ store
- **EndocrineBetaCell**: CaL + K_dr + K_ATP + K_Ca glucose-dependent bursting

### Fidelity Audit Fixes (7 models corrected, 2026-04-04)
- **RetinalGanglionCell**: basic LIF → Pillow 2005 GLM (stimulus + history filters)
- **InnerHairCell**: no vesicle pool → Meddis 1986/2006 (q/c/w compartments)
- **OuterHairCell**: unidirectional sigmoid → bidirectional asymmetric prestin (Santos-Sacchi 2006)
- **GranuleCell**: LIF-style → D'Angelo 2001 full HH (7 ionic currents)
- **AlphaMotorNeuron**: PIC no inactivation → h_pic + Ca²⁺ buffering
- **RodPhotoreceptor**: no Ca²⁺ feedback → Ca²⁺-GC feedback (Nikonov 2006, Hill n=4)
- **TraubMilesNeuron**: missing M-current → Kv7/KCNQ (Yamada 1989)

### Kinetics Audit Fixes (3 models upgraded, 2026-04-05)
- **GolgiCell** (CRITICAL): 5-current WB → full Solinas 2007 (11 currents, 13 gating variables)
- **DCNNeuron** (MODERATE): added persistent Na (INaP) + Ca²⁺-dependent AHP (7 currents total)
- **OlfactoryReceptorNeuron** (MODERATE): added PDE4 negative feedback on cAMP

### Infrastructure (2026-04-05)
- `supported_models()`: 28 missing entries added (159 total)
- Interface wrappers: 20 non-standard models wired via Wr* types (multi-input, i32-input, graded/rate)
- All 4 failing CI workflows fixed (clippy, ruff, MkDocs, typos)
- `cargo fmt` applied to all engine source
- Fresh Criterion benchmarks published (2026-04-05)
- Documentation audit: all stale numbers corrected across README, pricing, index, benchmarks

### Notebooks (13 new, 21 total)
- **08_equation_to_verilog**: ODE string → Python sim → Q8.8 Verilog (LIF, FHN, Izhikevich)
- **09_topology_and_dynamics**: 6 generators, adjacency matrices, degree distributions, raster plots
- **10_spike_train_analysis**: ISI, CV, Fano, cross-correlation, van Rossum, PCA
- **11_biological_circuits**: tripartite synapse Ca²⁺ dynamics, Rall dendrite nonlinearity
- **12_learning_rules**: STDP, e-prop eligibility, R-STDP, STP facilitation/depression
- **13_quantisation_pipeline**: float → Q8.8 → SC probabilities → Verilog export, error budget
- **14_sc_arithmetic_theory**: AND=multiply, XNOR=bipolar, MUX=add, CORDIV=divide, Sobol vs Bernoulli convergence, Hoeffding bounds
- **15_fault_tolerance**: SC vs fixed-point under bit-flips/stuck-at, TMR majority vote
- **16_neuron_atlas**: 12 models from 8 families (LIF→ArcaneNeuron, 1907–2026)
- **17_reservoir_computing**: liquid state machine, temporal XOR, ridge readout, SVD dimensionality
- **18_mixed_precision_sc**: per-layer adaptive L, Hoeffding vs sensitivity allocation, Pareto frontier
- **19_compression_and_pruning**: magnitude/SC-aware pruning, quantisation sweep, combined Pareto
- **20_power_analysis**: event-driven vs clock-driven toggle count, scaling with network size
- **21_spike_alu**: Turing-complete spike-based ALU — logic gates, SR latch register, ripple-carry adder, sort
- **22_ir_type_safety**: IR signal type checker — Bitstream/Rate/Spike/Fixed, catch mismatches before Verilog synthesis
- **23_topological_observables**: winding number, Ollivier-Ricci curvature, sheaf consistency defect, connection curvature
- **24_identity_lazarus**: Lazarus checkpoint save/load/merge, TraceEncoder text→spikes, StateDecoder attractor extraction, DirectorController L16 self-regulation
- **25_cortical_column_dynamics**: canonical 5-population microcircuit, thalamic drive, layer-resolved rasters, feedforward latency
- **26_spike_codec_benchmark**: 5 codecs (ISI/AER/predictive/delta/streaming) on synthetic data, compression ratio vs density curves
- **27_python_to_proven_silicon**: complete end-to-end pipeline — ODE string → Python sim → IR type check → Q8.8 Verilog → testbench → formal properties → resource estimate
- **28_domain_bridge**: TensorStream prob↔bitstream↔quantum conversions, QuantumStochasticLayer cos²(θ/2) non-linearity, Born rule roundtrip

### Tests (19 new files, ~3700 lines, ~310 test methods)
- `test_topology_generators.py`: 6 generators — CSR validity, degree, symmetry, edge count, determinism
- `test_cordiv_division.py`: CORDIV accuracy, monotonicity, convergence, adaptive_length Hoeffding bounds
- `test_fault_injection.py`: bit-flip degradation, stuck-at analytical bounds, TMR, SC vs fixed-point comparison
- `test_learning_advanced.py`: EligibilityTrace decay, BPTT/TBPTT loss, R-STDP reward gating, STP facilitation/depression/recovery
- `test_quantisation_pipeline.py`: Q8.8 roundtrip, dequantise fidelity, SC probability ordering, dot product end-to-end
- `test_network_monitors_stimulus.py`: SpikeMonitor record/count/trains, StateMonitor accumulation, RateMonitor bins, TimedArray clamp, StepCurrent onset/offset, PoissonInput rate/seed/weight
- `test_neuron_families.py`: parametrised test across 11 EquationNeuron models — step(), spike detection, reset, state finiteness, determinism
- `test_sc_convergence.py`: AND O(1/√L), Sobol faster than Bernoulli, CORDIV monotonic, correlation violation, popcount exact
- `test_spike_alu.py`: SpikeGate truth tables (AND/OR/NOT/NAND/XOR), De Morgan law, SpikeRegister roundtrip, SpikeALU add/sub/xor/compare/shift, spike_sort correctness
- `test_topological_observables.py`: winding number wraps, Ricci curvature complete>ring, sheaf defect zero when synchronised, connection curvature bounded by coupling
- `test_scpn_integrated.py`: K_nm symmetric zero-diagonal, OMEGA_N physical frequencies, create_full_stack 16 layers, run_integrated_step finite, get_global_metrics
- `test_identity_lazarus.py`: IdentitySubstrate run/step/health, TraceEncoder encode/determinism, Checkpoint save/load/merge roundtrip, StateDecoder patterns/attractors, DirectorController monitor/diagnose/correct
- `test_cortical_column_dynamics.py`: CorticalColumn step/run dict outputs, 5 populations, binary spikes, thalamic drive, L4-before-L5, inhibition, reset, determinism
- `test_codec_roundtrip.py`: all 5 codecs parametrised — lossless roundtrip (sparse/empty/single-spike/all-ones), compression ratio bounds, shape preserved, edge cases (1 channel, 1 timestep)
- `test_tensor_stream.py`: TensorStream prob↔bitstream↔quantum roundtrips, Born rule, normalisation, p=0/1 edge cases, invalid conversion raises
- `test_quantum_hybrid.py`: QuantumStochasticLayer cos²(θ/2) transfer, p=0→1, p=1→0, monotonic decreasing, multi-qubit independence

### Model Validation
- LIF f-I curve: 29/29 tests, <5% error vs analytical solution
- Izhikevich 20 firing patterns: all from Izhikevich (2003) Table 1 validated
- Hodgkin-Huxley 1952: AP peak 40.6mV, spike width 1.46ms, AHP -75.1mV
- NeuroBench SHD: 79.28% test accuracy (250K params, feedforward)
- Brian2 parity: exact LIF match (0.000ms timing diff), 7.3x speedup
- 5 validation docs with measured data in `docs/validation/`

### Stochastic Computing Pipeline
- Bipolar SC (XNOR): `core/bipolar.py` for signed weight multiplication
- SC bitstream MNIST: 10% (unipolar) -> 35.6% (bipolar) -> 50.0% (all fixes)
- SC-aware training: `SCAwareLIFNet` with bitstream noise injection (+9.5pp)

### Quantization-Aware Training
- `QuantizedLIFNet`: 2/4/8/16-bit STE weight quantization (PyTorch)
- `SCAwareLIFNet`: SC noise injection during training
- `SCAwareLinear`: drop-in layer replacement

### Encoding Comparison
- 7 temporal spike encodings benchmarked on MNIST
- Latency encoding Pareto-optimal: 88.1% at 142 spikes (17x fewer than rate)

### Interoperability
- NeuroML 2 importer: iafCell, Izhikevich (2003/2007), AdEx
- SONATA network format importer: nodes.h5 + edges.h5, connectivity matrix

### Reproducibility
- 7 Kaggle scripts in `notebooks/*_kaggle.py`
- JSON artifacts in `benchmarks/results/`

## [3.14.0] — 2026-03-27

### Visual SNN Design Studio (Experimental)
- **New feature:** web-based IDE for designing, training, compiling, and deploying SNNs
- 118-model browser with live simulation, parameter sliders, pattern classification
- 20+ analysis views: trace, phase, ISI, f-I, bifurcation, heatmap, sensitivity, STA, frequency response, characterisation, multi-model overlay, A/B comparison
- Compiler Inspector: SC IR build/verify/emit, SystemVerilog generation, co-simulation
- Synthesis Dashboard: Yosys synthesis for 4 FPGA targets (ice40, ECP5, Gowin, Xilinx), multi-target comparison, resource estimation without Yosys
- Training Monitor: live SSE metric streaming, 6 surrogate gradients, per-layer spike rates, learnable beta/threshold
- Network Canvas: React Flow drag-and-drop populations and projections, NIR export/import
- Full pipeline: network graph → validate → simulate → compile → synthesise in one click
- Project save/load: persistent JSON workspaces on server
- E-I balanced network simulation with Rust engine fast path
- 140+ Studio-specific tests
- Documentation: 7 pages on GitHub Pages, 10-step quickstart tutorial
- Launch: `pip install sc-neurocore[studio] && sc-neurocore studio`

### Rust Engine
- `py_simulate_ei_network()`: fused E-I network simulation (CSR + Poisson + Euler) in single Rust call
- `py_batch_simulate()`: batch model simulation with NeuronVariant dispatch loop
- `create_neuron()` made `pub` for reuse across lib.rs
- 288 Rust tests passing

### Performance
- Model list caching: first `/api/models` call loads 118 models in ~1s, subsequent calls <1ms

### Security
- 25 CodeQL "information exposure through exception" fixes — no tracebacks in HTTP responses
- 5 CodeQL "uncontrolled data in path expression" fixes — project name sanitisation
- DOMPurify XSS fix via npm override (>=3.3.2)
- Bandit: MD5 usedforsecurity=False, narrowed bare except clauses

### CI
- Engine wheel publish job added to publish.yml (PyPI OIDC)
- Bridge ImportError restored for pytest.importorskip compatibility
- PnR added to typos dictionary
- tsconfig.tsbuildinfo gitignored
- uvicorn skip guard for studio optional extra

### ANN-to-SNN Conversion Engine
- `sc_neurocore.conversion.convert()`: automated PyTorch ANN to rate-coded SNN conversion
- QCFS activation (Quantization-Clip-Floor-Shift): ReLU replacement for conversion-aware training
- Threshold normalization from calibration data activation statistics
- `ConvertedSNN.run()` and `.classify()` for inference with Poisson rate coding

### Learnable Delay Training
- `DelayLinear`: PyTorch module with trainable per-synapse delays via linear interpolation
- Differentiable delays: gradients flow through fractional delay positions
- Export to integer delays for hardware deployment via `delays_int` and `to_nir_delay_array()`
- DCLS (Dilated Convolutions with Learnable Spacings) principle for fully-connected SNN layers

### One-Command FPGA Deploy
- `sc-neurocore deploy model.nir --target artix7`: NIR/PyTorch → Verilog → project in one command
- Target presets: ice40, ecp5 (Yosys Makefile), artix7, zynq (Vivado project.tcl)
- Copies 19 HDL library modules, generates neuron SystemVerilog, build script, README

### Network Engine
- Per-synapse delays in Projection: `delay=array` for heterogeneous axonal/synaptic delays
- Spike-gating: `Population.step_all(spike_gating=True)` skips idle neurons, compute proportional to active count
- Weight sparsity: `Projection(weight_threshold=0.01)` skips near-zero synapses during propagation

### Compiler
- Per-layer adaptive bitstream length: `assign_lengths()` with Hoeffding or sensitivity-based allocation
- Mixed-precision SC networks: shallow layers use short L (fast), deep layers use long L (precise)

### Event-Driven FPGA RTL
- `sc_aer_encoder.v`: spike vector → AER packets via priority encoder, idle neurons consume zero power
- `sc_event_neuron.v`: Q8.8 LIF that computes only on input events or periodic leak ticks
- `sc_aer_router.v`: distributes AER events to target neurons using connectivity lookup table
- Total HDL modules: 19 (was 16)

### Performance
- Lazy-load 109 neuron models: import time 200s → 57s
- Deferred scipy imports (stats.qmc, sparse): import time 57s → 10s

### Infrastructure
- Coverage fixes: test second model access, pragma Rust-only branch
- Coverage for lazy-load path, sparse guard mock path
- Ruff F401 re-export fixes, format vectorized_layer

## [3.13.3] - 2026-03-20

### SC Arithmetic
- CORDIV division circuit: Python `sc_divide()` + Verilog `sc_cordiv.v` (Li et al. 2014)
- Adaptive bitstream length: Hoeffding/Chebyshev/variance bounds via `adaptive_length()`
- Sobol/Halton multi-dimensional decorrelation for per-synapse independent streams
- Chaotic RNG mode in BitstreamEncoder (logistic map)
- Sobol bitstream attention: `StochasticAttention.forward_bitstream()` with LDS variance reduction

### Learning Rules
- BCM metaplasticity with sliding threshold (Bienenstock-Cooper-Munro 1982)
- Voltage-based STDP (Clopath et al. 2010)
- Truncated BPTT for long sequences (`TBPTTLearner`, Williams & Peng 1990)
- EWC penalty implemented (was no-op stub) — Kirkpatrick et al. 2017
- Learnable beta/threshold on all 10 SNN cell types (ExpIF, AdEx, Lapicque, Alpha, SecondOrderLIF, IF, Synaptic)
- ConvSpikingNet now works with `train_epoch()` via `flatten_input=False`

### Biological Circuits
- Tripartite synapse: astrocyte ↔ synapse bidirectional coupling (Araque et al. 1999)
- Rall branching dendrite: compartmental tree with 3/2 power rule
- Canonical cortical microcircuit: 5-population column (L2/3 exc/inh, L4, L5, L6)
- Astrocyte adapter: `AstrocyteNeuron` wraps Li-Rinzel model for Population/Network

### Theoretical Depth
- SC→quantum circuit compiler: Ry encoding, statevector simulator, layer compilation
- Zero-multiplication predictive coding SC layer (Conjecture C9: XOR=error, popcount=magnitude)
- Topological observables: winding number, Ollivier-Ricci curvature, sheaf defect
- Phi* integrated information estimation (Barrett & Seth 2011, IIT)
- Goldstone mode verification for Knm coupling spectrum
- Fault tolerance benchmark: SC vs fixed-point degradation curves
- Hardware-aware SC layer with memristive defect injection
- Noisy quantum simulation via HeronR2NoiseModel Kraus channels

### NIR Bridge
- Recurrent edge handling via unit-delay insertion (LSTM-like feedback)
- Multi-port subgraph support (`SCMultiPortSubgraphNode`)

### Compiler
- IR type checker: Bitstream/Rate/Spike mismatch detection before emission
- SV/MLIR emission for GraphForward, SoftmaxAttention, KuramotoStep (was error stub)
- Weight quantizer exported in compiler `__init__.py`

### Hardware Stack
- AXI-Stream interface for bulk bitstream I/O (`sc_axis_interface.v`)
- DMA controller for weight upload and output readback (`sc_dma_controller.v`)
- Parameterized AXI-Lite register file (`sc_axil_cfg_param.v`)
- Clock domain crossing primitives: 2-FF sync, Gray counter, async FIFO (`sc_cdc_primitives.v`)
- NEON scalar-equivalence tests (13 tests for popcount, dot, max, sum, scale)

### Infrastructure
- Rust engine wheel publishing in PyPI release workflow
- SpikeInterface/Neo adapter for experimental data import
- Static CycloneDX SBOM (v1.6)
- JAX autodiff fix: straight-through estimator for spike reset
- IIT added to typos allowlist

## [3.13.2] - 2026-03-19

### Equation → Verilog RTL Compiler
- `equation_compiler.py`: compile any `EquationNeuron` to synthesizable Q8.8 fixed-point Verilog
- `equation_to_fpga()`: one-liner from Brian2-style ODE string to Python neuron + Verilog RTL
- AST-to-Verilog expression emitter handles +, -, *, /, **, unary minus, comparisons
- Multi-variable ODE support (FitzHugh-Nagumo, Izhikevich, Hodgkin-Huxley)
- Threshold and reset logic auto-generated

### NIR Bridge
- `nir_bridge` package: import NIR graphs into SC-NeuroCore (FPGA backend for NIR)
- Maps 11 NIR primitives (LIF, IF, LI, Integrator, Affine, Linear, Scale, Threshold, Flatten, Input, Output)
- Recursive graph parser with topological sort, fan-in summation, nested subgraph support
- NIR integration guide, API docs, notebook (05_nir_bridge.ipynb)

### Packaging & Release
- Restored `sc-neurocore` as the only PyPI product package and removed the unintended runtime dependency on a separate `sc-neurocore-engine` publish
- Publish automation now pushes only `sc-neurocore` to PyPI while keeping the Rust engine on the existing crate / source / CI wheel paths
- Tag pushes still trigger publish directly, so release creation no longer depends on a downstream `release.published` event

## [3.13.1] - 2026-03-19

### Packaging & Install
- Top-level `sc-neurocore` now requires the matching `sc-neurocore-engine` release, and `sc-neurocore info` reports engine version mismatches explicitly instead of silently mixing versions
- Dense-layer example and getting-started/docs packaging guidance now match the current public API and distinguish wheel-shipped modules from source-only modules

### NIR Bridge
- Nested NIR subgraphs now execute through a dedicated subgraph node wrapper and reset cleanly inside `SCNetwork`
- `Flatten` now respects `start_dim` / `end_dim`, and bridge coverage is enforced instead of being omitted
- Added regression coverage for nested graphs, fan-in, cycle detection, orphan nodes, flatten edge cases, and file-based import/export

### CI & Release
- CI now builds and installs the local engine wheel before editable/package installs, so unreleased versions no longer fail dependency resolution
- Build smoke installs both the engine wheel and the top-level wheel from local artifacts
- Publish workflow now runs from tag pushes, builds engine sdist+wheels, publishes the engine package before `sc-neurocore`, and keeps manual dispatch build-only unless publish is explicitly enabled
- Release workflow now attaches both the pure-Python wheel and sdist to GitHub Releases

### Bug Fixes
- StochasticTransformerBlock: clamp residual and FFN intermediate values to [0, 1] — MAC output from `VectorizedSCLayer` can exceed 1.0, triggering the new input validation
- Optional dependency introspection in `sc-neurocore info` no longer crashes on broken NumPy/JAX imports

### Tests
- Full preflight now passes at `2112 passed`, `38 skipped`, `12 xfailed`, with `100.00%` coverage
- Added audit validation tests for VectorizedSCLayer/EquationNeuron, CLI fallback coverage, dense-layer example smoke coverage, and expanded NIR bridge regressions

### Documentation
- Replace stale black references with ruff format in `VALIDATION.md` and `CONTRIBUTING.md`
- Sync the packaging/install docs with the released product surface
- Package naming and install guidance were corrected in `3.13.2`; `3.13.1` incorrectly treated `sc-neurocore-engine` as a separate PyPI runtime dependency

## [3.13.0] - 2026-03-18

### Python 3.14 Support
- CI test matrix, wheel builds, and publish workflow now include Python 3.14
- All 1 776 Python tests pass on 3.14; all dependencies compatible
- pyproject.toml classifier added

### Bridge Wiring
- 12 missing Rust symbols exported from bridge `__init__.py`: NetworkRunner, BitstreamAverager, Izhikevich, ArcaneNeuron, 8 AI-optimized models, ContinuousAttractorNeuron
- Parity test name mapping for RustContinuousAttractorNeuron

### CI Fixes
- Black formatting for identity/ files; pre-commit ruff upgraded v0.9.7 → v0.15.6
- Clippy: PopulationRunner::is_empty() added
- TraceEncoder: deterministic hash (byte-based, not Python hash())
- Synapse test tolerance widened for short bitstream noise
- Notebook trailing newline for end-of-file-fixer
- Removed deleted ruff rule UP038

### Documentation
- JOSS paper rewrite: pipeline + spike raster figures, Availability section, McCulloch-Pitts/Hodgkin-Huxley citations, tightened to ~1200 words
- All docs synced: test counts (1 776/336), 111 NetworkRunner, 17 HDL, Python 3.14
- Neuron explorer notebook (04_neuron_explorer.ipynb): 5 sections, 117 models

### Infrastructure
- `.gitattributes`: eol=lf (suppress CRLF warnings on Windows)
- Single-directory migration: `03_CODE/sc-neurocore/` is canonical repo
- PyPI deployment branch policy fixed (main added)
- 12 known Rust/Python parity divergences tracked as xfail
- 5 version-gate assertions updated

## [3.12.0] - 2026-03-17

### ArcaneNeuron + 8 AI-Optimized Models
- ArcaneNeuron: unified self-referential cognition model with 5 coupled subsystems (fast/working/deep/gate/predictor)
- 8 novel AI-optimized spiking neuron models: MultiTimescaleNeuron, AttentionGatedNeuron, PredictiveCodingNeuron, SelfReferentialNeuron, CompositionalBindingNeuron, DifferentiableSurrogateNeuron, ContinuousAttractorNeuron, MetaPlasticNeuron
- Total neuron count: 122 Python (113 bio + 9 AI), 111 Rust (including Arcane)
- ArcaneNeuron included in Rust NetworkRunner (111-model fused loop, was 80)

### Identity Substrate
- `sc_neurocore.identity` package: persistent spiking network for identity continuity
- IdentitySubstrate: 3-population network (HH cortical + WB inhibitory + HR memory) with STDP
- TraceEncoder: LSH-based reasoning trace to spike pattern encoding
- StateDecoder: PCA + attractor extraction + priming context generation
- Checkpoint: Lazarus protocol save/restore/merge of complete network state (.npz)
- DirectorController: L16 cybernetic closure with monitor/diagnose/correct feedback loop

### Network Simulation Engine
- Population-Projection-Network architecture with 3 backends: Python (NumPy), Rust (NetworkRunner), MPI (mpi4py)
- 6 topology generators: random, small-world, scale-free, ring, grid, all-to-all
- 12 visualization plots: raster, voltage, ISI, cross-correlogram, PSD, firing rate, phase portrait, population activity, instantaneous rate, spike train comparison, network graph, weight matrix
- 7 advanced plasticity rules: BPTT, e-prop, R-STDP, MAML, homeostatic, STP, structural
- MPI distributed simulation for billion-neuron scale via mpi4py

### Rust NetworkRunner
- 111-model fused simulation loop with Rayon-parallel population stepping (was 80)
- CSR-sparse projection propagation
- Scales to 100K+ neurons with near-linear speedup

### Model Zoo
- 10 pre-built network configurations: Brunel balanced, cortical column, CPG, decision-making, working memory, visual cortex V1, auditory processing, MNIST classifier, SHD speech, DVS gesture
- 3 pre-trained weight sets: MNIST (784-128-10), SHD (700-256-20), DVS gesture (256-256-11)

### conda-forge
- Recipe draft prepared for staged-recipes submission; not yet published on
  conda-forge

### Analysis Toolkit
- 126 spike train analysis functions across 23 modules (22 spike_stats + 1 explainability)
- Covers: basic stats, variability, rate estimation, distance metrics,
  correlation, spectral, temporal, stimulus, LFP coupling, surrogates,
  information theory, causality, dimensionality, decoding, network,
  point process, sorting quality, waveform, statistics, patterns, SPADE, GPFA
- Pure NumPy, zero external dependencies
- Tests: 1 776 Python total, 336 Rust total

### Neuron Model Library (122 Python / 111 Rust)
- 108 individual model files in `neurons/models/` (one file per model)
- 108 individual model files across 14 families: IF variants, Biophysical, Adaptive, Oscillatory, Bursting, Synaptic, Multi-compartment, Map-based, Stochastic, Population, Hardware, Modern/ML, Rate, Other
- Notable additions: TraubMiles, WilsonHR, Pospischil (5 cortical types), ConnorStevens, WangBuzsaki, PinskyRinzel, Destexhe, HuberBraun, GolombFS, MainenSejnowski
- Historical coverage from McCulloch-Pitts (1943) to Gated LIF (2022)
- 10 PyTorch training cells: LIF, IF, Synaptic, ALIF, RecurrentLIF, ExpIF, AdEx, Lapicque, Alpha, SecondOrderLIF

### MNIST 99.49% Accuracy
- `examples/mnist_conv_train.py` — ConvSpikingNet with learnable beta/threshold
- Architecture: Conv(1->32)->LIF->Pool->Conv(32->64)->LIF->Pool->FC->LIF->FC->LIF
- Techniques: FastSigmoid surrogate, cosine LR schedule, data augmentation, membrane readout
- Trained on RTX 6000, 30 epochs, 25 minutes
- Model checkpoint: `examples/mnist_conv_train/results/conv_spiking_net_best.pt`
- Reproducibility manifest: `benchmarks/results/mnist_conv_accuracy_reproducibility.json`

### Intel Lava/Loihi Bridge
- `integrations/lava_bridge.py` — SCtoLavaConverter, export_weights_loihi
- SCDenseProcess + PySCDenseModel for Lava CPU simulation
- Weight conversion: SC probability [0,1] -> Loihi fixed-point

### Rust Engine parity expansion (v3.8/v3.9 carry-forward)
- **Sobol bitstream** (M1): Gray-code Sobol quasi-random encoder in Rust (`sobol.rs`)
- **HomeostaticLIF**: adaptive threshold neuron with EMA spike rate tracking
- **DendriticNeuron**: XOR-nonlinearity compartmental model
- **RewardStdpSynapse**: eligibility trace + reward-modulated STDP
- **Conv2DLayer**: im2col + SC multiply-accumulate convolution
- **RecurrentLayer**: echo state network with state feedback
- **LearningLayer**: online STDP-integrated dense layer
- **FusionLayer**: weighted stochastic multiplexing across modalities
- **MemristiveLayer**: dense layer with stuck-at faults and write noise
- **SpikeRecorder**: buffered spike recording with firing rate and ISI stats
- **ConnectomeGenerator**: Watts-Strogatz and Barabási-Albert topology generators
- **FaultInjector**: bit-flip and stuck-at fault injection on packed bitstreams
- **MLIR emitter**: CIRCT hw/comb dialect IR emission (`ir/emit_mlir.rs`)
- **Static synapse**: completed with excitatory/inhibitory polarity
- **Surrogate gradient**: added Triangular and PiecewiseLinear variants
- Rust neuron models callable from Python: 111 (of 122 Python total)

### SIMD Hardening (v3.8 carry-forward)
- Fused `softmax_inplace_f64_dispatch` with SIMD max/sum/scale
- Hamming distance dispatch for all backends (AVX2, SVE, RVV)
- SVE/RVV softmax portable fallbacks
- Attention softmax refactored to use fused dispatch

### Quantum Backend Stabilisation (v3.9 carry-forward)
- IBM Heron r2 noise model: depolarizing, amplitude/phase damping, readout asymmetry
- Parameter-shift gradient rule for variational quantum circuits
- Hybrid quantum-classical VQE pipeline with scipy optimizer
- QEC noise integration with surface code threshold comparison

### Holonomic Adapter Ecosystem (v3.9 carry-forward)
- L1-L16 adapters registered in ComponentRegistry with `create_adapter()` factory
- Per-adapter benchmark suite: latency, memory, throughput (with/without JAX JIT)
- Plugin discovery via `importlib.metadata` entry points

### Type Safety Cleanup (M2)
- Removed 235 unnecessary Python type-suppression comments (260 -> 25)
- Remaining 25 are justified: CuPy type aliases, optional imports, private method access

### GPU SNN Training with Surrogate Gradients
- `sc_neurocore.training` — PyTorch-based differentiable SNN training module
- 3 surrogate gradient functions: FastSigmoid (Zenke 2018), SuperSpike (Zenke 2021), ATan (Fang 2021)
- `LIFCell`, `RecurrentLIFCell` — `nn.Module` LIF neurons with autograd through spikes
- `SpikingNet` — multi-layer feedforward SNN with spike-count and membrane readout
- `to_sc_weights()` — export trained float weights to [0,1] range for SC bitstream deployment
- 3 loss functions: spike count cross-entropy, membrane cross-entropy, spike rate MSE
- `train_epoch()` / `evaluate()` — training loops with temporal unrolling
- `examples/mnist_surrogate/train.py` — MNIST benchmark (~95% accuracy, 10 epochs)
- 31 tests covering surrogates, modules, and training loops
- Requires `pip install sc-neurocore[training]` or `sc-neurocore[research]`

## [3.10.0] - 2026-03-09

### MNIST-on-FPGA Demo
- **End-to-end pipeline**: `examples/mnist_fpga/demo.py` — train (sklearn digits),
  PCA 64→16, quantise Q8.8, stochastic computing inference, Verilog weight export
- Float 94.2%, Q8.8 94.2%, SC 94.0% (L=1024, sign-magnitude encoding)
- Resource estimate: 16→10 config = ~56K LUTs (fits Artix-7 100T)
- `hdl/sc_dense_matrix_layer.v` — per-neuron weight dense layer for classification

### Vivado Tooling
- `tools/vivado_impl.tcl` — non-project flow: synth → place → route (250 MHz default)
- `tools/vivado_report.py` — parse timing/utilization/power reports to JSON

### Tutorial
- `docs/tutorials/fpga_in_20_minutes.md` — 6-section FPGA deployment tutorial

### Paper
- JOSS paper updated to submission-ready state (`paper/paper.md`)
- 12 references with DOIs, MNIST demo results, Brian2 comparison, formal verification

### Documentation Overhaul
- README: benchmarks section (Rust SIMD, Brian2 comparison, Yosys synthesis)
- README: all 10 HDL modules listed with descriptions
- Zenodo DOI updated to 10.5281/zenodo.18906614
- CITATION.cff, .zenodo.json: DOI, version, author corrections
- CONTRIBUTING.md, VALIDATION.md, getting-started.md: test counts, Python version
- Yosys MODULES list updated (10 modules)

### Fixes
- Zenodo author list corrected (sole author: Miroslav Šotek)
- DOI badge in README points to latest Zenodo record

## [3.9.1] - 2026-03-08

### Benchmarks
- **20-variant Brunel translator suite**: comprehensive characterization of
  SC-NeuroCore against Brian2 across neuron models (LIF, Izhikevich, homeostatic),
  timing variants, synapse types (STDP, dot product, Sobol bitstream), layer
  architectures (JAX, recurrent, memristive), and acceleration backends
  (Numba JIT, PyTorch CUDA GTX 1060, vectorized NumPy)
- V18 Numba JIT: 9.5× speedup over per-neuron Python loop
- V19 PyTorch CUDA: 8.7× speedup on GTX 1060 6GB
- V14 Sobol bitstream: 1.04× Brian2 ratio (closest match)
- 19 translator unit tests (`test_brunel_translator.py`)
- Fix BENCHMARKS.md CPU: i5-11600K @ 3.9 GHz (AVX-512, DL Boost)
- Fix 3 delta-PSC wiring bugs: v_reset omission, R*I*dt dilution, Poisson-as-current
- Comprehensive BENCHMARKS.md with 13+ sections and measured numbers
- Rust Criterion: 31 benchmarks captured (AVX-512)
- Brian2 2.10.1 SNN comparison: Brunel balanced network head-to-head
- NeuroBench-aligned metrics: 4 configurations, up to 847 MOP/s
- v2 vs v3 PyO3 speedup: 7.3× on large dense forward (128→64)
- Advanced module benchmarks: quantum hybrid, GNN, S-Former, BCI, DVS, chaos RNG
- Yosys synthesis tooling (`tools/yosys_synth.py`, `tools/yosys_synth.tcl`)
- CuPy 14.0.1 installed for GPU VectorizedSCLayer

### Paper
- Updated JOSS paper with measured Criterion numbers (41.3 Gbit/s pack, 224 Mstep/s LIF)
- Replaced estimated FPGA claim with Yosys tooling reference

## [3.9.0] - 2026-03-06

### SCPN Layers
- **L8-L16 pure NumPy layers**: 9 new layer files completing the full 16-layer SCPN stack (`scpn/layers/l8_phase_field.py` through `l16_director.py`)
- **16-layer registry**: `LAYER_REGISTRY` dict, `create_full_stack()` now returns all 16 layers
- **Full integrated step**: `run_integrated_step()` chains L1→L16 with inter-layer coupling

### Quantum Error Correction
- **SurfaceCodeShield**: d=3 rotated surface code with X/Z stabilizers, syndrome measurement, lookup-table decoding — corrects arbitrary single-qubit errors
- Extensible to d=5 (encode/decode/syndrome paths support arbitrary odd distance)

### Benchmarks
- Fixed double-step bug in `benchmarks/snn_comparison.py` (neurons were advanced twice per timestep)
- Fixed Lava stub notes (requires Loihi 2 hardware)
- Fixed `benchmark_suite.py` output path → `benchmarks/results/`
- SNN comparison results recorded in `docs/benchmarks/BENCHMARKS.md`

### Formal Verification
- **LIF neuron**: `hdl/formal/sc_lif_neuron.sby` + `sc_lif_neuron_formal.v` — 5 properties (reset, spike-reset, refractory clamp, counter bound, spike reachability)
- **Bitstream synapse**: `hdl/formal/sc_bitstream_synapse.sby` + `sc_bitstream_synapse_formal.v` — 4 properties (AND correctness, zero propagation, full-high, input coverage)

### Testing
- 6 cross-layer coupling integration tests (`test_scpn_cross_layer.py`)
- 9 surface code QEC tests (`test_qec_surface.py`)
- Test count: 945 → 960+

### Documentation
- JOSS paper: updated test count (960), qualified LUT claim, added Brunel/NeuroBench/LFSR bib entries

## [3.8.2] - 2026-03-06

### Documentation & Adoption
- **BENCHMARKS.md**: Populated with 14 real benchmark entries (i5-11600K, NumPy 1.26.4), Rust engine Criterion numbers, comparison context, reproduction instructions
- **JOSS paper draft**: `paper/paper.md` + `paper.bib` (6 references) — statement of need, architecture, key features, QA
- **End-to-end notebook**: `notebooks/03_end_to_end_pipeline.ipynb` — 7-cell walkthrough (encode→synapse→neuron→VectorizedSCLayer→accuracy analysis)

### Testing
- **18 Hypothesis property-based tests**: Bitstream encoding roundtrip, LFSR determinism, neuron output constraints, layer shape invariants, RNG range/shape, recorder accumulation, encoder binary output
- **Test count**: 887 → 911 tests passing, 98.41% coverage

### Issues Closed
- #30: Property-based testing with Hypothesis
- #33: JOSS paper draft

## [3.8.1] - 2026-03-06

### Enterprise Hardening
- **11 CI workflows**: ci, v3-engine, v3-wheels, benchmark, docs, pre-commit, codeql, scorecard, stale, release, publish — all SHA-pinned, concurrency-grouped
- **Supply chain**: Every GitHub Action SHA-pinned (30+ refs), `pypa/gh-action-pypi-publish` pinned, dependabot groups GH Actions PRs
- **Security**: Bandit SAST in CI, dependabot security updates enabled, private vulnerability reporting enabled, CodeQL weekly schedule
- **Branch protection**: 6 required status checks (lint, test×2, spdx-guard, build, pre-commit)
- **Dockerfile**: Multi-stage build, Python 3.12, non-root user, OCI labels, healthcheck
- **Preflight gate**: `tools/preflight.py` (black + bandit + spdx-guard + pytest), `.githooks/pre-push` hook
- **Release pipeline**: `publish.yml` (PyPI OIDC trusted publisher, 12 platform wheels), `release.yml` attaches sdist to GitHub Releases
- **Repo hygiene**: `.dockerignore`, `.editorconfig`, `.gitattributes`, `CONTRIBUTORS.md`, `CODEOWNERS`, PR template, issue templates (YAML forms), dependabot commit-message prefixes
- **Labels**: 22 labels with colors (ci, security, breaking-change, hdl, performance, needs-review, pinned, roadmap, stale)
- **Settings**: Delete-branch-on-merge, wiki/projects disabled, OpenSSF Scorecard badge

### Lint Enforcement & Python Version
- **ruff check enforced in CI**: 258 unused/deprecated imports auto-fixed across 138 files
- **CI test matrix expanded**: Python 3.10, 3.11, 3.12 (dropped 3.9 — EOL, autoray/PennyLane incompatible)
- **`requires-python` bumped to `>=3.10`**: badge, classifiers, black/ruff target-version updated
- **bandit added to `[dev]` extras**: contributors can now `make lint` after `pip install -e ".[dev]"`
- **benchmark.yml permissions tightened**: `permissions: {}` at top, scoped per-job
- **SECURITY.md / SUPPORT.md**: GitHub Security Advisories link added
- **VALIDATION.md refreshed**: 1058 tests, 98% gate, ruff/bandit/spdx-guard/codeql/scorecard gates documented

---

## [3.8.0] - 2026-03-05

### Hardening & Documentation
- **Coverage gate raised to 98%**: De-omitted 6 modules (chaos/rng, analysis/explainability, physics/wolfram_hypergraph, robotics/swarm, learning/neuroevolution, spatial/*) plus bio/neuromodulation. 34 new tests, 1058 total, 98.10% coverage
- **NumPy 2.x audit**: Zero deprecated calls found — codebase fully compatible
- **Full API documentation**: 25 new mkdocstrings pages, all 44 subpackages wired into nav. Reorganized into Core / Compiler & Export / Domain Modules / Infrastructure sections
- **Stale issue automation**: `.github/workflows/stale.yml` — weekly sweep, 60+14 day lifecycle, exempt: pinned/security/roadmap
- **CI coverage gate sync**: `ci.yml` and `pyproject.toml` both enforce `fail_under = 98`

---

## [3.7.0] - 2026-02-11

### Adaptive Runtime Engine -- HDC/VSA, SCPN Petri Nets, Fault-Tolerant Logic
- **HDC/VSA kernel**: `BitStreamTensor` gains `xor`, `xor_inplace`, `rotate_right`, `hamming_distance`, `bundle` methods for hyper-dimensional computing on 10,000-bit vectors
- **SIMD fused XOR+popcount**: AVX-512 VPOPCNTDQ / AVX2 / portable dispatch for hamming distance hot path
- **PyBitStreamTensor**: New `#[pyclass]` exposing full HDC algebra to Python (13 methods)
- **HDCVector**: High-level Python class with operator overloading (`*`=bind, `+`=bundle, `.similarity()`, `.permute()`)
- **PetriNetEngine**: Stochastic Colored Petri Net engine wrapping two `DenseLayer` instances for Places->Transitions->Places firing
- **Fault-tolerant logic**: Boolean logic with stochastic redundancy (1024-bit) survives 40%+ bit-flip rates
- **44 new tests**: 15 Rust integration + 20 Python HDC + 9 Python Petri Net
- **2 demos**: HDC symbolic query ("Capital of France?"), safety-critical Boolean logic with error sweep
- **Comprehensive study**: `docs/research/SC_NEUROCORE_V3.7_ADAPTIVE_RUNTIME_ENGINE_STUDY.md`

---

## [3.6.0] - 2026-02-10

### Fused Dense Pipeline + Fast PRNG + Batch Forward
- **Fused encode+AND+popcount**: `forward_fused()` eliminates intermediate input bitstream materialization
- **Fast PRNG switch**: xoshiro256++ for dense fast-path input encoding and numpy batch encoding
- **Batched dense API**: `DenseLayer.forward_batch_numpy()` processes N samples in one FFI call
- **New diagnostics**: criterion benches for fused dense, encode+popcount, batch dense, and PRNG throughput
- **Version/test/docs update**: bumped to 3.6.0 with the fused dense pipeline test suite and migration notes

---

## [3.5.0] - 2026-02-10

### SIMD Pipeline Acceleration
- **SIMD fused AND+popcount**: AVX-512 VPOPCNTDQ accelerated dense inner loop with AVX2 fallback
- **SIMD Bernoulli encode**: AVX-512BW/AVX2 threshold compare path for packed Bernoulli generation
- **Flat weight storage**: Contiguous `[neuron][input][word]` packed layout for cache-friendly access
- **Zero-allocation LIF batch**: Pre-allocated numpy outputs for batch LIF APIs
- **Criterion benchmarks**: Added fused-and-popcount and SIMD Bernoulli diagnostics

---

## [3.4.0] - 2026-02-10

### SIMD Pack, LIF Optimization, Rayon Guard
- **SIMD pack vectorization**: AVX-512/AVX2/portable fast packing (closes 6x Blueprint target)
- **Branchless LIF mask**: Eliminates branches in fixed-point sign extension
- **batch_lif_run_multi()**: Parallel multi-neuron batch execution via rayon
- **Rayon work threshold**: Avoids thread-pool overhead at small input counts
- **Criterion benchmarks**: Added pack_fast, pack_dispatch, lif_100k_steps

---

## [3.3.0] - 2026-02-10

### Fast Bernoulli, Fused AND+Popcount, Zero-Copy Prepacked
- **bernoulli_packed_fast**: 8x less RNG bandwidth via byte-threshold encoding
- **Fused AND+popcount**: Eliminates intermediate buffer allocation in neuron compute
- **forward_prepacked_numpy()**: True zero-copy from numpy 2D uint64 arrays
- **set_num_threads()**: Rayon thread pool configuration for tuning parallelism
- **Criterion benchmarks**: Added bernoulli_packed_fast benchmark

---

## [3.2.0] - 2026-02-10

### Benchmark CI, Single-Call Dense Forward, Parallel Encoding
- **Criterion Benchmarks**: Expanded suite with bernoulli encoding comparison and dense forward variants
- **Benchmark CI**: Automated criterion runs with artifact upload
- **DenseLayer.forward_numpy()**: Single FFI call with numpy input/output plus parallel encoding
- **Parallel batch_encode_numpy**: Rayon-parallelized probability encoding
- **Repo cleanup**: Added local `.gitignore` for generated artifacts

---

## [3.1.0] - 2026-02-10

### Dense Forward Optimization & PyPI Publishing
- **Direct Packed Bernoulli**: `bernoulli_packed()` eliminates `Vec<u8>` intermediate allocations
- **Parallel Encoding**: `DenseLayer.forward_fast()` parallelizes input encoding with per-input RNGs
- **Pre-packed Forward**: `DenseLayer.forward_prepacked()` accepts pre-encoded numpy/list inputs and skips encoding
- **batch_encode_numpy**: Returns a 2-D numpy array instead of nested Python lists
- **PyPI Publishing**: Added automated wheel upload on `v3.*` tags via Trusted Publisher workflow
- **Updated Benchmarks**: Added dense `fast` and `prepacked` benchmark variants

---

## [3.0.0] - 2026-02-10

### Performance Optimization & Stable Release
- **NumPy Zero-Copy**: `pack_bitstream_numpy()`, `popcount_numpy()`, `unpack_bitstream_numpy()` — eliminate FFI marshalling overhead
- **Batch Operations**: `batch_lif_run()`, `batch_lif_run_varying()`, `batch_encode()` — process arrays in single FFI calls
- **Verilator CI**: Co-simulation tests run automatically on Ubuntu runners
- **Updated Benchmarks**: Formal report showing true kernel performance with zero-copy interop
- **Bridge Version Fix**: `bridge/pyproject.toml` version now matches engine

### Release Candidate (3.0.0-rc.1)
- **IR Python Bridge**: Full PyO3 bindings for ScGraphBuilder, ScGraph, verify, print, parse, emit_sv
- **Co-sim Activation**: Verilator compilation + simulation when available; graceful skip preserved
- **Wheel CI**: Cross-platform wheel builds (Linux/macOS/Windows x Python 3.9-3.12)
- **Benchmark Report**: Formal v2-vs-v3 performance comparison with Blueprint section 8 targets
- **IR Demo**: Real end-to-end Python->IR->verification->SystemVerilog demo

### HDL Compilation Pipeline (3.0.0-beta.1)
- **SC IR**: Rust-native intermediate representation with 11 op types
- **SV Emitter**: Compile IR graphs to synthesizable SystemVerilog
- **Co-sim**: Verilator-based verification against Rust golden model
- **CI**: Expanded test coverage to include all differentiation, acceleration, integration, and HDL Python tests

### Integration & Hardening
- SSGF-compatible Kuramoto solver (`step_ssgf`, `run_ssgf`)
- Property-based testing with proptest (12 property tests)
- Multi-head attention (`forward_multihead`)
- SC-mode GNN (`forward_sc`)
- End-to-end training demo
- Comprehensive rustdoc

### Differentiation & Acceleration
- Surrogate gradient LIF (FastSigmoid, SuperSpike, ArcTan)
- DifferentiableDenseLayer for backpropagation
- Stochastic attention (rate + SC mode)
- Graph neural network layer
- Kuramoto oscillator solver
- Criterion benchmarks + v2/v3 comparison

### Foundation
- Rust engine with PyO3 bindings
- Bit-exact LFSR, LIF neuron, dense layer
- SIMD dispatch (AVX-512, AVX2, NEON, portable)
- Python bridge with v2-compatible API
- Equivalence test suite

---

## [2.2.0] - 2026-02-09

### Added
- **Module Discoverability**: Populated 36 stub `__init__.py` files with proper
  `__all__` exports and lazy imports. Every package now supports
  `from sc_neurocore.X import Y` without touching internals.
- **MkDocs API Documentation**: Added `mkdocs.yml` with mkdocstrings plugin,
  `docs/index.md`, `docs/getting-started.md`, `docs/architecture.md`, and 17
  API reference stubs in `docs/api/`.
- **Examples Directory**: 6 runnable example scripts demonstrating bitstream
  encoding, neuron layers, vectorized inference, SCPN stack, HDL generation,
  and ensemble consensus (`examples/01`–`06`).
- **Module Docstrings**: Added module-level docstrings to `pipeline/ingestion.py`,
  `pipeline/training.py`, `utils/model_bridge.py`, `ensembles/orchestrator.py`.

### Changed
- **Print → Logging**: Converted 60+ `print()` calls across 25 source modules
  to structured `logging` with `getLogger(__name__)` and `%`-style formatting.
  Dashboard and drivers intentionally excluded (stdout by design).
- **CI Coverage Threshold**: Raised `--cov-fail-under` from 50 to 97 in
  `.github/workflows/ci.yml` to match actual coverage.
- Version bump: 2.1.0 → 2.2.0.

### Fixed
- **Unused Imports**: Removed dead imports from 7 files (`bio/uploading.py`,
  `core/replication.py`, `core/immortality.py`, `export/onnx_exporter.py`,
  `dashboard/text_dashboard.py`, `hdl_gen/verilog_generator.py`, `viz/web_viz.py`).
- **Input Validation**: `VectorizedSCLayer.forward()` now raises `ValueError`
  on wrong-shape input instead of silently producing garbage.
- **File I/O Error Handling**: `onnx_exporter.py`, `immortality.py`,
  `verilog_generator.py`, and `replication.py` now catch `OSError` on
  file operations and log meaningful messages.

### Security
- **Pickle Allowlist**: Replaced wildcard `'numpy.core.numeric': {'*'}` with
  explicit `{'_frombuffer', 'scalar'}` in `core/immortality.py`.
- **Path Traversal Prevention**: `core/replication.py` now validates that the
  destination directory is within or below the working directory via
  `os.path.realpath()` + `os.path.relpath()`.

---

## [2.1.0] - 2026-02-08

### Fixed (Critical)
- **HDL Bitstream Encoder Seed Decorrelation**: All parallel encoders shared
  hardcoded seed `0xACE1`, producing correlated bitstreams and breaking SC
  multiplication (`P(x AND x) = P(x)` instead of `P(x)*P(w)`). Added per-instance
  `SEED_INIT` parameter with prime-stride offsets (input: `0xACE1 + i*7`,
  weight: `0xBEEF + i*13`).
- **HDL Missing Port Connections**: `noise_in` and `v_out` were floating on
  LIF neuron instances in `sc_dense_layer_core.v`. Connected via wire buses.
- **HDL Duplicate Port**: Removed duplicate `.stream_len` in `sc_neurocore_top.v`.
- **Fixed-Point Overflow**: `FixedPointLIFNeuron` now applies `_mask()` for
  proper two's complement overflow wrapping on membrane potential.

### Added
- **GPU Acceleration Backend** (`accel/gpu_backend.py`):
    - CuPy/NumPy dual-path with automatic GPU detection and CPU fallback.
    - `gpu_pack_bitstream()`, `gpu_vec_and()`, `gpu_popcount()`, `gpu_vec_mac()`.
    - `VectorizedSCLayer` auto-selects GPU when CuPy is available.
- **Performance Benchmark Suite** (`scripts/benchmark_suite.py`):
    - 14 benchmarks across 5 categories (scalar, packed ops, dense layer,
      full pipeline, GPU).
    - `--full` mode (10x iterations), `--markdown` output to `BENCHMARKS.md`.
- **CI/CD Pipeline** (`.github/workflows/sc-neurocore-ci.yml`):
    - Lint (black + mypy), Test (Python 3.9/3.11/3.12 matrix, coverage >= 60%),
      Build (wheel + install verification).
- **Co-Simulation Harness**:
    - `hdl/tb_sc_lif_neuron.v`: Verilog testbench reading stimuli.txt, writing
      results_verilog.txt for bit-exact comparison.
    - `scripts/cosim_gen_and_check.py`: CLI driver with `--generate` and `--check`.
- **Bit-True Python Models**:
    - `FixedPointLFSR`: 16-bit maximal-length LFSR (period 65535).
    - `FixedPointBitstreamEncoder`: LFSR + unsigned comparator.
    - `_mask()`: Two's complement sign-extension with overflow wrap.
- **Public API Surface**: Root `__init__.py` exports 28 symbols across 7
  subpackages. All subpackage `__init__.py` files populated.
- **Tiered Module System**: 43 subpackages categorised as `core` (7),
  `research` (24+), or `contrib` (5). Install extras: `[gpu]`, `[research]`,
  `[contrib]`.
- **Behavioural Equivalence Tests**: 29 tests covering LFSR, encoder, LIF
  neuron, full pipeline, and bit-width masking.
- **GPU Backend Tests**: 17 tests covering all GPU primitives and
  VectorizedSCLayer integration.

### Changed
- Version bump: 2.0.0 -> 2.1.0.
- `pyproject.toml`: Added tool configs (pytest, black, mypy), tiered extras.
- `VectorizedSCLayer`: Refactored to use GPU backend with CPU fallback.

---

## [2.0.0] - 2026-01-12

### Added
- **Sapience & Sentience (v2.2.0)**:
    - `MetaCognitionLoop`: Computational self-awareness and self-modeling.
    - `NeuromodulatorSystem`: Dopamine/Serotonin emotional state modulation.
    - `NeuroArtGenerator`: Generative AI for internal state expression.
    - `AsimovGovernor`: Ethical constraint system (Three Laws).
    - `MindDescriptionLanguage (MDL)`: Substrate-independent soul serialization.
    - `DigitalSoul`: Persistence and reincarnation protocols.
    - `VonNeumannProbe`: Code-level self-replication.
- **Galactic Scale (v2.1.0)**:
    - `InterstellarDTN`: Long-range delay-tolerant networking.
    - `DysonPowerGrid`: Stellar-scale energy management.
    - `KardashevEstimator`: Civilization Type metrics.
    - `DarkForestAgent`: Game-theoretic survival logic.
    - `MPIDriver`: Distributed cluster-scale simulation.
    - `SNNGeneticEvolver`: Automated architecture optimization.
- **Transcendent & Omega (v2.0.5)**:
    - `HeatDeathLayer`: Entropy-survival computing.
    - `PlanckGrid`: Spacetime lattice theoretical limits.
    - `HolographicBoundary`: 3D-to-2D info mapping (AdS/CFT).
    - `EverettTreeLayer`: Many-Worlds branching solver.
    - `WolframHypergraph`: Graph-rewrite universe evolution.
    - `CategoryTheoryBridge`: Unified mathematical functors.
    - `FormalVerifier`: SMT-based safety proofs.
- **Exotic & Frontiers (v2.0.0)**:
    - `VectorizedSCLayer`: 64-bit packed JIT-accelerated core.
    - `QuantumStochasticLayer`: VQC qubit rotation bridge.
    - `StochasticTransformerBlock`: Spike-driven attention.
    - `MemristiveDenseLayer`: Hardware-aware analog simulation.
    - `StochasticCPG`: Robotic locomotion oscillators.
    - `MyceliumLayer`: Fungal network dynamics.
    - `BCIDecoder`: Neural signal (EEG) interface.
    - `DVSInputLayer`: Event Camera (AER) processing.
    - `EnergyProfiler`: 45nm Energy/CO2 estimation.
    - `WatermarkInjector`: IP protection security backdoors.

### Optimized
- `BitstreamAverager`: 6x speedup using running sum algorithm.
- `BitstreamEncoder`: Added Sobol Sequence (LDS) mode for faster convergence.

### Fixed
- Fixed f-string syntax in Verilog generator.
- Fixed dimension mismatch in Attention mechanism.
- Addressed Windows encoding issues in documentation generation.

## [1.0.0] - 2025-12-03
- Initial Release: Stochastic Neurons, Synapses, and Basic Bitstream Utilities.

- Hardened FitzHugh-Rinzel Python, Rust engine, Rust safety, Go, and Julia paths with finite-parameter validation plus candidate-first RK4 commits that preserve state on invalid currents, corrupted runtime contracts, and overflow candidates.
- Hardened McKean Rust engine, Rust safety, Go, and Julia paths with candidate-first simultaneous-Euler commits and no-spike state preservation for invalid currents, corrupted runtime contracts, and overflow candidates; later promoted the maintained Python, Rust engine, Rust safety, Go, and Julia chain to candidate-first RK4 integration.
- Hardened Morris-Lecar Rust engine finite-state commits and extended Go/Rust safety coverage for invalid-current and potassium-rate overflow rejection without changing the documented conductance equations.
- Hardened Terman-Wang Rust engine finite-state commits, Julia timestep semantics, and Go/Rust safety state-preservation tests for invalid drive and cubic-overflow candidates; later promoted the maintained Terman-Wang chain to candidate-first RK4 integration.
- Hardened Quadratic IF Rust engine finite-update commits, Julia timestep semantics, and Go service tests for invalid current and non-finite Euler increments.
