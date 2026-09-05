# Model profiles: scientific model, numerical realisation, lowering

A bundled schema (`src/sc_neurocore/neurons/model_schemas/<stem>.toml`, with a
JSON twin) states equations, state, parameters, a step, a threshold and a
reset. Those fields belong to three different things that must be judged
separately: what the *science* says, how the schema *advances* it in time, and
what the RTL emitter can *lower* from that. The model profile contract
`sc-neurocore.model-profile.v1` (`sc_neurocore.neurons.model_profile`) resolves
every schema into those three layers, derives what it can, lets a schema author
what it cannot, and reports every contradiction. It never changes a number.

## The three layers

| Layer | Carries | Source |
|---|---|---|
| scientific model | name, author, year, DOI; authored equations; biological state with units; source parameters with units; `science.equations_as_published` | schema metadata, `[state]`, `[parameters]`, `[dynamics]`, `[science]`, authored roles |
| numerical realisation | method and family; exactness class; `dt` and time unit; sub-steps and their kind; macro step; evaluation order; auxiliary registers; implementation and timebase parameters; admissible methods; randomness contract; event contract (detection, condition, stochastic expression, reset, edge logic, refractory register) | `[integration]`, `[threshold]`, `[reset]`, authored roles, derived rules |
| lowering profile | whether the equation compiler accepts the realisation, its limits, the methods Studio co-simulates, the recommended precision hint | mirrors `sc_neurocore.compiler` acceptance rules, `[hints]` |

Every layer has a JSON projection (`ModelProfile.to_public_dict`) and
`parse_profile` rebuilds an equal profile from it.

## Authoring a profile section

Everything a schema does not author is derived: all state is biological, all
parameters are source parameters, a parameter named `dt` (or `dt_ms`) that
equals `integration.dt` is a timebase parameter, sub-steps of an ODE subdivide
time, a map with `dt = 1` and no timebase is indexed by iteration. What cannot
be derived is authored in an optional `[profile]` table:

```toml
[profile]
contract = "sc-neurocore.model-profile.v1"
time_unit = "ms"
substep_kind = "stage-iteration"        # or "time-subdivision"
exactness = "exact-flow"                # optional claim, see below
refractory_register = "refractory_time" # a state variable holding the hold
notes = "..."

[profile.state]                          # role per state variable
v = "biological"
spike_flag = "auxiliary: event flag raised at stage 4"

[profile.parameters]                     # role per parameter
dt = "timebase"
delta_ge = "implementation: constant per-macro-step boundary conductance event"
c_m = "source"

[profile.units]                          # unit per state variable or parameter
v = "mV"
dt = "ms"
```

Roles are `biological` / `auxiliary` for state and `source` /
`implementation` / `timebase` for parameters; text after a colon is the
meaning. Twelve representative schemas author a section: `lapicque`,
`sc_lapicque_lif`, `lif`, `adex`, `hodgkin_huxley`, `wang_buzsaki`, `exp_if`,
`rulkov_map`, `chialvo_map`, `escape_rate`, `poisson`, `coba_lif`. The TOML
and JSON twins must carry the same section; the static validator checks it.

## Mapping table: method, exactness, family, lowering

| Method | Family | Derived exactness | Claimable exactness | Lowering |
|---|---|---|---|---|
| `euler` | ode | first-order | none | registered next-state datapath; sub-steps only for crossing, non-resetting models |
| `gauss_seidel` | ode | first-order-sequential | none | sequential derivative wires in declaration order; same sub-step limit |
| `rk4` | ode | fourth-order | none | four staged derivative evaluations per clock; same sub-step limit |
| `exp_euler` | ode | linearised-exponential | `exact-linear-relaxation`, verified: each equation is affine in its own variable and reads no other state | exprel-scaled increment from the symbolic Jacobian the golden compiles |
| `map` | map | recurrence | `exact-flow`, authored: the recurrence is a sampled closed-form solution | next state is the map value; stage iteration folds several clocks per macro step |

An event-only schema (no state, no dynamics) has family `event-only` and
exactness `event-only`. A schema whose method or detection is outside the
executable vocabulary (`euler`, `map`, `rk4`, `exp_euler`, `gauss_seidel`;
`level`, `crossing`, `escape_rate`, `poisson`) is a *descriptive record*: it
documents a hand model, cannot run through `UniversalNeuron`, has no macro step
and no lowering, and may not author a profile section. Thirteen bundled schemas
are descriptive records; the ledger names them.

Sub-steps have two meanings and the profile says which: `time-subdivision`
(each sub-step advances `dt`; the public step is `substeps × dt`, as in
Hodgkin-Huxley's 100 × 0.01 ms) or `stage-iteration` (the sub-steps are the
stages of one scheme folded into a map; the public step is one `dt`, as in the
COBA schema's four RK4 stage registers). A map with `substeps > 1` must declare
which; an ODE cannot declare stage iteration.

The evaluation order is derived from the family and the detection: integrate
(or iterate, or stage) × sub-steps, then the refractory hold encoded in the
dynamics if a register is named, then the threshold (level, or rising edge on
the macro boundary for a crossing model without a reset rule), or the hazard
and LFSR trial for an escape rate, or the probability and LFSR trial for a
Poisson bin, then the reset rule if any.

The randomness contract is `none`, `lfsr16-threshold` (a model-scoped 16-bit
LFSR seeded from `threshold.rng_seed` decides stochastic trials; replayable),
`diffusion-noise-global-rng` (an expression names `xi`, drawn from NumPy's
process-global stream; not replayable by the model alone), or both. No bundled
schema uses diffusion noise.

## Contradictions and admission

`resolve_profile` lists every contradiction as a problem; a problem makes the
schema unusable by the executable consumers. Contradictions include a role for
an undeclared quantity, a timebase parameter that differs from
`integration.dt` or whose unit differs from the time unit, a `dt` parameter the
expressions read that is not bound to the timebase, an exactness claim the
method cannot carry or the symbolic check refuses, `substep_kind` with one
sub-step, stage iteration on an ODE, a map with sub-steps and no declared kind,
admissible methods or `extensions.integrator_options` that leave the family,
and a profile section on a descriptive record.

Overrides are admitted through `admit_overrides`:

- a method must be one of the profile's admissible methods: any ODE method for
  an ODE (the realisation is then *derived* and reported as such), only `map`
  for a recurrence;
- a step override moves every timebase parameter with it, is refused when a
  parameter override contradicts the effective step, and is refused for a
  recurrence without a continuous timebase unless it equals the declared step;
- a seed is refused for a profile that draws no randomness.

## Where the contract is enforced

- `UniversalNeuron` resolves the profile of every schema and admits every
  override through it; `neuron.profile` is the authored profile and
  `neuron.realised_profile()` the realisation actually run.
- The static validator (`schema_validator`, `python -m sc_neurocore.neurons validate`)
  reports profile problems as errors, descriptive records as classified
  warnings, and checks the profile section for TOML/JSON parity.
- Studio's model detail carries `profile_contract`; the compile configuration
  offers only integrators inside the family, refuses one that leaves it, and
  the compile evidence records the realised profile next to the schema digest.
- The reference-trace runner instantiates every protocol through the profile,
  so a protocol step is admitted or refused by the same rules.
- The descriptor skeleton generator resolves the curated schema per class,
  not per module, so an `SC` compatibility identity never inherits its source
  identity's profile.

## Per-profile inventory and validator registry

`tools/model_profile_ledger.py --write` renders
`docs/_generated/model_profile_ledger.json` from
`sc_neurocore.neurons.profile_registry`: one row per bound (class, schema
stem) with the three layers, the descriptor's own integration label next to
the profile's method (a hand class may keep a different default than its
schema profile; the row says `same` or `differs`), the readiness record
verified for exactly that profile, and the validator registry: for every facet,
each validator the descriptor or the schema declares, its resolution, its scope
and the receipt bound to this profile.

Admission is the gate the source-oracle, native-admission and
dependency-closure work consume:

| Admission | Meaning |
|---|---|
| `admitted` | executable, contradiction-free, and every facet the descriptor declares through an evidence field has at least one executable validator scoped to this profile |
| `blocked` | executable, but a declared facet is backed by prose or a report alone, or only by class-scoped tests of another profile |
| `not-executable` | a descriptive record |

Descriptor evidence fields are class-scoped: they validate the class's
canonical profile (the one `model_identity.schema_for_class` returns). A
non-canonical profile is validated only by its own schema
`[validation].evidence` or by a receipt recorded for it, never by the canonical
profile's tests, so no test admits a profile it did not exercise. Backend
facets have no descriptor evidence field yet; declared-but-unvalidated backends
are listed on the row instead of deciding admission. `--check` fails when the
tracked ledger is stale; `--summary` prints the partition.

## Migration decisions and limits

- No equation, parameter, step, threshold or reset value changed. The twelve
  authored sections classify; the other schemas derive.
- The COBA schema declares its four map iterations as stage iteration of one
  0.1 ms step and its twelve stage registers as auxiliary, which is what the
  folded RK4 always was; the descriptor and the hand model are untouched.
- Four descriptors (`AiharaMapNeuron`, `NagumoSatoMapNeuron`,
  `SCAdaptiveThresholdMapNeuron`, `SCChaoticMapNeuron`) previously stated
  `method = "euler"`, `dt = 0.1` because the generator resolved their schema by
  module name and found none; regenerated per class they state the bound map
  profile (`map`, `dt = 1.0`). No numerics changed; the hand classes are maps.
- The `ermentrout_kopell_theta_euler_doi` reference protocol declared
  `dt = 1.0` for a map whose 0.1 Euler step is fixed inside its expression; the
  override was inert. It now declares the schema's 0.1 and its expected
  features are unchanged.
- Descriptor `integration.method` remains the hand class's label
  (`exact_constant_voltage_flow`, `baseline_euler`, …); the ledger joins it to
  the profile method and does not rewrite it.
- The `sc_wb_nmda_magnesium_block` record uses `dt = 0.5` as the macro step
  subdivided fifty times, the opposite of the executable convention; being a
  descriptive record it carries no macro step, and its hand model owns the
  correction.
- Units are declarations checked for consistency with the time unit; they are
  not dimensional analysis (that remains `EquationNeuron(units="strict")`).
- Admission judges validator existence and scope, not scientific sufficiency:
  an admitted profile still needs the independent source oracle, and a bound
  receipt still means one executed run, not replication.
