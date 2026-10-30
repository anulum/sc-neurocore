# Candidate Models

A **candidate** is a proposed neuron model: one model written in the
Universal DSL (`sc_neurocore.neurons.universal_dsl`) together with everything a reviewer
needs to judge it. The Studio's **Candidate model** view imports, edits, validates,
diffs, simulates and reviews candidates, and exports both the candidate and a
review packet.

A candidate is a proposal. It is never listed as a catalogue model, and
nothing in the Studio writes a canonical file for it. Promoting a candidate
into the catalogue is a separate, authorised step that needs the evidence the
catalogue requires of every model.

## The package

A candidate package is one JSON document:

| Field | What it holds |
|-------|---------------|
| `schema_version` | `sc-neurocore.studio.candidate.v1`; any other version is refused |
| `name` | an identifier of at most 64 characters; it may not be a catalogue model's name |
| `parent` | the catalogue model the candidate derives from, or `null` |
| `model` | a Universal DSL schema: `metadata`, `state`, `parameters`, `integration`, `dynamics`, `threshold`, `reset` and any free-form sections the DSL carries |
| `units` | the unit of every state variable (`units.state`) and parameter (`units.parameters`), and of the injected `current` and of `time` |
| `source` | `citation` (required), and optionally `doi` and `url` |
| `assumptions` | at least one stated assumption |
| `authors` | at least one author |
| `reference_tests` | up to 16 tests the author proposes: a `name`, a constant `current`, a whole number of `steps` (1 to 100 000) and `expect` bounds on `spike_count` and/or on variables of `final_state` |

State variables and parameters are separate: a name may not be both, and
neither may be the input `I`, the noise sample `xi`, or a function name.
Expressions may read state, parameters, `I`, `xi`, and the functions and
constants of the equation namespace (`exp`, `log`, `sqrt`, `abs`, `sin`, `cos`,
`tanh`, `cosh`, `sinh`, `exprel`, `sigmoid`, `clip`, `max`, `min`, `pi`); they
pass the same safety gate as every Studio equation. Units are parsed with
[pint](https://pint.readthedocs.io), which reads unit expressions without
evaluating code; a unit it does not know, or a scaled unit such as `2*mV`, is
refused.

## Validation

Validation reports every problem at once, each with a JSON pointer to the
field that caused it — `/units/state/v`, `/model/dynamics/w`,
`/reference_tests/1/steps` — so the editor can show the message beside the
input. Unknown fields are refused rather than ignored, so an export carries
nothing the Studio did not check. A document with no field-level problem is
finally handed to the Universal DSL, which admits or refuses the model under
its own numerical profile; its refusal is reported at `/model`. Once that
profile is admitted, a static unit check inspects every executable expression,
including both branches of conditional expressions and every Boolean operand.
It does not run the model, draw noise, or convert the candidate's magnitudes.

### Dimensions and numeric scales

An ODE right-hand side must have the state variable's unit divided by the
declared time unit. A map right-hand side is the next state and must have the
state unit itself. Resets preserve state units; threshold predicates and
Poisson probabilities are unscaled dimensionless quantities; escape rates
have inverse-time units. Graded spike values may be physical scalars, with
their internal arithmetic checked. A declaration in `model.profile.units`
must agree with the package, and an explicit numerical time unit must agree
with `units.time`. ODE integration requires physical time.
An iteration-only map uses an unscaled dimensionless time unit. To give a map
a physical clock, author that clock explicitly in the model's numerical
profile and give the candidate the same unit; admission never supplies one.

Addition, subtraction, comparisons, clipping, extrema and conditional branches
require compatible dimensions **and the same numeric scale**. For example,
`v + I` cannot add millivolts to picoamps; `exp(v)` cannot exponentiate voltage;
and `v_mV + v_V` requires an explicit conversion written by the author. The
numeric DSL does not insert one. The usual AdEx identities `nS * mV = pA`
and `pA / pF = mV / ms` are accepted. A literal zero may take the other
operand's unit. Initially zero state variables and named zero parameters keep
their declared units and cannot conceal a mismatch.

The diffusion symbol `xi` is the runtime's `N(0,1)/sqrt(dt)` sample, so it has
inverse-square-root time units. For example, a voltage derivative
`sigma * xi` requires `sigma` in `mV / sqrt(ms)` when the clock is in `ms`.
It is not an unscaled normal draw. Validation inspects this unit without
consuming the seeded noise stream. The opt-in strict-unit equation runtime
uses the same noise-unit declaration; its existing conversion to base-unit
magnitudes remains separate from candidate admission.

| Function group | Unit rule |
|----------------|-----------|
| `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `erf`, `softplus`, `exprel`, `sigmoid`, `expit` | unscaled dimensionless argument and result |
| `sqrt`, `rsqrt` | square-root or inverse-square-root units |
| `abs`, `floor`, `ceil`, `trunc`, `round` | preserve the declared unit; rounding acts on its unchanged numeric magnitude |
| `sign`, `heaviside` | dimensionless result; `heaviside`'s value at zero must be dimensionless |
| `atan2` | same-scale compatible arguments, dimensionless angle |
| `hypot`, `clip`, `min`, `max` | same-scale compatible quantities, preserving their common unit |

Lists, tuples, permitted array arguments, literal indexing and slicing, and
the `real` and `imag` attributes retain quantity information. Optional output
arguments remain literal `None`; clipping bounds may be `None`. Every sequence
element is inspected, including elements not selected at the initial state.
A unitful base raised to a power needs a bounded, statically established
dimensionless exponent; a state-dependent exponent cannot establish a fixed
physical dimension. No arbitrary expression is executed to infer its units.
Affine and logarithmic units, such as Celsius and decibels, are refused;
write a model in finite multiplicative units. Scaled dimensionless units such
as percent and degrees must be made explicit before a dimensionless function.

Changing only `units.state.v` from `mV` to `ms` therefore refuses validation
and all candidate actions, with diagnostics beside the affected equations,
thresholds, resets and any contradictory declaration. This check establishes
internal dimensional and scale consistency. It does not establish that the
declared units or equations reproduce the cited scientific source.

Diagnostics use explanations written for authors. Numbers must be finite,
text must be valid Unicode, and every value must be representable in JSON.
If the document cannot be encoded, `candidate_sha256` is `null` and validation
reports the encoding problem. Valid finite documents retain their canonical
digest. Function calls are checked against the equation namespace's supported
arguments before admission; validation does not advance the model or draw noise.

## Diff against the parent

The diff compares the candidate with the canonical schema of its parent,
section by section. Values are listed as added, removed, changed or unchanged.
Equations are compared as mathematics as well as text: both sides are
converted from their syntax tree to [SymPy](https://www.sympy.org) — never
evaluated — and their difference is simplified. A rewritten equation that is
equal is reported as `equivalent`; a real change as `changed` with the
simplified difference; an equation using a construct with no exact symbolic
reading (a comparison, a conditional, a modulo) as `not_comparable`; and a
difference too large to simplify here as `undecided`. `exprel`, `sigmoid`,
`clip`, `max` and `min` stay uninterpreted functions, so two equations using
them are equal only when they use them identically. Threshold, reset and the
integration profile are compared too. A candidate without a parent, or whose
parent has no canonical schema, is told so instead of being diffed against
nothing.

## Simulation and reference tests

A candidate runs under its own declared profile — the method and timestep it
states — with a constant current, for at most 100 000 steps; longer traces are
returned at a stride of at most 5 000 samples per variable. A run whose state
stops being finite reports the step and the reason instead of numbers.
Static quantity and shape faults are refused during validation. A dimensionally
admitted expression can still divide by zero, leave a function's numerical
domain, or produce a non-finite state. It reports the failed step with an
authored reason and a `null` `final_state`. Its reference tests fail, and the review packet retains the
candidate and digest so the author can repair that proposal. Corrupt canonical
resources and internal engine faults fail the job; synchronous diff faults remain server errors.

The reference tests are the author's proposals. Running them shows that the
candidate does what its author says; it is not an independent check against
the cited source.

## The review packet

**Run reference tests** assembles the packet a reviewer receives: the
candidate unchanged, its validation, its diff against the parent, each test's
observed values and whether each bound held, the environment that produced
them, and what the packet does not establish:

- catalogue promotion;
- independent validation against the cited source;
- any fixed-point, RTL or co-simulation evidence.

The packet is bound under one `packet_sha256`, and the candidate under its own
`candidate_sha256`.

## Durable execution and recovery

**Simulate** and **Run reference tests** submit supervised process jobs and
return HTTP 202. Validation happens before job allocation. Each immutable
request is limited to 4,000,000 encoded bytes, further restricted by the
configured artifact ceiling. The job records its operation, original candidate
name, canonical candidate digest, exact ordered request digest, and projected
model steps (the simulation length or the sum of every reference-test length).
The original numerical profile, seed and magnitude conventions are preserved.

**Candidate jobs** lists the current actor's latest jobs; **Refresh candidate
jobs** recovers history after reconnecting or reopening the browser. Select a
stored job to recover its result. Editing or importing a different draft does
not rename the selected job or change its input identity. **Export review
packet** downloads that selected review's stored bytes under its original
candidate name. The request, candidate and result are also downloadable as
`request.json`, `candidate.json` and `result.json` once declared by the job.

**Cancel candidate job** requests a supervisor stop. `cancelling` means the
worker has not yet been confirmed stopped; `cancelled` and `timed_out` retain
distinct outcomes. An observation failure does not resubmit the computation.
Refresh explicitly retries observation. A completed calculation remains in the
configured durable job root across API restarts. If a supervisor dies during
execution, normal ledger recovery reports `interrupted` or `unknown` until its
worker custody is resolved. The calculation is not resumed or replayed.

The existing analysis resource class supplies finite process deadlines and
worker limits in embedded and isolated storage profiles. Embedded workers run
as the launching user; process dispatch does not establish OS isolation.
History responses contain at most 100 records (50 by default) and omit result
arrays. Individual status responses retain the full stored result. All new
custody routes enforce actor ownership; another actor receives no record,
control or artifact. Preserve the configured job root to retain this history.

Packages currently start through import and text editing; starting a candidate
directly from the catalogue is deferred. Independent scientific acceptance and
authorised catalogue promotion remain separate work, with no automatic
catalogue changes.

## In the workspace

The draft belongs to the workspace and is saved with it, in the workspace's
`candidates` block, exactly as typed — a half-written draft that is not yet
JSON is kept as well. **Export candidate** saves the draft byte for byte, so an
imported package exported again is the same file. A draft that is empty or is
not JSON is not sent to the server; the panel says why.

## For authors and reviewers

1. Start from a catalogue model's schema or an existing package, name the
   candidate and its parent, and state the source, the assumptions and every
   unit.
2. Validate until no problem remains, then diff against the parent and check
   that every change listed is one you intended.
3. Propose reference tests that pin the behaviour the source describes, run
   them, and export the review packet.
4. A reviewer reads the packet: the diff says what changed, the test results
   say what was observed, and the `not_established` list says what is still
   owed before the model could enter the catalogue.

## API

| Route | Answers |
|-------|---------|
| `POST /api/candidates/validate` | the located diagnostics, with HTTP 200 whether or not the candidate is valid |
| `POST /api/candidates/diff` | the diff against the parent |
| `POST /api/candidates/simulate` | HTTP 202 `studio.candidate.job.v1` receipt for a bounded simulation (`current`, `steps`) |
| `POST /api/candidates/review-packet` | HTTP 202 receipt for all proposed reference tests and the review packet |
| `GET /api/candidates/jobs?limit=50` | newest-first actor-owned summaries; `limit` is 1–100 |
| `GET /api/candidates/jobs/{job_id}` | actual state, immutable input identity and stored result |
| `POST /api/candidates/jobs/{job_id}/cancel` | actual state after requesting cancellation |
| `GET /api/candidates/jobs/{job_id}/artifacts/{artifact_path}` | declared bytes with SHA-256 and size headers |

Each computation or validation POST takes `{"candidate": <package>}`. The three routes that act on a
candidate refuse an invalid one with HTTP 422, `detail.reason` =
`invalid_candidate` and the same located diagnostics in
`detail.validation`. With route policies enforced, all candidate routes require an
authenticated principal, like the other simulation routes. Units and diffs
need `pint` and `sympy`, which the `studio` extra installs.

Existing direct numerical Python functions still return their results directly.
HTTP consumers must follow the new receipt's `status_route`, wait for
`status == "completed"`, then read `result`; HTTP 202 is admission, not success.
An unknown submission response must be investigated in history before a manual
retry, since a second submission can create a second job. This API does not
promise automatic submission idempotency.
