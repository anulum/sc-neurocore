# Candidate Models

A **candidate** is a proposed neuron model: one model written in the
Universal DSL (`sc_neurocore.neurons.universal_dsl`) together with everything a reviewer
needs to judge it. The Studio's **Candidate** tab imports, edits, validates,
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
its own numerical profile; its refusal is reported at `/model`.

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
- dimensional consistency of the equations against the declared units;
- any fixed-point, RTL or co-simulation evidence.

The packet is bound under one `packet_sha256`, and the candidate under its own
`candidate_sha256`.

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
| `POST /api/candidates/simulate` | a bounded run under the candidate's own profile (`current`, `steps`) |
| `POST /api/candidates/review-packet` | the reference-test results and the review packet |

Each takes `{"candidate": <package>}`. The three routes that act on a
candidate refuse an invalid one with HTTP 422, `detail.reason` =
`invalid_candidate` and the same located diagnostics in
`detail.validation`. With route policies enforced, all four require an
authenticated principal, like the other simulation routes. Units and diffs
need `pint` and `sympy`, which the `studio` extra installs.
