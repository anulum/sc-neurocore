<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Readiness evidence contracts -->

# Readiness evidence: declared versus verified

A model descriptor *declares* readiness: `validation.dynamics_faithful`,
`validation.metric` with `validation.evidence`, `silicon.compiles`,
`silicon.cosim_validated` with `silicon.cosim_evidence`, and so on up the
silicon ladder. The dual-axis tiers on the
[model fidelity status page](model_fidelity_status.md) and in Studio are
computed from those declarations (tier semantics v1: a boolean anchor plus a
non-empty evidence string credits a rung).

A declaration is not proof that the evidence exists, that it belongs to this
model, or that it is still current. This page describes the layer that
checks it. Nothing here changes the declared tiers; it reports, next to them,
how much of each declaration is bound to an executed and still-fresh receipt.

## Facets

Each rung is one facet with a fixed set of subjects (the inputs whose change
invalidates it):

| Facet | Axis | Rung | Required subjects | Descriptor field |
|---|---|---|---|---|
| `dynamics_faithful` | science | S4 | descriptor contract, model module, validator | `validation.evidence` |
| `class_validated` | science | S5 | descriptor contract, model module, validator | `validation.evidence` |
| `backend:python` | software | none | descriptor contract, model module, validator | `backends.python.evidence` |
| `backend:rust`, `backend:rust_safety`, `backend:julia`, `backend:go`, `backend:mojo` | software | none | descriptor contract, model module, validator, native source | `backends.<lane>.evidence` |
| `rtl_compile` | silicon | H0 | schema profile, compiler, validator | `silicon.cosim_evidence` |
| `cosim` | silicon | H1 | descriptor contract, model module, schema profile, compiler, validator | `silicon.cosim_evidence` |
| `synthesis` | silicon | H2 | committed RTL, report, validator | `silicon.synth_report` |
| `timing` | silicon | H3 | committed RTL, report, validator | `silicon.timing_report` |
| `formal_equivalence` | silicon | H4 | committed RTL, compiler, report, validator | `silicon.equivalence_proof` |
| `formal_safety` | silicon | none | committed RTL, report, validator | none |
| `ppa` | silicon | H5 | committed RTL, report, validator | `silicon.ppa_report` |
| `physical` | silicon | none | committed RTL, report, validator | none |

Subject kinds: the *descriptor contract* is the digest of the descriptor's
identity, state, parameter, integration and dynamics sections only, so a
documentation or evidence edit never invalidates a receipt while a changed
equation, default, `dt` or method always does. The *compiler* subject is the
digest of `src/sc_neurocore/compiler` plus the schema-DSL front end, so a
changed shared compiler invalidates every generated-RTL receipt at once. The
*validator* is the test file the evidence names. A bounded safety proof is
recorded under `formal_safety` and can never credit `formal_equivalence`: the
two facets require different claim scopes.

Every lane a descriptor may declare has one facet, sixteen facets in all. A
lane's evidence is the `evidence` text of its `[backends.<lane>]` table. The
*native source* subjects of a lane are what a repository can recompute on any
host: the files that implement the model in that lane and the manifests, lock
files, build configurations and tool pins that fix the lane's build
(`LANE_BUILD_INPUTS` in
`sc_neurocore.neurons.readiness` names them per lane). A source becomes a
subject only when it is a tracked file spelled as Git spells it: in a Git
checkout it must be in the index, compared as written, and elsewhere it must
be written in normal form; either way the file must exist. A declared lane
that lists any other source is `unavailable`, and the problem names the path,
before a receipt is judged; where a checkout's index cannot be read, every
source is reported unchecked, with the reason. The Rust lane also binds every
tracked file of the engine's Python package and of the vendored crate that
the bridge's and the root Cargo configuration build in place of a registry
copy (`LANE_BUILD_TREES`), each with its own digest. The library a run
loaded is a fact of that run and never a subject, because a built file is
absent on a clean checkout and differs between hosts. It is kept as an
*artefact* of the receipt: a receipt of `backend:rust`,
`backend:rust_safety`, `backend:go` or `backend:mojo` credits only when it
carries the `native-binary` artefact, and a receipt of any other facet credits
only when it carries no artefact.

A backend lane's receipt states as its claim scope the word the model's
descriptor gives the lane (`backend_claim_scope`): the parity word of a
compiled lane (`exact`, `ulp-bounded` or `approximate`), and `reference` for
the Python lane, which is compared with itself. The recorder refuses a lane
whose parity is `n/a`, and a claim scope given by hand that is another word.
The parity word lies outside the descriptor's contract digest, so a change of
it moves no subject; the verifier compares the receipt's claim scope with the
descriptor of today and reads the receipt `invalid` when they differ.

## Facet receipts

Receipt schema v2 binds the result to `(class, facet, profile)`. Select the
profile explicitly for models with several numerical definitions; an unselected
multi-profile model receives no class-wide verified claim. For example,
`verify_model("LapicqueNeuron", profile="lapicque")` checks only that profile,
not its alternative LIF definition. Returned records identify the selected
profile. This selection is not a claim that the underlying numerical profiles
are mathematically interchangeable.

The recorder accepts a direct pytest invocation and retains its JUnit testcase
identities and results. Selected evidence must be declared by the descriptor
and actually appear in that execution. `--evidence` can select a reviewed subset
from a descriptor listing several validators; the receipt states that subset,
not execution of all the other tests. A successful unrelated command, a passing
different test or invented suite totals cannot credit the selected facet.
Native-tool result adapters remain unsupported rather than crediting exit zero
as a scientific result. No shipped descriptor names a lane's evidence or
sources yet, so no lane has a receipt and every declared lane reads
`declared`.

Current subject membership and digests are checked, including the recorder and
verifier implementations. Scientific metric/tolerance edits invalidate evidence.
The recorder checks inputs again after execution and refuses credit if they
changed. Receipt publication is atomic and exclusive; an existing file is never
overwritten. Default execution timeout is 600 seconds; a positive finite override
is allowed. POSIX execution waits for the owned process group, including members
remaining after its leader exits; Windows execution owns the direct child.
Timeout stops that owned scope. Cleanup has a separate finite budget equal to
the execution timeout.
If the validator cannot be started, the CLI reports a fixed refusal and writes
neither a receipt nor a raw execution archive.
Missing formal mapping code or required local model inputs also refuses the
run before starting a validator. For `cosim`, committed RTL is an optional
freshness subject; the descriptor, model, schema, compiler and validator
subjects remain mandatory.

Historical v1 receipts remain readable and immutable but cannot provide v2
credit. They need a fresh run, not a changed seal. SHA-256 seals detect content
changes; they are not signatures, independent scientific review or proof of
the recorder's identity. A receipt covers its declared inputs and validated
domain, not arbitrary imported dependencies or all possible operating points.

### Lane evidence written by the run

A backend lane's receipt must name the lane that ran, not the one a test asked
for. A model that offers compiled lanes records, together with the state update
of a complete run, the lane that executed and the native library files it
loaded (`sc_neurocore.neurons.lane_runs`; `LapicqueNeuron` is the first such
model). The record is kept outside the model's attributes, because the Rust
dispatch of a population, the quiescence signature and the Studio's mutation
audit read every attribute a model carries. The test fixture `lane_evidence`
(`tests/lane_evidence_support.py`) writes the record into the running test's
JUnit report: one `sc_neurocore.lane` property and one
`sc_neurocore.native_binary` property per library, as `<file name>=<sha256>`,
under pytest's default report family. `junit_lane_evidence` in
`sc_neurocore.neurons.receipt_execution` returns each executed case's lane and
library digests and every inconsistency: a case with no lane or with two, a
malformed library property, one library name of one lane with two digests in a
report (two lanes may load libraries of the same name). A test module that
writes lane evidence registers the fixture's module with `pytest_plugins`.

The recorder reads the properties from the report it retains in the receipt
(`lane_run_evidence` in `sc_neurocore.neurons.facet_receipts`): every executed
case must report the facet's lane, a case of a lane that builds a library must
name one, and each library name must be a plain file name. The distinct
libraries become the receipt's `native-binary` artefacts, by file name and
digest. A problem makes the outcome `error`, each problem a line of the notes.
The verifier derives the same from the retained report and refuses a receipt
whose artefacts are not exactly those libraries. For a backend facet the
receipt's `extra_tools` holds the version line of each tool of the lane's
toolchain that answers (`rustc` and `cargo`, `go`, `mojo`, `julia`), asked in
the folder the lane's build starts in. The line names the toolchain the
recording environment resolves there; the library that ran is identified by
its digest.

### Runtime identity and public execution projection

New executions use the `repository-relative-execution.v1` public projection.
The command records a logical `python -m pytest` or `pytest` invocation with
repository-relative selectors. A plugin running inside the actual validator
process records its Python implementation, version and executable SHA-256,
and the pytest version used by that process. The recorder's own interpreter
is not substituted for these facts. Missing runtime evidence refuses credit.

Public JUnit XML omits hostname attributes and private runtime-location
properties. Known repository, home, interpreter and temporary locations in
diagnostic text become descriptive tokens. Every testcase identity, outcome,
count and timing remains present; a private identity in a testcase identifier
refuses credit rather than changing which validator ran. Unclassified
private locations also refuse credit.

Exact original stdout, stderr and JUnit bytes, together with actual argv,
working directory and exit code, remain in a fresh private execution directory.
Output is captured in private temporary files and snapshotted when the owned
scope finishes or its cleanup returns. Output handles inherited by detached
processes cannot make capture wait indefinitely. Those detached processes are
outside the recorder's ownership; their later writes are outside the snapshot.
POSIX snapshots use positional reads bounded by the initially observed file size.
Concurrent truncation can shorten that snapshot without moving the inherited
writer's offset or mapping pages that another process can invalidate.
The receipt retains their SHA-256 values and the exact private manifest hash.
The default archive is `docs/internal/receipt_runs/`; `--raw-evidence-dir`
selects another private archive. Keep this archive outside public documentation,
packages and release artifacts. These additive fields use the existing v2
string maps; historical seals are never edited to claim a new execution.

| Public field | Meaning |
|---|---|
| `command` | Logical direct pytest command and repository-relative arguments |
| `runtime.implementation`, `runtime.python` | Implementation and version observed inside the validator process |
| `runtime.executable_sha256` | SHA-256 of that process's resolved interpreter binary |
| `tool.version` | The validator process's pytest version |
| `validator.public_projection` | The location-normalisation contract applied to its public evidence |
| `tool.raw_manifest_sha256` | Hash of the exact private `manifest.json` bytes |
| `tool.raw_junit_sha256`, `tool.raw_stdout_sha256`, `tool.raw_stderr_sha256` | Hashes of the exact original artifacts before decoding or projection |

Each private execution directory contains `manifest.json`, `junit.xml`,
`stdout.bin` and `stderr.bin`. The manifest records original and executed argv,
working directory, exit code, artifact hashes and projection contract. A failed
projection preserves these originals and refuses scientific credit. The
low-level command runner still reports the measured process outcome; that
outcome alone does not establish receipt credit.

A parent interruption retains the exact captured artifacts before propagating
the interruption, and creates no public receipt. On POSIX the recorder terminates
the validator process group and reaps its direct child before saving the run.
The group is terminated even when its leader has already exited while a
remaining member still owns the captured output handles. Failure to reap the
direct child within the cleanup budget refuses publication after retaining the
captured artifacts.

A receipt records one execution of the evidence command. Use the publication
command to record it and refresh every catalogue view together:

```
python tools/record_readiness_receipt.py record --model LapicqueNeuron --facet cosim --profile lapicque \
    -- python -m pytest "tests/test_cosim_lapicque.py::test_source_q3232_preserves_first_attainment_and_polarization_bound"
```

The recorder derives the facet's subjects from the identity registry, runs the
command (a pytest command runs with a JUnit report so the passed, failed and
skipped counts come from the run itself), records the tool and runtime
versions, the git head and any uncommitted subject, seals the payload with its
own SHA-256 and writes it as a new file under
`src/sc_neurocore/neurons/facet_receipts/`. Receipts are append-only: the
recorder refuses to overwrite, and the verifier reads the newest receipt per
(class, facet, profile). A receipt credits its facet only when it is sealed, names the
class it is read for, carries every required subject kind, ended with
`outcome = "passed"`, exit code 0, at least one passed check and no failed,
errored or skipped check, and states the claim scope the facet requires.

Native tool metadata uses the first nonempty line of a successful `-V` or
`--version` invocation. Unavailable tools and probes that supply no successful
version leave the version empty.

The publication command also creates the receipt's adjacent `.json.license`
ownership sidecar and refreshes the model profile ledger, readiness evidence
ledger and packaged Studio readiness seal. It refreshes those views for failed
or uncreditable runs too, so an older verified label cannot conceal a newer
failed validator. It returns zero only after a creditable run and all three
updates; an uncreditable run returns one, and a publication failure returns two.
Stage the receipt, its sidecar and the three generated files in the same commit.

If updating the views fails, the immutable receipt remains. Repair the reported
problem and refresh the views without running the validator again:

```
python tools/record_readiness_receipt.py refresh
```

All three views are rendered and staged before replacement; each individual
file is replaced atomically. The low-level `tools/facet_receipt.py` recorder
remains available for archive-only receipt directories. Those directories do
not feed the catalogue and are refused by the publication command.

The fast readiness regression checks that the established AdaptiveThresholdIF,
BrunelWang, Lapicque and ResonateAndFire co-simulation profiles retain their
source-bound receipts. A shared compiler change therefore exposes withdrawn
labels before the longer distribution checks run.

## Recorder validation

The reusable `recorder-coverage.yml` workflow combines the actual Linux and
Windows recorder branch databases. Both artifacts and branch measurements are
required; a missing platform database fails the check. The combined database
must preserve both native inputs under canonical source paths. The Windows
input must contribute recorder lines and branches, including at least one
executed Windows-only statement. The production scope is
`tools/facet_receipt.py`, `tools/facet_receipt_evidence.py` and
`tools/facet_receipt_runtime.py`. Its named coverage floor is paired with an
exact set of eligible Windows-only gaps, so an uncovered portable statement or
branch fails even when the aggregate percentage clears the floor. The process
observer has a separate diagnostic report. These artifacts remain separate from
the package's existing 100% coverage union, whose required check also requires
the recorder coverage job to succeed.

## Statuses

For every registered class and facet the verifier reports one status:

| Status | Meaning |
|---|---|
| `not-declared` | the descriptor does not claim the facet |
| `declared` | claimed; the evidence field names nothing that can be located (prose, inline configuration, or no field) |
| `unavailable` | claimed; at least one named file or test node does not exist |
| `located` | every named file and test node exists; no receipt records a run |
| `bound` | the newest receipt is creditable and every subject digest still matches |
| `stale` | the newest receipt was creditable, but a subject changed or vanished |
| `invalid` | the newest receipt cannot credit the facet |

Verified tiers climb only over `bound` facets that the descriptor also
declares, one rung at a time. The corpus test
`tests/test_readiness_verification.py` fails on any `unavailable` facet, so a
pointer to a renamed or never-written test cannot stay in a descriptor.

## Generated ledger

`tools/readiness_evidence_ledger.py --write` renders
`docs/_generated/readiness_evidence_ledger.json`: the facet definitions, the
invalidation matrix, the status vocabulary, a summary partition of the
catalogue and, per model, the declared and verified tiers with every facet's
status, parsed evidence references, newest receipt and changed subjects. It
carries no timestamps or commit hashes and is kept current by
`tests/test_readiness_evidence_ledger.py`. `--summary` prints the partition;
`--check` fails when the tracked file is stale.

Studio exposes the same data: every catalogue entry carries
`verified_science_label` and `verified_silicon_label` next to the declared
labels, the model detail carries a `readiness.verified` block with the
per-facet statuses, and the facet summary counts verified tiers.

## Profiles

A receipt is keyed by `(class, facet, profile)`, where the profile is a schema
stem. What a profile *is* — the authored scientific model, the numerical
realisation that advances it and what the RTL emitter can lower from it — is
the [model profile contract](model_profiles.md). The generated
`docs/_generated/model_profile_ledger.json` lists, per profile, the validators
declared for every facet, whether they resolve, whether a receipt binds them
for that profile, and whether the profile is admitted for evidence at all.

## What this layer does not do

It does not run evidence on its own, does not decide whether a test is the
right oracle for its model (that is the per-identity source audit), does not
package receipts into the wheel, and does not change the declared tiers. A
`located` facet is a claim whose evidence exists; only `bound` is a claim whose
evidence was executed against the current subjects.
