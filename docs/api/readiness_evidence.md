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
| `backend:python` … `backend:mojo` | software | none | descriptor contract, model module, validator | none |
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

## Facet receipts

A receipt records one execution of the evidence command:

```
python tools/facet_receipt.py record --model LapicqueNeuron --facet cosim \
    -- python -m pytest "tests/test_cosim_lapicque.py::test_source_q3232_preserves_first_attainment_and_polarization_bound"
```

The recorder derives the facet's subjects from the identity registry, runs the
command (a pytest command runs with a JUnit report so the passed, failed and
skipped counts come from the run itself), records the tool and runtime
versions, the git head and any uncommitted subject, seals the payload with its
own SHA-256 and writes it as a new file under
`src/sc_neurocore/neurons/facet_receipts/`. Receipts are append-only: the
recorder refuses to overwrite, and the verifier reads the newest receipt per
(class, facet). A receipt credits its facet only when it is sealed, names the
class it is read for, carries every required subject kind, ended with
`outcome = "passed"`, exit code 0, at least one passed check and no failed,
errored or skipped check, and states the claim scope the facet requires.

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

## What this layer does not do

It does not run evidence on its own, does not decide whether a test is the
right oracle for its model (that is the per-identity source audit), does not
package receipts into the wheel, and does not change the declared tiers. A
`located` facet is a claim whose evidence exists; only `bound` is a claim whose
evidence was executed against the current subjects.
