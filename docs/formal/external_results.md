<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — External formal evidence -->
# External formal evidence

The formal workflow distinguishes native hosted proofs from independently
attested external results. CI validates the external evidence and its bindings;
it does not reproduce that external execution.

Full formal qualification remains incomplete for 23 resource-intensive jobs.
The `formal-all` gate is expected to fail until their results and required
assertion-enable evidence qualify. Full formal qualification is not claimed until this gate
is honestly green.
The indexed Amari primary PASS has a separate, independently reviewed
assertion-enable audit over its retained prepared model. All six enables are
reached at step 2 within the original prove depth 12. Timeouts, memory limits
and solver execution errors do not count as proofs.

| Workflow status | Location | Meaning |
|---|---|---|
| `PASS` | `hosted` | The current hosted invocation passed. |
| `ATTESTED` | `ML350, attested` | An actual native PASS has matching source, tool, measurement and independent review records. |
| `UNPROVEN-RESOURCE` | `ML350, attested` | Both admitted solvers exhausted their resources; exact attempts, reached assertion enables and bounded scope have an independent review. No successful proof is claimed. |
| `ERROR` | `external, refused` | Indexed evidence is missing, stale, unsuccessful or inconsistent. |

Indexed jobs have no hosted fallback. A refused external result fails its shard.
Hosted jobs retain their 12 GiB memory cap, zero swap and 1,800-second runtime
limit. Their primary proof and assertion-enable audit share that single
budget. The qualification report retains the primary verdict separately;
an incomplete audit never turns a native PASS into a qualified PASS.
External ML350 primaries use Z3 with a 40 GiB cap, zero swap and a
7,200-second runtime limit. A primary memory stop permits one Z3 retry at
64 GiB. A Z3 timeout, or a resource stop on that memory retry, permits one
Yices attempt at 64 GiB with the same runtime ceiling and zero swap.
No resource retry changes the declared mode, depth, assumptions or properties;
the Yices configuration changes only the SMTBMC solver token.

## Retained external results

| Model | Scope | Native proof | Separate enable audit |
|---|---|---|---|
| Amari field | prove, depth 12 | PASS | Six enables reached at step 2. |
| Chaotic map | prove, depth 12 | Yices PASS after Z3 MEMORY40 and TIMEOUT64 | Four enables reached at steps 1, 3, 3, 3. |
| Chialvo map | bmc, depth 4 | Yices PASS under the exact solver-crash repair contract | Four enables reached at steps 0, 0, 0, 3. |

The Chialvo result is bounded BMC evidence. The Chaotic result includes
successful temporal induction. Each audit uses the same retained prepared
model as its primary. These checks establish their stated obligations and
enable reachability, without claiming property strength or broader model
correctness. All earlier incomplete or failed executions remain retained.

## Validate a result

Use the verified SDK on `PATH` and its matching provenance file:

```bash
python -m tools.formal_external_evidence \
  hdl/formal/catalogue/sc_amari_field.sby \
  --provenance "$HDL_SDK/PROVENANCE.txt"
```

The index is `benchmarks/results/formal/index.json`. Each entry binds a receipt
and an independent attestation by SHA-256. The receipt binds the current SBY
configuration and complete source-file closure, resolved by the native SBY
parser. `--dumptaskinfo` also checks that the recorded solver matches every
actual task engine. The shared SDK pins Z3 4.8.12 to the same executable bytes
used by the external measurements; package, executable and upstream copyright
checks apply when building and restoring the SDK.

A qualifying PASS receipt requires all task verdicts and final DONE markers to be
PASS, a successful native exit, finite measurements within the declared limits,
zero OOM events, and an inactive service with no remaining processes. Retained
systemd execution accounting supplies the exit status and monotonic timestamps.
The attestation must identify a distinct reviewer and bind the exact receipt,
source hashes, log hashes, normalizer source and any execution exception.

## Explicit resource contracts

Each external index entry and its independent review name the same
`resource_contract`. The validator selects the host, solver, memory and time
limits from that registered contract; observed memory use never selects it.

| Contract | Solver | Cap | Admission |
|---|---|---|---|
| `z3-primary-40-v1` | Z3 | 40 GiB | First native attempt. |
| `z3-memory-retry-64-v1` | Z3 | 64 GiB | One closed 40 GiB primary with actual kernel OOM accounting. |
| `yices-resource-retry-64-v1` | Yices | 64 GiB | Closed Z3 timeout, or closed resource stop on the admitted 64 GiB Z3 retry. |
| `amari-historical-64-v1` | Z3 | 64 GiB | Only the exact historical Amari receipt identified below. |
| `chialvo-solver-repair-yices-64-v1` | Yices | 64 GiB | Only the exact reviewed Chialvo Z3 crash, diagnosis and solver-only variant. |

Retries bind the complete ordered `predecessors` chain. Receipts retain
distinct invocation identities and actual monotonic start/stop times, closed
process trees, zero swap, actual caps and native exit facts. A MEMORY result
requires retained kernel OOM and kill counts; a TIMEOUT requires its real
runtime ceiling and zero OOM counts. A solver crash, counterexample, ERROR,
UNKNOWN, missing accounting or changed proof input refuses the row. A tool
repair after an ERROR requires a separate reviewed execution contract.

The Chialvo repair contract pins the original ERROR receipt and crash
diagnosis, the exact configuration variant and portable Yices binary. Its
independent PASS review binds that lineage; the new invocation must follow
the original crash and have its own successful proof and enable audit. It
does not admit another model, another ERROR, a changed depth, or an
`UNPROVEN-RESOURCE` result. The original crash remains an ERROR.

An index entry may explicitly declare `outcome: UNPROVEN-RESOURCE` only when
its complete `attempts` chain ends with an actual incomplete Yices attempt.
Every attempt has its own receipt and independent review. The row also names
`nonvacuity_parent_sha256`: the exact attempt whose retained prepared model
was audited separately. Missing or unreachable assertion enables still refuse
the row. No primary proof is rerun merely to obtain this audit.

The row review binds all ordered attempt receipt/review hashes, the audit and
its independent review, the exact parent, and every task's mode and depth.
`release_notes` binds a public file by SHA-256. One explicit line in that file
must name the job, `UNPROVEN-RESOURCE`, every receipt hash, the audit hash and
each `task: mode=bmc; depth=…` scope (using the actual task mode and depth).
The review and release notes cannot silently describe an unbounded proof.

The workflow accepts a fully qualified resource-incomplete row separately
from PASS, displays its status and prominent count, and excludes it from the
native-proof total. A green evidence gate can therefore include explicitly
reviewed incomplete rows. Their scientific proof obligations remain open.


## Solver placement

Explicit `smtbmc z3` jobs use the SDK's pinned Z3 executable. Unqualified
`smtbmc` entries use SBY's default Yices engine and the SDK's pinned Yices 2.7.0.
The workflow prints solver versions before running the shard. No apt Z3 or
Yices executable substitutes for these SDK binaries. Every run restores one
published SDK archive, the `scn-hdl.tar.gz` asset of this repository's release
`hdl-sdk-yosys070-yices270-524357fd`, after checking its SHA-256, and checks
its Yices and Z3 binaries against the digests the receipts record. That
archive's Yices 2.7.0 was built from commit
`85cf17e44eac76b5d14b297c09fc9bfecf47ef65` with
`-O2 -march=x86-64 -mtune=generic`. Its source archive is hash-pinned; GMP and
gperf packages are pinned against Ubuntu's signed archive index. The build
scanned Yices' own objects for unsupported instruction families and checked an
actual SAT/UNSAT result. GMP's CPUID-dispatched routines are excluded from the
object scan. Each run verifies the binary, recipe and scanner hashes and
repeats the solver smoke. The release also carries the Yices and GMP sources
the binary was built from. Cross-host byte identity comes from distributing
this one archive: a build from the same pinned sources on a later runner image
produced a different Yices binary, so independent-build reproducibility has
not been established. A source build runs only when a caller sets
`rebuild_sdk`, for a deliberate SDK update that publishes a new archive and
records the receipts again. sv2v retains its pinned GitHub release-asset hash
without a signed archive-index chain.

The following hosted jobs explicitly select `cvc5`. They use the runner's
installed cvc5, whose version is recorded in the workflow log. These are native
hosted results, and are not external Z3 attestations:

```text
hdl/formal/catalogue/energy_lif.sby
hdl/formal/catalogue/sc_brunel_wang.sby
hdl/formal/catalogue/sc_compte_wm.sby
hdl/formal/catalogue/sc_compte_wm_ring16.sby
hdl/formal/catalogue/sc_dpineuron.sby
hdl/formal/catalogue/sc_integerqifneuron.sby
hdl/formal/catalogue/sc_mat.sby
hdl/formal/catalogue/sc_nmda_autapse.sby
hdl/formal/catalogue/sc_non_resetting_adaptive_lif.sby
hdl/formal/catalogue/sc_non_resetting_lif.sby
hdl/formal/catalogue/sc_normalized_energy_lif.sby
hdl/formal/catalogue/sc_resetting_mat.sby
hdl/formal/catalogue/sc_sigma_delta.sby
hdl/formal/catalogue/sc_sigma_delta_accumulator.sby
hdl/formal/catalogue/sc_terman_wang.sby
hdl/formal/catalogue/sc_wb_nmda_magnesium_block.sby
hdl/formal/sc_aer_priority_queue.sby
hdl/formal/sc_dcls_layer_core.sby
hdl/formal/sensors/adc_to_spike_quantiser.sby
hdl/formal/timing/example_cdc_two_flop_synchroniser.sby
hdl/formal/timing/example_dense_layer_core_latency.sby
```

The SDK Z3 package is Ubuntu Noble `4.8.12-3.1build1` for amd64. Its runtime
requires libc6 >= 2.38, libgcc-s1 >= 3.4 and libstdc++6 >= 13.1. Ubuntu 24.04
runners satisfy this contract; the SDK verifies these versions and executes the
pinned binary before packaging it. Its authenticated archive-index and package
hashes are retained in SDK provenance.

## Log custody

`tools.formal_log_normalization` produces a public technical log while the
original raw bytes remain in private custody. Its versioned identity
placeholders replace only explicitly reviewed workspace, unit, hostname and
username literals. Every other byte, line ending, line order, timestamp,
solver message, status and numeric value remains unchanged. Tests use an actual
native proof log and verify exact inverse reconstruction of its original hash.

The receipt names **two different hashes**: `raw_sha256` identifies the retained
raw log, and `normalized_sha256` identifies the public file. The public log is
never represented as raw evidence. The reviewer independently reruns the
normalizer against the original log and attests both hashes and the exact
normalizer version and source hash.

## Exact historical accounting exception

`benchmarks/results/formal/exceptions.json` admits only explicitly reviewed
receipt hashes. Changing any receipt byte removes its exception. An exception
does not apply to another invocation or another receipt for the same job.

The `sc_amari_field` receipt records a native PASS, a true cgroup peak of
14,327,123,968 bytes, and a journal-recovered interval of 3,382.523562 seconds.
Its successful transient service was garbage-collected before the old observer
retained its exit accounting. The receipt therefore preserves `unit_exit_code`
and `unit_result` as null, identifies the native exit through the actual final
SBY DONE line and status file, and leaves the original `wall_seconds` null.
The explicit recovered field is checked against the journal monotonic interval.
Its independent attestation names the exact exception and registry hash.
Future measurements retain the service until execution accounting is captured.

An attested result is evidence for its stated formal properties and source
scope. It is not a hardware performance measurement or a broader claim about
model correctness.

## Assertion-enable non-vacuity

`python -m tools.formal_qualified_run JOB --work-root WORK --report REPORT`
runs the unchanged native primary tasks and audits their actual prepared
models. Each auxiliary model comes from that task's `model/design_prep.il`;
the report records its SHA-256. Native Yosys flattens and serializes this
model. Every `$assert` becomes one `$cover` with constant-true A and the
original EN. Original assumptions, state, initializers and enable logic
remain unchanged. Pre-existing covers are removed from this auxiliary.

Each enable cover must be reached within the primary task's own depth.
The report maps every original assertion to its first reached step and binds
the auxiliary graph, configuration, log and implementation hashes. Zero
assertions, missing models, disabled assertions and incomplete auxiliary
runs remain unqualified. A native cover task requires actual nonzero original
cover goals and its own successful native completion.

Mixed reachable and unreachable assertions remain unqualified; the report
lists the missing assertion enables separately from the solver's status.
Unsupported `$check`, `$live`, `$fair` and `$print` cells are refused rather
than leaving a possible check unaudited. The primary PASS is persisted before
the auxiliary starts, so an outer timeout can retain the completed primary
verdict while the audit remains incomplete.

This checks **assertion-enable non-vacuity**, not property strength. A trivial
true property can have a reachable enable. Killed RTL mutants provide separate
strength evidence. External receipts also require separate independently
reviewed non-vacuity evidence; the primary receipt itself is not rewritten.

An external index entry supplies `nonvacuity` and `nonvacuity_sha256`, plus
`nonvacuity_review` and `nonvacuity_review_sha256`. The separate artifact binds
the unchanged receipt, source closure, tools, every native task's mode and
depth, prepared-model custody and per-assertion witness steps. Normalized audit
logs must preserve the successful DONE marker and each reported enable witness.
The independent review binds the exact artifact, receipt, sources and prepared
model hashes. Missing or unreachable enables refuse the whole external result.

When a native empty `cgroup.procs` observation is converted from an empty
string to the public empty-list schema, `receipt_conversion` in both the
audit artifact and independent review records the original and converted
receipt digests, field and values. The original native audit-parent receipt
remains retained; this explicitly reviewed representation change does not
change its execution, model, source or measured outcome.

External audits have their own service invocation, zero swap, an 8 GiB cap and
a 1,800-second ceiling. They retain native exit and monotonic accounting; the
primary's historical accounting exception cannot cover the audit. A retained
original model must match the audited model. A regenerated model is explicitly
labelled `regenerated, original not retained`, with independent review; it is
never represented as the primary's retained original.

`tools.formal_nonvacuity.check_prepared_assertions` audits retained models from
incomplete bmc/prove tasks without rerunning their primary proof. The caller
must separately validate the primary outcome and model custody. Its
`NONVACUOUS` result describes assertion enables only; it cannot qualify the
primary as PASS. Original cover tasks still require a successful primary.
The ordinary `check_task` entry point retains its primary-PASS requirement.

Changing `formal_nonvacuity.py` or `formal_assertion_enables.py` invalidates
every existing audit artifact, even if the primary proof sources are unchanged.
The current implementation and transform hashes must match; run the audit again
before relying on that evidence.

## Public execution identity conversion

Public receipts use separate descriptive labels for primary and assertion-enable audit execution. A versioned conversion manifest records the original and public artifact hashes, the hashes of changed identity fields and a digest of every preserved field. Private coordination lease identities remain in the immutable native custody records; public receipts retain the complete release-receipt hash. Native systemd invocation identities and all model, tool, clock and measured outcome fields remain unchanged. The index and independent reviews bind the conversion manifest as well as the converted evidence. Historical execution exceptions remain limited to an exact receipt representation.
