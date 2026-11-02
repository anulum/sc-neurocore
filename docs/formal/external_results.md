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

Full formal qualification remains incomplete for 26 resource-intensive jobs.
The `formal-all` gate is expected to fail until their results and required
assertion-enable evidence qualify. Core 4.1.0 remains blocked until this gate is honestly green.
The indexed Amari primary PASS currently lacks its required non-vacuity
artifact and review, so the validator refuses it. Timeouts, memory limits and
solver execution errors do not count as proofs.

| Workflow status | Location | Meaning |
|---|---|---|
| `PASS` | `hosted` | The current hosted invocation passed. |
| `ATTESTED` | `ML350, attested` | An actual native PASS has matching source, tool, measurement and independent review records. |
| `ERROR` | `external, refused` | Indexed evidence is missing, stale, unsuccessful or inconsistent. |

Indexed jobs have no hosted fallback. A refused external result fails its shard.
Hosted jobs retain their 12 GiB memory cap, zero swap and 1,800-second runtime
limit. Their primary proof and assertion-enable audit share that single
budget. The qualification report retains the primary verdict separately;
an incomplete audit never turns a native PASS into a qualified PASS.
External ML350 measurements use a 64 GiB cap, zero swap and a
7,200-second runtime limit. Neither resource limit changes the declared SBY
mode, depth, engine, assumptions or properties.

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

A qualifying receipt requires all task verdicts and final DONE markers to be
PASS, a successful native exit, finite measurements within the declared limits,
zero OOM events, and an inactive service with no remaining processes. Retained
systemd execution accounting supplies the exit status and monotonic timestamps.
The attestation must identify a distinct reviewer and bind the exact receipt,
source hashes, log hashes, normalizer source and any execution exception.


## Solver placement

Explicit `smtbmc z3` jobs use the SDK's pinned Z3 executable. Unqualified
`smtbmc` entries use SBY's default Yices engine and the SDK's pinned Yices 2.7.0.
The workflow prints solver versions before running the shard. No apt Z3 or
Yices executable substitutes for these SDK binaries. The SDK workflow builds
Yices 2.7.0 from commit `85cf17e44eac76b5d14b297c09fc9bfecf47ef65` with
`-O2 -march=x86-64 -mtune=generic`. Its source archive is hash-pinned; GMP and
gperf packages are pinned against Ubuntu's signed archive index. The build
scans Yices' own objects for unsupported instruction families and checks an
actual SAT/UNSAT result. GMP's CPUID-dispatched routines are excluded from the
object scan. Restoring the SDK verifies its binary, recipe and scanner hashes
and repeats the solver smoke. Cross-host byte identity comes from distributing
one artifact; independent-build reproducibility has not been established.
sv2v retains its pinned GitHub release-asset hash without a signed archive-index
chain.

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
