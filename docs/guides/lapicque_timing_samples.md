<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Retain the frozen Lapicque harness's actual seven timing samples -->

# Retain Lapicque timing samples

The replay command runs the existing Lapicque benchmark and preserves its seven
individual public-call durations for each of Python, Rust, Julia, Go and Mojo.
It keeps the original 100,000-step workload, 20-step warmup, backend order,
clock reads and numerical parity checks. It does not replace the measured
functions or modify the original report.

Use a Linux checkout with CPython 3.12 and all five native backends
built and available. Use the same runtime and dependency environment for the
reference benchmark and the replay command. The command verifies the exact
benchmark source it supports and refuses source drift. This replay profile
admits only CPython 3.12. A different minor version needs its own returning-frame
and local-monitoring qualification before it is admitted; passing static checks
does not qualify another runtime.

Run inside the single-CPU allocation supplied by your benchmark supervisor:

```bash
python benchmarks/capture_lapicque_samples.py \
  --checkout . \
  --output-dir lapicque-samples
```

The destination must not already exist. A second invocation needs a new
output directory; earlier results are never replaced. The original harness
refuses an unpinned CPU allocation and missing backends. The replay command
supplies no override for either condition.

The output directory contains:

- `original-harness-report.json`: the original report, including numerical
  parity, event equality, source hashes, workload, CPU affinity, host load and
  runtime versions.
- `original-seven-duration-samples.json`: all 35 measured durations, their
  backend identities, medians and minima, interpreter details, the harness and
  replay-tool digests, the original exit code and capture-integrity findings.

The replay tool enables only the local return event of the original measurement
function. Its five callbacks run after each backend has completed all seven
clock reads. Each callback copies the existing duration list and checks its
median and minimum against the function's original returned values. After the
benchmark finishes, the tool checks those aggregates against the original
report and releases its own monitoring slot. It clears global and local events
for its own slot before releasing it. If another tool takes ownership, that
foreign slot is retained and the capture is rejected.

`performance_admissible` is true only for default-mode captures with complete
raw retention and no instrumentation observed at the recorded boundaries.
A comparison consumer must reject a capture when this field is false. A true
value passes the replay tool's local checks and still requires the external
isolation and provenance evidence described below.

A zero exit code and `raw_retention_valid: true` establish successful retention
of the complete original measurements. They do not establish exclusive CPU
access or stable thermals. A performance comparison also needs an externally
verified physical-core allocation, an idle SMT sibling, recorded NUMA and
thread settings, frequency and temperature observations, absence of overlapping
compute, and exact implementation, binary and dependency identities. Keep those
supervisor records with the original outputs. Report sample counts and units;
seven measurements do not support a meaningful p99 claim.

## Investigate a traced run

The default mode refuses existing tracing, profiling or Python monitoring. It
also refuses a run when foreign instrumentation is present at a capture event
or completion. This prevents a traced execution from being silently treated as
the normal measurement procedure.

For debugging under a profiler or coverage collector, pass `--diagnostic`:

```bash
python benchmarks/capture_lapicque_samples.py \
  --checkout . \
  --output-dir lapicque-diagnostic-samples \
  --diagnostic
```

Diagnostic mode still executes the full original workload and all five
backends. It records `diagnostic_mode: true` and the observed instrumentation.
Instrumentation records describe setup, each measurement return and completion.
It always sets `performance_admissible: false`. The replay uses CPython's
OPTIMIZER_ID (slot 5), leaving the debugger, coverage and profiler slots free.
If an optimizer or JIT already owns slot 5, both modes refuse to displace it.
Its success confirms complete raw retention and numerical checks; its timings
are excluded from performance comparisons. It cannot use another owner's
monitoring slot, change workload sizes, permit a missing backend or weaken
parity. Every output retains `production_performance_claim: false`; publishing
an isolated comparison requires the additional supervisor evidence and review.

Malformed or partial original reports are recorded as capture errors. The tool
still writes the raw-retention record and then propagates any original harness
exception. Only an exception raised by this invocation is recorded; a wrapper's
already handled outer exception is not attributed to the benchmark.
If the filesystem itself prevents that record from being written,
the original exception remains the primary failure and records the retention
error as an additional note.

If the original command fails, samples are missing, a backend repeats,
aggregates disagree, or monitoring ownership changes, the command does not
accept the capture. Keep the true exit status and existing output files when
investigating a failed run. A missing sidecar after forced termination supplies
no measurement credit.

## Validate the return-record contract

The live observer forwards each genuine returning frame to `TimingFrame.capture`
and `record_return`. `TimingReturns` keeps the checked rows and any rejected
contracts. The public recording API also supports diagnostic replay of the real
observations; replay alone never establishes benchmark eligibility or isolation.

Run its dedicated tests with all five original backends available:

```bash
python -m pytest benchmarks/tests/test_lapicque_return_records.py
```

The tests execute the complete original workload once, retain the actual
interpreter frames and returned packets, then compare replay with the original
sidecar. Negative cases explicitly damage copies of those observations; they
never substitute successful timings or change the measured computation.
The separate CLI tests exercise the live monitoring callback and ownership.
