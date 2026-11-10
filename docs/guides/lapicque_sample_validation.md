<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Validate original Lapicque timing samples -->

# Validate Lapicque timing samples

Validate retained durations before using them in a comparison. The command
checks all five language lanes against their original seven durations, report
aggregates and source files. It rejects diagnostic captures and failed runs.

First [capture the complete original workload](lapicque_timing_samples.md)
on its qualified CPython 3.12 Linux profile:

```bash
python benchmarks/capture_lapicque_samples.py \
  --checkout . \
  --output-dir results/lapicque
```

Then validate both output files against the same checkout:

```bash
python benchmarks/validate_lapicque_samples.py \
  --checkout . \
  --samples results/lapicque/original-seven-duration-samples.json \
  --report results/lapicque/original-harness-report.json
```

Exit status zero means sample integrity was verified. The JSON output contains
the input hashes and one row per language, with the sample count, median,
minimum and speedup relative to Python. Exit status two means the evidence was
rejected; the reason is written to standard error. Validation reads the inputs
without changing them.

The checks require:

- `performance_admissible: true`, successful raw retention and a successful
  original harness invocation;
- diagnostic mode disabled and no instrumentation observed at the documented
  installation, return and completion boundaries;
- the exact capture source and original benchmark version, with every source
  declared by that benchmark still matching the report;
- an unchanged original report and the original single-CPU, 100,000-step,
  seven-repetition workload;
- all five executed backends, seven positive finite durations per backend,
  matching medians, minima and speedups, and the original event and voltage
  parity requirements.

A valid diagnostic capture still fails this command. Its durations remain
available for investigating the capture mechanism, but cannot enter a
performance comparison.

Sample integrity is one part of benchmark acceptance. Keep the executing
hardware, operating system, runtime and binary provenance with the run, and
verify physical CPU isolation and overlapping workloads separately. The
validation result therefore retains `production_performance_claim: false` and
`external_isolation_and_runtime_provenance_required: true`.

The dedicated consumer tests use those real capture outputs as their input:

```bash
python -m pytest benchmarks/tests/test_validate_lapicque_samples.py \
  --lapicque-checkout . \
  --lapicque-samples results/lapicque/original-seven-duration-samples.json \
  --lapicque-report results/lapicque/original-harness-report.json
```

They check acceptance through the CLI and public Python API, preserve the
original files, and reject altered copies of eligibility fields, durations,
report bytes, aggregates and parity results.
