<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Run the complete Lapicque capture and validation checks -->

# Verify Lapicque capture and validation

The capture profile exercises the original five-language benchmark through the
[capture command](lapicque_timing_samples.md) and the
[sample validator](lapicque_sample_validation.md). It requires Linux, CPython
3.12, all five built backends and a separately installed unsupported interpreter
for the actual runtime-refusal test. The refusal is traced with coverage inside
that interpreter, so it needs the same hash-locked coverage as the primary one.
The CI profile selects CPython 3.12.16, uses CPython 3.11.18 for that refusal,
and installs the shard's test tool pins from `requirements/ci-dev.txt` into it.

From the checkout, supply a new output directory and the existing interpreter:

```bash
bash benchmarks/verify_lapicque_capture.sh \
  --checkout . \
  --output-dir results/lapicque-capture-checks \
  --unsupported-python /absolute/path/to/python3.11
```

The parent of the output directory must exist. Select the primary interpreter
explicitly with `--python /absolute/path/to/python3.12` when needed. Existing
output directories are refused, preserving earlier evidence.

The profile runs these steps sequentially:

1. Execute all nine normal CLI contracts without coverage instrumentation.
2. Run the original capture command on one permitted CPU and retain its complete
   report and all 35 durations as the consumer's genuine input.
3. Execute all 57 diagnostic, monitoring-ownership, post-computation integrity,
   return-record and admission cases with branch and subprocess tracing.
4. Combine the original coverage components, keep them, and require 100% of the
   capture module's statements and branches with no excluded lines.
5. Run all 44 consumer contracts against the new normal-mode report and samples,
   then require the same complete coverage of the consumer module.
6. Verify the three JUnit reports contain the expected executed cases and no
   failures, errors or skipped tests.

The output includes `normal-junit.xml`, `diagnostic-junit.xml`,
`consumer-junit.xml`, `capture-coverage.json`, `consumer-coverage.json`, raw
parallel coverage files, the normal consumer seed and the individual test
outputs. Each failure stops the profile and preserves the files already written.
The two coverage configurations are `benchmarks/lapicque_capture.coveragerc` and
`benchmarks/lapicque_consumer.coveragerc`.

These checks establish runtime behaviour, capture integrity and consumer
acceptance. The runner's single-CPU affinity does not establish exclusive use of
the physical core. Diagnostic durations remain excluded from performance
comparisons; normal captures additionally require the external isolation,
hardware, thermal and runtime evidence described in the capture guide.
