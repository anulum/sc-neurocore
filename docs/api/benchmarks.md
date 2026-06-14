# NeuroBench Benchmarking

NeuroBench-compatible standardized SNN evaluation framework.

## Evidence boundary

Benchmark APIs produce structured measurements; they do not make claims by
themselves. A performance, accuracy, power, latency, or hardware-efficiency
claim is release evidence only when it points to one of these committed
artefact classes:

- raw JSON or CSV under `benchmarks/results/`;
- a benchmark report in `docs/benchmarks/` that names the command,
  environment, and source result file;
- a companion paper artefact that carries the same command and environment
  provenance.

If a number is not traceable to one of those artefacts, treat it as an
unpublished local measurement. Do not copy it into README, roadmap, release, or
paper prose as a product claim.

Module-owned pytest throughput checks are load-tolerant smoke guards by
default. They assert finite positive progress and a low non-strict floor so
functional suites can run while ORCA, synthesis, or other workstation jobs are
active. To enforce the historical strict numeric thresholds, run the affected
tests on isolated benchmark cores with:

```bash
SC_NEUROCORE_STRICT_THROUGHPUT=1 pytest tests/test_model_fitzhugh_nagumo.py tests/test_model_ai_optimized.py
```

Strict throughput output is still local benchmark evidence until the raw
artefact records CPU affinity, host load, governor, frequency, versions, and
the command that produced it.

## Metrics

::: sc_neurocore.benchmarks.metrics
    options:
      show_root_heading: true
      members:
        - compute_metrics
        - BenchmarkResult

## Tasks

::: sc_neurocore.benchmarks.tasks
    options:
      show_root_heading: true
      members:
        - TASKS
        - BenchmarkTask
