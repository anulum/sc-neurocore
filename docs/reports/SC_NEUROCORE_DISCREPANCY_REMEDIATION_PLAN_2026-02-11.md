<!-- © 1998–2026 Miroslav Šotek. All rights reserved. -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- ORCID: https://orcid.org/0009-0009-3560-0851 -->
<!-- License: GNU AFFERO GENERAL PUBLIC LICENSE v3 -->
<!-- Commercial Licensing: Available -->

# SC-NeuroCore Discrepancy Remediation Plan

**Date:** 2026-02-11
**Scope:** Entire SC-NeuroCore benchmarking, documentation, and runtime path policy
**Goal:** Remove ambiguity, recover small-batch performance, and publish hardware-relevant metrics.

## 1. Problems and Risks (No Sugar-Coating)

1. **Single-sample regression**
`dense_fused` at ~0.380 ms is slower than `forward_fast` at ~0.171 ms for one sample.
Impact: latency-critical usage (interactive or control loops) regresses.

2. **Benchmark narrative is incomplete**
The historical `512x`-class speedup evidence is real, but it must stay attached
to the measured workload, baseline, and artefact. `docs/benchmarks/BENCHMARK_REPORT.md`
records `512.4x` for the `LIF multi (100x100K)` row against the SC-NeuroCore v2
Python reference path; `benchmarks/results/bench_adc_to_spike_kernel.json`
records later polyglot ADC-to-spike rows at `525.51x` (Rust vs Python) and
`653.28x` (Mojo vs Python).
Impact: numbers look strong but remain challengeable when detached from their
exact benchmark artefacts.

3. **Hardware target evidence is missing**
Most published numbers are CPU timings.
Impact: no clock-cycle evidence for FPGA/RTL deployment decisions.

4. **Onboarding communication gap**
Users are not clearly told when to use `forward_fast` vs `forward_batch_numpy`.
Impact: wrong API path chosen, user-visible slowdown.

## 2. Success Criteria

1. Publish a benchmark matrix that includes internal and external references with reproducible commands.
2. Recover or exceed Phase 11 single-sample latency for the default "fast path" API.
3. Publish Verilator/Icarus cycle-level metrics for dense and LIF kernels.
4. Update user docs so path selection is explicit and testable.
5. Gate release on benchmark and documentation checks in CI.

## 3. Workstreams

### Workstream A: Runtime Path Policy (Latency Recovery)

1. Add an explicit runtime selector:
`forward_auto(input_or_batch, policy="latency|throughput|deterministic")`.
2. Default policy:
`latency` for batch sizes `<= 4`, `throughput` for `>= 10`, calibration zone in between.
3. Add calibration utility:
`python scripts/calibrate_dense_threshold.py` to detect hardware-specific threshold.
4. Add tests:
`test_dense_auto_prefers_fast_for_single_sample` and `test_dense_auto_prefers_batch_for_large_batches`.

**Exit condition:** single-sample route never dispatches to slower fused/batch path by default.

### Workstream B: Benchmark Baseline Integrity

1. Define comparison sets:
- SC-NeuroCore v2 pure Python path.
- SC-NeuroCore v2 NumPy vectorized path.
- SC-NeuroCore v3 forward/fast/fused/batch.
- Optional external libraries (Norse/Sinabs/Lava CPU) with pinned versions.
2. Use one canonical harness:
`python examples/03_benchmark_report.py --matrix --json`.
3. Record machine fingerprint:
CPU model, SIMD tier, OS, Python version, Rust version, thread count.
4. Export artifacts:
`benchmarks/results/*.json` and markdown report generated from JSON only.

**Exit condition:** every speedup claim states exact baseline and command.

### Workstream C: Hardware-Centric Validation

1. Run RTL/co-sim with cycle counters:
- dense layer datapath
- LIF update pipeline
2. Produce metrics:
- cycles/sample
- max throughput at target clock
- end-to-end latency in us at 100/200/300 MHz
3. Add provisional power estimate:
toggle-rate based estimate + TODO hooks for vendor timing/power reports.
4. Publish:
`docs/HARDWARE_BENCHMARK_REPORT.md`.

**Exit condition:** at least one cycle-level report exists for each core kernel.

### Workstream D: Documentation and Onboarding

1. Update `README.md` and `docs/getting-started.md` with explicit path policy:
- `forward_fast` for small batches
- `forward_batch_numpy` for larger batches
2. Add one copy-paste benchmark quickstart:
`python examples/03_benchmark_report.py`.
3. Add FAQ entries:
- "Why is fused slower on one sample?"
- "What does the `512x`-class speedup compare against, and which artefact anchors it?"
- "How do I get hardware-relevant metrics?"
4. Add a one-page index:
`docs/SC_NEUROCORE_PERFORMANCE_INDEX.md` linking all benchmark/hardware docs.

**Exit condition:** a new user can choose the right path in under 2 minutes.

## 4. Agent Parallelization Plan

1. **Agent 1 (Runtime):** auto-dispatch API, thresholds, tests.
2. **Agent 2 (Benchmark):** baseline matrix harness + JSON schema + report generator.
3. **Agent 3 (Hardware):** Verilator/Icarus cycle pipeline + report.
4. **Agent 4 (Docs):** README, getting-started, FAQ, performance index.

All agents merge only against `03_CODE/sc-neurocore` and avoid `03_CODE/SCPN-Fusion-Core`.

## 5. Proposed Milestones

1. **M1 (Day 1):** path policy + docs warning shipped.
2. **M2 (Day 2):** full baseline matrix and reproducible report artifacts.
3. **M3 (Day 3):** cycle-level hardware report published.
4. **M4 (Day 4):** CI gating for benchmark/documentation integrity.

## 6. Immediate Next Actions

1. Resolve version drift in metadata/tests (`3.6.0` alignment for the Phase stream).
2. Run Phase 8-12 test subset and benchmark smoke.
3. Publish first discrepancy status snapshot in `SESSION_LOG_*`.
