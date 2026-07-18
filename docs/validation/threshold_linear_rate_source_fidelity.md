# Threshold-linear rate source and polyglot fidelity evidence

This page records the scientific scope, algebraic equation, five-runtime
parity, failure atomicity, and controlled benchmark used to promote
`ThresholdLinearRateNeuron` to the polyglot-complete catalogue.

## Scientific scope

Gerstner, Kistler, Naud, and Paninski (2014),
[doi:10.1017/CBO9781107447615](https://doi.org/10.1017/CBO9781107447615),
define the piecewise-linear gain
\(F(h)=[h]_+=\max(0,h)\) in
[Section 18.2, Eq. 18.23](https://neuronaldynamics.epfl.ch/online/Ch18.S2.html).
The maintained SC-NeuroCore transfer exposes an explicit translation and
scale:

\[
r = g\max(0,I-\theta).
\]

The citation supports the underlying population-rate gain function. It does
not turn this algebraic transfer into a spiking neuron, a fitted single-cell
model, or a temporal population ODE.

## Algebraic contract

Every non-empty evaluation overwrites the cached output with the transfer for
the current input. The previous `r` is validated before execution but does not
enter the right-hand side. An empty batch preserves it.

All runtimes enforce finite `r`, `theta`, `gain`, and current, with non-negative
`r` and `gain`. They compute the complete candidate before visible mutation and
reject non-finite output. Reset clears `r` while preserving `theta` and `gain`.

## Executable parity matrix

| Runtime | Executed surface | Enrolled result |
|---|---|---|
| Python | public scalar and atomic batch | reference |
| Rust engine | PyO3 modular batch, zero crate-root delta | bit-exact |
| Rust safety | independently compiled module | bit-exact; 6/6 tests pass |
| Julia | `simulate_trace` through JuliaCall | bit-exact |
| Go | service plus generated C-shared ABI | bit-exact |
| Mojo | exported shared-library C ABI | bit-exact |

The configured parity case is `r=0.25`, `theta=1.5`, `gain=2`, and `I=3`.
Each of the 200,000 output values is exactly `3.0`; the canonical little-endian
float64 trace SHA-256 in every runtime is
`cdb90f105692311ba359cfbf0574faa23586215e1a253ddcad29276b9bf69402`.

The focused cohort also executes below-threshold, threshold-equality, and
above-threshold branches, empty batches, explicit-unavailable backends,
overflow rejection, corrupted runtime state, and invalid Go/Mojo contracts
whose caller buffers must remain unchanged.

## Generated fixed-point co-simulation

The paired TOML and JSON schemas reproduce the configured hand transfer
exactly. The production equation compiler lowers the same schema to Q16.16
Verilog with `theta=1.5` and `gain=2.0`. Icarus Verilog co-simulation drives
193 representable inputs from `-4.0` through `8.0` in `1/16` increments and
establishes cycle-exact public `r_out` words across the below-threshold,
equality, and above-threshold branches. `spike_out` remains zero throughout.

This is an H1 generated-RTL result for the declared algebraic transfer. It does
not recast positive rates as binary events.

## Controlled benchmark

`benchmarks/bench_model_threshold_linear_rate.py` measures the complete
200,000-value trace through each public dispatcher five times after warm-up.
It fails if any runtime is absent, the standalone Rust-safety binary fails, or
any trace value or final cached output differs from Python.

The run was pinned to logical CPU 4, but the CPU was not exclusively isolated
and host load was high. The artifact records every raw sample, exact source and
native-binary hashes, tool versions, affinity, and load. These are local
regression timings, not production, cross-host, hardware, or universal-ranking
claims.

| Backend | Median call | Median ns/evaluation | Trace mismatches |
|---|---:|---:|---:|
| Python | 1.621 ms | 8.107 | 0 |
| Mojo | 3.388 ms | 16.938 | 0 |
| Rust | 3.892 ms | 19.458 | 0 |
| Go | 9.824 ms | 49.122 | 0 |
| Julia | 12.425 ms | 62.123 | 0 |

Python uses a vectorised fill for this constant-input algebraic workload and
was the shortest raw call. The compiled-backend dispatcher policy keeps Python
as its always-available floor; among native lanes the measured order is Mojo,
Rust, Go, then Julia.

The dedicated
`threshold-linear-rate-five-backend-local-regression` evidence gate adds no
failure. The aggregate repository report still records 66 older source-hash
mismatches, including prior gates whose shared bridge/engine sources changed;
those inherited refreshes are not presented as Model35 failures.

## Boundaries

- Positive output is continuous rate, not a binary event or spike count.
- `r` is a cached observable, not integrated state or memory.
- The book equation supports the rectified gain; the explicit threshold and
  gain are the declared translated/scaled form.
- The benchmark is local non-exclusive regression evidence.
- The generated Q16.16 claim is cycle-exact co-simulation only. No formal
  equivalence, synthesis, timing, device, or PPA result is claimed.

## Reproduction

```bash
go test ./services -run ThresholdLinearRate -count=1
rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/threshold_linear_rate.rs \
  -o /tmp/threshold_linear_rate_tests
/tmp/threshold_linear_rate_tests

PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_threshold_linear_rate.py \
  tests/test_cosim_threshold_linear_rate.py \
  tests/test_threshold_linear_rate_backend_loading.py \
  tests/test_threshold_linear_rate_backends.py \
  tests/test_bench_threshold_linear_rate.py

taskset -c 4 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_model_threshold_linear_rate.py \
  --json benchmarks/results/local_python_2026-07-14_threshold_linear_rate.json
```
