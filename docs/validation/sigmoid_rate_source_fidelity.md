# Sigmoid-rate source and polyglot fidelity evidence

This page records the scientific scope, finite-step equation, five-runtime
parity, failure atomicity, and controlled benchmark used to promote
`SigmoidRateNeuron` to the polyglot-complete catalogue.

## Scientific scope

Wilson and Cowan (1972),
[doi:10.1016/S0006-3495(72)86068-5](https://doi.org/10.1016/S0006-3495(72)86068-5),
derive coupled nonlinear differential equations for interacting excitatory and
inhibitory populations. `SigmoidRateNeuron` does not claim to reproduce that
complete system. It declares a reduced scalar motif:

\[
\tau\dot r=-r+\sigma(\beta(I-\theta)).
\]

The citation therefore supports the population-rate and sigmoid inspiration;
fidelity is evaluated against the explicitly declared scalar equation. The
coupled family remains a separate `WilsonCowanUnit` surface.

## Exact finite-step contract

With current held constant over one step, the logistic target is constant and
the scalar ODE has the exact update

\[
r_{n+1}=d r_n+(1-d)\sigma(\beta(I-\theta)),
\qquad d=e^{-\Delta t/\tau}.
\]

All five runtimes implement that update, branch-stable logistic evaluation,
finite validation, a bounded-rate postcondition, and candidate-before-mutation
failure atomicity. Reset changes only the dynamic rate.

## Executable parity matrix

| Runtime | Executed surface | Enrolled result |
|---|---|---|
| Python | public scalar and atomic batch | reference |
| Rust engine | PyO3 modular batch, zero crate-root delta | byte-identical to Python |
| Rust safety | independently compiled module | byte-identical to Python; 8/8 tests pass |
| Julia | `simulate_trace` | byte-identical to Python |
| Go | service plus generated C-shared ABI | byte-identical to Python |
| Mojo | exported shared-library C ABI | maximum absolute difference `3.08e-14` |

The configured parity case is `r=0.25`, `tau=10`, `beta=2`, `theta=1`,
`dt=0.5`, and `I=3`. The first six Python values are:

```text
0.2857007338135623
0.3196603222932904
0.3519636820991432
0.38269158845670403
0.41192087713731845
0.43972463658754457
```

`tests/test_sigmoid_rate_backends.py` also executes empty batches, a timestep
fifty times the time constant, explicit-unavailable backends, malformed native
results, and invalid Go/Mojo contracts whose output buffers must remain
unchanged.

## Controlled benchmark

`benchmarks/bench_model_sigmoid_rate.py` measures the full 200,000-step trace
through each public dispatcher, five times after warm-up. It fails if any
backend is absent, the standalone Rust-safety test binary fails, or a trace or
final rate differs by more than `5e-12`.

The run is pinned to one logical CPU but is not exclusively isolated. The
artifact records raw samples, source hashes, exact Rust/Go/Mojo binary hashes,
runtime versions, affinity, and load. It explicitly rejects production-speed,
cross-host, hardware, and universal-ranking interpretations.

| Backend | Median call | Median ns/step | Mismatches | Maximum error |
|---|---:|---:|---:|---:|
| Python | 73.788 ms | 368.938 | 0 | 0 |
| Rust | 46.985 ms | 234.926 | 0 | 0 |
| Julia | 17.493 ms | 87.467 | 0 | 0 |
| Go | 97.270 ms | 486.350 | 0 | 0 |
| Mojo | 14.610 ms | 73.048 | 0 | `3.08e-14` |

The Python/Rust/Julia/Go canonical trace SHA-256 is
`5241be414683ce92ba9886c13c0a9f5ef84886d5d48ddda05fc892b72274e07d`.
Mojo has a distinct binary trace hash because the tolerated libm-level
difference is real and disclosed.

## Boundaries

- This is continuous rate output, not a spike train. Positive rates are not
  counted as events.
- The paper citation does not turn the scalar unit into the full coupled
  Wilson-Cowan model.
- The benchmark is local regression evidence, not a deployment claim.
- No fixed-point schema, RTL, formal equivalence, synthesis, timing, or device
  result is claimed.

## Reproduction

```bash
go test ./services ./neurons/sigmoid_rate -run SigmoidRate -count=1
cargo test --manifest-path src/sc_neurocore/accel/rust/Cargo.toml \
  sigmoid_rate --lib -j 4

PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_sigmoid_rate.py \
  tests/test_sigmoid_rate_backend_loading.py \
  tests/test_sigmoid_rate_backends.py \
  tests/test_bench_sigmoid_rate.py

taskset -c 4 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_model_sigmoid_rate.py \
  --json benchmarks/results/local_python_2026-07-14_sigmoid_rate.json
```
