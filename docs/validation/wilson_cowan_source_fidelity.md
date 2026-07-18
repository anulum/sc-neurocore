# Wilson-Cowan source and polyglot fidelity evidence

This page records the scientific boundary, equations, five-runtime parity,
failure atomicity, and controlled benchmark used to promote
`WilsonCowanUnit` to the polyglot-complete catalogue.

## Scientific scope

Wilson and Cowan (1972) derive coupled coarse-grained equations for excitatory
and inhibitory population activities. Their final population equations include
response functions and availability/refractory factors. The maintained
SC-NeuroCore model uses the common normalised reduction

\[
\tau_e\dot E=-E+S(w_{ee}E-w_{ei}I+I_{ext}),\qquad
\tau_i\dot I=-I+S(w_{ie}E-w_{ii}I),
\]

with a shifted logistic satisfying \(S(0)=0\). It omits the paper's explicit
availability/refractory multipliers and does not expose an independent
inhibitory external input. This scope is declared in the Python model, paired
schemas, descriptor, and public documentation.

Primary source: [Wilson and Cowan 1972](https://pmc.ncbi.nlm.nih.gov/articles/PMC1484078/),
[doi:10.1016/S0006-3495(72)86068-5](https://doi.org/10.1016/S0006-3495(72)86068-5).

## Numerical contract

Every runtime evaluates the same four RK4 stages from the same pre-step E/I
pair and commits both candidates together. The branch-stable logistic avoids
overflow at finite saturation inputs. Configuration requires finite,
non-negative weights; positive time constants, steepness, and `dt`; finite
`theta`; and E/I values inside the normalised `[-beta, 1]` state envelope,
where `beta = logistic(-a * theta)`. The shifted response itself remains in
`[-beta, 1-beta]`; accepting the saturated initial boundary `1` is safe because
the relaxation term immediately points inward.

The public batch accepts one constant excitatory current. Existing raw native
parity tests additionally exercise varying drive vectors. All native batch
surfaces validate the whole contract before caller-visible writes.

## Executable parity matrix

| Runtime | Executed surface | Enrolled result |
|---|---|---|
| Python | public scalar and atomic batch | reference |
| Rust engine | PyO3 modular batch | complete trace within `1e-9` |
| Rust safety | independently compiled module | RK4 reference, saturation boundary, and atomic errors; 7/7 tests |
| Julia | `simulate_wilson_cowan!` through JuliaCall | complete trace within `1e-9` |
| Go | generated C-shared ABI | complete trace within `1e-9` |
| Mojo | exported shared-library C ABI | complete trace within `1e-8` |

The controlled trajectory starts from `E=0.1`, `I=0.05`, uses the default
weights, time constants, shifted sigmoid, and `dt=0.1`, then applies
`I_ext=1.5` for 100,000 RK4 steps. The Python canonical interleaved E/I trace
has SHA-256
`0033492a00af00c389e88bee83b5a48cad74137f311a4bfb36e9882c42b6c50e`.

The focused tests also cover sigmoid asymptotes, quiescent, driven, and
sustained oscillatory regimes, an RK4-versus-Euler discriminator, empty
batches, explicit-unavailable
backends, malformed native results, corrupted mutable state, invalid external
drive, and Go/Mojo contracts whose caller buffers must remain unchanged.

## Generated-RTL co-simulation

`tests/test_cosim_wilson_cowan.py` first proves that the authored TOML and JSON
schemas are structurally identical and that both reproduce the maintained hand
RK4 trajectory within `1e-15`. It then generates the production Q32.32 Verilog
module and executes it with Icarus/Verilog over 96 mixed-drive samples:

- 8 samples at `I_ext=0`;
- 16 each at `I_ext=1.5`, `3.0`, and `5.0`;
- 8 at `I_ext=-1.0`;
- 32 sinusoidally varied samples centred at `2.0`.

The measured maximum absolute error across both public E/I outputs is
`0.019371701775768302`, below the declared `0.021` envelope. Both rates remain
inside the normalised Wilson-Cowan state envelope and every `spike_out` value
is zero. The test asserts Icarus availability instead of skipping, so missing
hardware tooling fails the lane closed.

## Controlled benchmark

`benchmarks/bench_wilson_cowan.py` measures the complete 100,000-step E/I
trace through each public dispatcher five times after warm-up. It fails if any
runtime is absent, standalone Rust-safety tests fail, or a trajectory or final-
state delta exceeds `1e-9` for Rust/Julia/Go or `1e-8` for Mojo.

The committed artefact records every raw sample, exact source and native-binary
hashes, tool versions, affinity, load, and parity deltas. It is single-logical-
CPU, non-exclusive local regression evidence, not production, cross-host,
hardware, or universal-ranking evidence.

The exact measured timing table is reproduced in
[Benchmark Comparison](../benchmarks/comparison.md). The dedicated
`wilson-cowan-rk4-five-backend-local-regression` evidence gate rejects missing
backends, source drift, parity drift, or missing numerical timing records.
The aggregate 2026-07-14 gate report evaluates this Wilson-Cowan gate with zero
Wilson-Cowan failures. Its overall status remains red because 68 source-hash
drifts belong to older benchmark artefacts elsewhere in the repository; those
inherited failures are retained in the report rather than suppressed.

## Boundaries

- E and I are continuous normalised population activities, not spikes.
- The maintained equations are an explicit normalised reduction, not a
  verbatim reproduction of the paper's availability/refractory factors.
- Only the excitatory population receives an exposed external drive.
- RK4 is a maintained numerical choice, not a method prescribed by the paper.
- The benchmark is local non-exclusive regression evidence.
- Generated Q32.32 trajectory execution is H1 co-simulation evidence only.
- No formal equivalence, synthesis, timing, device, PPA, or production-speed
  result is claimed.

## Reproduction

```bash
rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/wilson_cowan.rs \
  -o /tmp/wilson_cowan_tests
/tmp/wilson_cowan_tests

PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_wilson_cowan.py \
  tests/test_wilson_cowan_accel_dispatch_contracts.py \
  tests/test_wilson_cowan_backends.py \
  tests/test_bench_wilson_cowan.py \
  tests/test_cosim_wilson_cowan.py

taskset -c 4 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_wilson_cowan.py \
  --json benchmarks/results/bench_wilson_cowan.json
```
