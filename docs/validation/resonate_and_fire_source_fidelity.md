<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Resonate-and-fire fidelity evidence -->

# Resonate-and-fire source, runtime, and co-simulation fidelity

This page records the publication boundary, independent trace, five-runtime
parity, failure atomicity, generated-RTL envelope, bounded formal property,
and controlled benchmark for `ResonateAndFireNeuron`.

## Primary-source boundary

Izhikevich (2001) defines `z=x+i*y`, calls `x` current-like and `y`
voltage-like, and writes

\[
\dot z=(b+i\omega)z+I.
\]

The source fires at the voltage condition `y=1` and then resets `z=i`.
SC-NeuroCore generalizes only the threshold magnitude, producing
`old_y < threshold <= candidate_y` and post-spike
`(x, y)=(0, threshold)`. Radius thresholding and origin reset are different
models and are not maintained here.

Primary source:
[Izhikevich 2001](https://doi.org/10.1016/S0893-6080(01)00078-8).

## Exact-flow and event convention

For real constant input during one sample, the implementation evaluates the
closed-form two-dimensional linear flow. All runtimes independently compute
the equilibrium `(x_ss, y_ss)`, damped rotation, candidate state, sampled
crossing, and source reset. The implementation’s `dt=0.01` default and sampled
crossing rule are engineering contracts; the paper does not mandate that
sampling interval or a numerical solver.

The event rule is intentionally phase-sensitive. A high current can leave the
first post-reset candidate above threshold; because a new event requires an
upward crossing from below, spike count can fall again at high drive. The
descriptor therefore records both phasic and tonic behavior but does not call
the output a monotone rate code.

## Independent reference evidence

`src/sc_neurocore/neurons/reference_trace_data/resonate_fire_subthreshold_resonance_doi.json`
pins a source-default, constant-input subthreshold trajectory. Its test
re-derives the exact matrix flow without importing production equation
helpers, checks the recorded features, and compares all production states and
events.

The descriptor uses a separate 256-step varied-input replay and interleaves
post-update `x`, `y`, and `spikes` as little-endian float64. The canonical
digest is
`d91c7ef4a469f7b50498943419288b12329b17ba8380bcc876d4963480ae7130`.

## Numerical and atomic contract

Each scalar step:

1. validates the complete state, configuration, and current;
2. derives finite equilibrium and damped-rotation coefficients;
3. computes both candidate coordinates without mutation;
4. evaluates the sampled voltage-coordinate crossing;
5. validates the candidate and commits both states together.

Public dispatch validates all three trace shapes, finiteness, binary events,
spike-reset coordinates, final-state receipts, and the integral event count
before mutating the caller’s model. Go and Mojo reject null, mis-sized, and
overlapping ABI regions and do not write partial outputs. Julia validates
element type, stride, writability, and overlap before writing. Rust returns
owned traces only after a complete successful batch. Empty batches preserve
state on every runtime.

## Executable parity matrix

| Runtime | Executed surface | Enrolled result |
|---|---|---|
| Python | scalar reference and atomic batch | reference |
| Rust engine | modular PyO3 batch | all traces/finals/count within `1e-12` |
| Rust safety | standalone `rustc --test` module | equation, errors, reset, atomicity |
| Julia | JuliaCall batch and typed error predicates | all traces/finals/count within `1e-12` |
| Go | status-coded C-shared ABI | all traces/finals/count within `1e-12` |
| Mojo | status-coded shared ABI | all traces/finals/count within `1e-10` |

The configured test starts at `(0.13, -0.27)`, uses `b=-0.8`, `omega=7.5`,
`threshold=0.9`, `dt=0.006`, and a varied 128-step drive. It compares complete
`x`, `y`, and binary-event traces, both final states, and spike count. Explicit
selection of an absent runtime fails rather than substituting Python.

## Paired schema and Q32.32 co-simulation

The TOML and JSON schemas are structurally identical. Direct floating-point
schema execution reproduces the hand model. Their chained threshold comparison
lowers to a logical conjunction, preserving the two-sided sampled crossing;
fixed-point multiplication of Boolean intermediates would be incorrect and is
not used.

The compiler’s transcendental lookup grid has step `0.125`. Default
`omega*dt=0.1` is therefore not an exact lookup point. The H1 co-simulation
honestly enrolls `b=0`, `omega=8`, `dt=1/64`, and non-zero initial state, so
`exp(b*dt)=1` and `omega*dt=0.125` land exactly on the lookup grid. Across 64
varied samples:

- the full spike vector matches, including a non-trivial event;
- maximum Q32.32 `x` and `y` errors remain below `1e-8`;
- an independent 96-step alternating signed-drive stress stays below `5e-8`.

These results establish the declared operating-point envelope. They do not
claim arbitrary-parameter, default-parameter, or real-number equivalence for
the finite lookup tables.

The canonical `sc_resonate_and_fire.sby` job runs depth-4 BMC over the same
Q32.32 generated RTL and proves only that asynchronous reset clears the public
spike flag. The neuron has real event semantics, so no event-silence property
is asserted. The job does not prove equation equivalence or H4.

## Controlled benchmark

`benchmarks/bench_model_resonate_and_fire.py` measures the same complete
configured batch through all five public dispatchers. It rejects an unpinned
run, any unavailable runtime, a standalone Rust-safety failure, malformed
evidence, or trajectory/final/event-count drift. The committed JSON binds raw
timings, source hashes, exact loaded Rust/Go/Mojo binaries, tool versions,
affinity, load, and every parity delta.

Timings are classified as local, single-logical-CPU, non-exclusive regression
evidence. They are not production, cross-host, cross-framework, silicon, or
universal-ranking claims. The generated comparison table is the public timing
surface.

## Boundaries

- Only real piecewise-constant current is exposed by the scalar pipeline.
- Sample boundaries, not continuous root localization, define event time.
- The voltage-like coordinate `y`, not radius, is thresholded.
- Source reset is generalized `z=i*threshold`, not the origin.
- The high-drive response is not claimed monotone.
- Q32.32 co-simulation reaches H1 only.
- No synthesis, timing closure, formal equivalence, resource, or PPA result is
  claimed.

## Reproduction

```bash
WHEEL_SITE=/path/to/isolated-wheel-site
BUILD_ROOT=/path/to/writable/build-root
TMPDIR="$BUILD_ROOT/tmp"
mkdir -p "$TMPDIR" "$BUILD_ROOT/bin"

TMPDIR="$TMPDIR" rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/resonate_and_fire.rs \
  -o "$BUILD_ROOT/bin/resonate_and_fire_safety_tests"
"$BUILD_ROOT/bin/resonate_and_fire_safety_tests"

TMPDIR="$TMPDIR" PYTHONPATH=$WHEEL_SITE:src:. python -m pytest -q \
  tests/test_model_resonate_and_fire.py \
  tests/test_reference_resonate_fire.py \
  tests/test_resonate_and_fire_dynamics.py \
  tests/test_resonate_and_fire_backends.py \
  tests/test_resonate_and_fire_native_abi.py \
  tests/test_cosim_resonate_and_fire.py \
  tests/test_bench_resonate_and_fire.py

(cd hdl/formal/catalogue && sby -f sc_resonate_and_fire.sby)

taskset -c <cpu> env \
  TMPDIR="$TMPDIR" \
  PYTHONPATH=$WHEEL_SITE:src:. python \
  benchmarks/bench_model_resonate_and_fire.py \
  --json benchmarks/results/bench_resonate_and_fire.json
```
