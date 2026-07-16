<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — MPR source and runtime fidelity evidence -->

# Montbrió–Pazó–Roxin source, runtime, and co-simulation fidelity

This page records the publication boundary, independent equation trace,
five-runtime parity, failure atomicity, fixed-point envelope, and controlled
benchmark used to promote the legacy-named `ErmentroutKopellPopulation`.

## Primary-source boundary

Montbrió, Pazó, and Roxin (2015) derive an exact macroscopic description for
the thermodynamic limit of all-to-all coupled quadratic integrate-and-fire
neurons with Lorentzian-distributed excitability. Their equations (12a–b)
close the dimensionless network dynamics in population rate \(R\), mean
membrane potential \(v\), and source time \(t'\):

\[
\frac{dR}{dt'}=\frac{\Delta}{\pi}+2Rv,
\qquad
\frac{dv}{dt'}=v^2+\bar\eta+JR+I(t')-(\pi R)^2.
\]

Primary source:
[Montbrió, Pazó, and Roxin 2015](https://doi.org/10.1103/PhysRevX.5.021028).
The public class name is a compatibility boundary only; attributing these
equations to Ermentrout and Kopell (1986) would be incorrect.

## Time-scale and solver convention

SC-NeuroCore restores a physical population rate and time scale with the
exact change of variables

\[
R=\tau r,\qquad t'=\frac{t}{\tau}.
\]

Therefore \(dR/dt'=\tau^2\,dr/dt\) and
\(dv/dt'=\tau\,dv/dt\). Substitution into the printed dimensionless
equations yields the maintained flow:

\[
\frac{dr}{dt}=\frac{\Delta}{\pi\tau^2}+\frac{2rv}{\tau},
\qquad
\frac{dv}{dt}=
\frac{v^2+\bar\eta+J\tau r+I(t)-(\pi\tau r)^2}{\tau}.
\]

This is a change of variables, not a numerical modification. The paper
defines continuous equations and does not require a particular solver.
SC-NeuroCore uses simultaneous explicit Euler at `dt=0.01` as its
deterministic implementation contract and records that choice separately
from the scientific source.

## Independent reference trace

`src/sc_neurocore/neurons/reference_trace_data/ermentrout_kopell_pop_eq12_euler_doi.json`
pins a 256-step trace with:

- a deterministic sinusoidal external drive;
- both post-update states;
- the DOI and exact equation/solver boundary;
- first, final, minimum, maximum, and mean features for both traces.

`tests/test_reference_ermentrout_kopell_pop.py` independently re-derives the
two equations without importing production equation helpers, checks every
recorded feature, and compares every Python production state exactly. The
canonical interleaved little-endian float64 digest is
`6bf2d0b8611cba646e5b575ed3253090e1e02955c75c01efac13e603578edae5`.
The deterministic drive is a parity protocol, not a paper-figure
reproduction.

## Numerical and atomic contract

Each scalar step:

1. validates all parameters, the complete current state, and external drive;
2. evaluates both derivatives from the same pre-update state;
3. evaluates both Euler candidates;
4. rejects a non-finite or negative-rate candidate;
5. commits both states together.

Public batch dispatchers validate both trace shapes, finiteness, non-negative
rate, and both final-state receipts before mutating the model instance. Go
computes into private scratch traces before writing its C outputs. Mojo runs a
validation pass before its output pass. Rust returns owned traces only after
the complete batch succeeds. Empty batches preserve both states.

## Executable parity matrix

| Runtime | Executed surface | Enrolled result |
|---|---|---|
| Python | scalar reference and atomic batch | reference |
| Rust engine | modular PyO3 scalar and batch | two traces/two final states within `1e-12` |
| Rust safety | standalone `rustc --test` module | equation wiring, validation, atomic mutation, reset |
| Julia | JuliaCall batch | two traces/two final states within `1e-12` |
| Go | generated C-shared ABI | two traces/two final states within `1e-12` |
| Mojo | exported shared-library C ABI | two traces/two final states within `1e-10` |

The configured test begins from non-zero state values, changes every public
parameter, supplies a varied 128-step drive, and compares both trajectories
plus both final states. Explicit selection of an absent runtime fails rather
than substituting the Python path.

## Paired schema and Q32.32 co-simulation

The TOML and JSON schemas are structurally identical and their direct floating-
point execution matches the hand model within `1e-13`. One input edge carries
one external-drive sample and advances both states simultaneously.

Generated Q32.32 Verilog compiles with Icarus and preserves the varied 64-step
trajectory with maximum firing-rate and voltage errors below `2e-6`. This is a
bounded fixed-point co-simulation result, not exact RTL equivalence.

The canonical `sc_ermentroutkopellpopulation.sby` job runs depth-4 BMC over
the same Q32.32 generated RTL. It proves asynchronous reset clears
`spike_out` and, after the first sampled clock, the public spike output
remains zero because this population model has no binary event semantics.
It does not claim equation equivalence, synthesis, timing, or H4 evidence.

## Controlled benchmark

`benchmarks/bench_ermentrout_kopell_pop.py` measures the same complete
deterministic-drive batch through all five public dispatchers. It rejects an
unpinned run, any unavailable runtime, a standalone Rust-safety failure,
trace/final-state drift, or an invalid run size. The committed JSON binds raw
timing samples, source hashes, exact loaded Rust/Go/Mojo binaries, tool
versions, affinity, load, and every parity delta.

The recorded run used logical CPU 10, five 50,000-step samples after a
1,000-step warm-up, and the exact isolated Rust wheel built for this source
tree. Its one-minute load average was 23.65, so the timings are deliberately
classified as non-exclusive diagnostics. The record separately identifies
the JuliaCall runtime (1.11.9) and PATH Julia CLI (1.12.6), the Go shared
library's embedded builder (1.26.3) and PATH Go CLI (1.24.0), and the pinned
Pixi Mojo builder (0.26.2) and PATH Mojo CLI (1.0.0b1).

| Runtime | Median call | Median ns/step | Maximum trace difference | Trace mismatches |
|---|---:|---:|---:|---:|
| Julia | 3.844 ms | 76.875 | `0` | 0 |
| Mojo | 4.088 ms | 81.761 | `2.220e-16` | 0 |
| Go | 4.571 ms | 91.429 | `0` | 0 |
| Rust | 4.703 ms | 94.058 | `0` | 0 |
| Python | 170.959 ms | 3,419.182 | `0` | 0 |

Python, Rust, Julia, and Go share interleaved-trace SHA-256
`0e9c59cbe73cb9019d309fc484fa67c838ae503d0c9f9f7a6825bb6fa857cb7b`.
Mojo has a distinct byte digest because of its bounded final-bit difference.
The complete JSON evidence SHA-256 is
`7c5c1ac6acd7e3593dd7289612d1a8687f361d1b52c6d564ca8b95961036206e`.

The record is one-logical-CPU, non-exclusive local regression evidence. It is
not a production, cross-host, cross-framework, or hardware performance claim.
The timing table is reproduced in
[Benchmark Comparison](../benchmarks/comparison.md).

## Boundaries

- The output is a continuous population firing rate, not a binary spike.
- The Euler step is an implementation choice, not imposed by the paper.
- The deterministic input protocol does not reproduce a published figure.
- The exact reduction does not model finite-size fluctuations, delays,
  adaptation, or sparse connectivity.
- Q32.32 co-simulation reaches H1 only.
- No synthesis, timing closure, formal equivalence, device resource, or PPA
  result is claimed.

## Reproduction

Keep temporary and build outputs on a writable working volume:

```bash
WHEEL_SITE=/path/to/isolated-wheel-site
BUILD_ROOT=/path/to/writable/build-root
TMPDIR="$BUILD_ROOT/tmp"
mkdir -p "$TMPDIR" "$BUILD_ROOT/bin"

TMPDIR="$TMPDIR" \
  rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/ermentrout_kopell_pop.rs \
  -o "$BUILD_ROOT/bin/mpr_safety_tests"
"$BUILD_ROOT/bin/mpr_safety_tests"

TMPDIR="$TMPDIR" \
PYTHONPATH=$WHEEL_SITE:src:. .venv/bin/python -m pytest -q \
  tests/test_model_ermentrout_kopell.py \
  tests/test_ermentrout_kopell_pop_dynamics.py \
  tests/test_reference_ermentrout_kopell_pop.py \
  tests/test_cosim_ermentrout_kopell_pop.py \
  tests/test_ermentrout_kopell_pop_backends.py \
  tests/test_bench_ermentrout_kopell_pop.py

taskset -c 10 env \
  TMPDIR="$TMPDIR" \
  PYTHONPATH=$WHEEL_SITE:src:. .venv/bin/python \
  benchmarks/bench_ermentrout_kopell_pop.py \
  --json benchmarks/results/bench_ermentrout_kopell_pop.json
```
