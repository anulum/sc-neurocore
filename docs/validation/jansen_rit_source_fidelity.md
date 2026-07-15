<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Jansen–Rit source and runtime fidelity evidence -->

# Jansen–Rit source, runtime, and co-simulation fidelity

This page records the publication boundary, independent equation trace,
five-runtime parity, failure atomicity, fixed-point envelope, and controlled
benchmark used to promote `JansenRitUnit`.

## Primary-source boundary

Jansen and Rit (1995) model one cortical column as interacting pyramidal,
excitatory-interneuron, and inhibitory-interneuron populations. Equation (6)
uses three second-order synaptic operators and the connectivity constants

\[
C_1=C,
\qquad C_2=0.8C,
\qquad C_3=C_4=0.25C.
\]

For the excitatory feedback term, the published wiring is
\(C_2S(C_1y_0)\). The \(C_2\) factor therefore multiplies the sigmoid output;
it does not scale the sigmoid input.

Primary source:
[Jansen and Rit 1995](https://doi.org/10.1007/BF00199471).
Source-bound executable transcription:
[pinned Brian2 example](https://github.com/brian-team/brian2/blob/1bfa1a9275bd9672b49f4bf61ffbaf6f7cb55fc9/examples/frompapers/Jansen_Rit_1995_single_column.py).

## Timestep and sigmoid convention

The publication defines continuous equations and does not require a particular
numerical solver. The pinned Brian2 example uses explicit Euler at `0.1 ms`;
SC-NeuroCore adopts that step as the maintained default and records the solver
choice as implementation scope rather than attributing it to the paper.

Brian2 names the complete sigmoid numerator `e0=5`. SC-NeuroCore names the
half-maximum rate `e0=2.5` and evaluates `2*e0/(1+exp(...))`; the two
conventions are algebraically identical at their defaults.

## Independent reference trace

`src/sc_neurocore/neurons/reference_trace_data/jansen_rit_eq6_euler_brian2.json`
pins a 256-step trace with:

- a deterministic, varied external pulse-density drive;
- every post-update state and the `y1-y2` EEG proxy;
- the DOI, Brian2 URL, and exact source commit;
- first, final, minimum, maximum, and mean features for all seven traces.

`tests/test_reference_jansen_rit.py` independently re-derives equation (6)
without importing production equation helpers, checks every recorded feature,
and compares every production state exactly. The canonical interleaved
little-endian float64 digest is
`84e9273e381d543a5fb32a510a4c82ea977c32cdd267e88b790296e6e8364933`.
The deterministic drive is a parity protocol, not a paper-figure reproduction.

## Numerical and atomic contract

Each scalar step:

1. validates parameters, the complete current state, and external drive;
2. evaluates all sigmoid responses from the pre-update state;
3. evaluates all six simultaneous Euler candidates;
4. validates the complete candidate;
5. commits all six states together.

The sigmoid uses a branch-stable exponential form. Public batch dispatchers
also validate all seven trace shapes, finiteness, the `eeg == y1 - y2`
identity, and every final-state receipt before mutating the model instance.
Empty batches preserve all six states.

## Executable parity matrix

| Runtime | Executed surface | Enrolled result |
|---|---|---|
| Python | scalar reference and atomic batch | reference |
| Rust engine | modular PyO3 scalar and batch | seven traces/six final states within `1e-11` |
| Rust safety | standalone `rustc --test` module | equation wiring, validation, atomic mutation, reset |
| Julia | JuliaCall batch | seven traces/six final states within `1e-11` |
| Go | generated C-shared ABI | seven traces/six final states within `1e-11` |
| Mojo | exported shared-library C ABI | seven traces/six final states within `1e-8` |

The configured test begins from non-zero values for all six states, changes
every public parameter, supplies a varied 128-step drive, and compares all
seven trajectories plus all six final states. Explicit selection of an absent
runtime fails rather than substituting the Python path.

## Paired schema and Q32.32 co-simulation

The TOML and JSON schemas are structurally identical and their direct floating-
point execution matches the hand model within `1e-13`. One input edge carries
one external-drive sample and advances all six states simultaneously.

Generated Q32.32 Verilog compiles with Icarus and preserves the varied 64-step
trajectory within:

- maximum potential-state error below `0.02 mV`;
- maximum derivative-state error below `3.2 mV/s`;
- maximum EEG-proxy error below `0.02 mV`.

These envelopes include fixed-point quantisation and the generic exponential
lookup. They are an explicit bounded co-simulation result, not exact
transcendental equivalence.

## Controlled benchmark

`benchmarks/bench_jansen_rit.py` measures the same complete deterministic-drive
batch through all five public dispatchers. It rejects an unpinned run, any
unavailable runtime, a standalone Rust-safety failure, trace/final-state drift,
or an invalid run size. The committed JSON binds raw timing samples, source
hashes, exact loaded Rust/Go/Mojo binaries, tool versions, affinity, load, and
every parity delta.

The record is one-logical-CPU, non-exclusive local regression evidence. It is
not a production, cross-host, cross-framework, or hardware performance claim.
The timing table is reproduced in
[Benchmark Comparison](../benchmarks/comparison.md).

## Boundaries

- The EEG proxy is a continuous mean-field potential, not a binary spike.
- The default Euler step follows the pinned executable transcription; it is not
  imposed by the continuous publication equations.
- The deterministic input protocol does not reproduce a published figure.
- Q32.32 co-simulation reaches H1 only.
- No synthesis, timing closure, formal equivalence, device resource, or PPA
  result is claimed.

## Reproduction

The focused test invocation preloads the installed Rust extension before
pytest adds repository support paths to the import search order.

```bash
rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/jansen_rit.rs \
  -o /tmp/jansen_rit_tests
/tmp/jansen_rit_tests

PYTHONPATH=bridge:src:. .venv/bin/python - <<'PY'
import sc_neurocore_engine
import pytest

raise SystemExit(
    pytest.main(
        [
            "-q",
            "tests/test_model_jansen_rit.py",
            "tests/test_jansen_rit_dynamics.py",
            "tests/test_reference_jansen_rit.py",
            "tests/test_cosim_jansen_rit.py",
            "tests/test_jansen_rit_backends.py",
            "tests/test_bench_jansen_rit.py",
        ]
    )
)
PY

taskset -c 11 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_jansen_rit.py \
  --json benchmarks/results/bench_jansen_rit.json
```
