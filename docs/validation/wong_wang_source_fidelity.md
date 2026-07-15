<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Wong-Wang source and runtime fidelity evidence -->

# Wong-Wang source, runtime, and co-simulation fidelity

This page records the source boundary, independent equation trace, five-runtime
parity, failure atomicity, fixed-point envelope, and controlled benchmark used
to promote `WongWangUnit`.

## Primary-source boundary

Wong and Wang (2006) reduce a recurrent spiking decision circuit to two NMDA
gating variables. The Appendix reduction uses the transfer

\[
\phi(I)=\frac{aI-b}{1-\exp[-d(aI-b)]},
\]

the coupled currents

\[
I_1=J_NS_1-J_{cross}S_2+I_0+I_{stim,1}+I_{noise,1},
\]

with the symmetric expression for population two, and

\[
\dot S_i=-S_i/\tau_s+(1-S_i)\gamma\phi(I_i).
\]

SC-NeuroCore also carries the paper's AMPA Ornstein-Uhlenbeck current state,
using an explicit external standard-normal sample for each population and
physical step. The maintained scope excludes recurrent AMPA.

Primary source:
[Wong and Wang 2006](https://doi.org/10.1523/JNEUROSCI.3733-05.2006).
Author-lab implementation:
[pinned `wong06.m`](https://github.com/xjwanglab/wong-wang-2006/blob/c39c6742329f89f1b5137f32910d55ad52d4bc24/wong06.m).

## Timestep and rate-order decision

The paper Methods state `dt=0.1 ms`; the pinned author script uses `0.5 ms`.
The maintained default is the paper value, `0.0001 s`. This discrepancy is
recorded rather than silently selecting the code value.

The author script stores an initial `nu` value before entering its indexed
update loop. SC-NeuroCore instead returns the algebraic pre-update transfer of
the complete current state on every call. This removes a one-index storage lag
without changing the published recurrence.

## Independent reference trace

`src/sc_neurocore/neurons/reference_trace_data/wong_wang_appendix_euler_ou_doi.json`
pins a 256-step trace with:

- varied, deterministic stimuli to both populations;
- non-zero explicit noise samples;
- all four post-update physical states;
- both pre-update rates;
- the DOI, author-lab URL, and source commit;
- every first/final/minimum/maximum/mean feature.

`tests/test_reference_wong_wang.py` independently re-derives the equations
without importing production equation helpers, checks every recorded feature,
and then compares the complete production trace byte for byte. The canonical
interleaved little-endian float64 digest is
`d39f219d3cd21d505c71749a1d9547d4cef550299f8e829bb2aa2a30d66daf44`.

## Numerical and atomic contract

Each scalar or batch step:

1. validates parameters, current state, stimuli, and the two samples;
2. evaluates both currents and both rates from the pre-update state;
3. evaluates two Euler gating candidates and two OU-current candidates;
4. validates all four candidates;
5. commits all four together.

No runtime clips an invalid gating candidate. Public Python batch state and C
ABI output buffers remain unchanged when a complete result cannot be produced.
Empty batches preserve all four physical states.

## Executable parity matrix

| Runtime | Executed surface | Enrolled result |
|---|---|---|
| Python | scalar reference and atomic batch | reference |
| Rust engine | modular PyO3 batch | six traces/four final states within `1e-12` |
| Rust safety | standalone `rustc --test` module | scalar equations, validation, symmetry, reset |
| Julia | JuliaCall batch | six traces/four final states within `1e-12` |
| Go | generated C-shared ABI | six traces/four final states within `1e-12` |
| Mojo | exported shared-library C ABI | six traces/four final states within `1e-9` |

The configured test begins from `s1=0.24`, `s2=0.11`, `noise1=0.01 nA`, and
`noise2=-0.02 nA`; changes every public parameter; supplies varied currents and
samples for 128 steps; and compares all six traces plus all four final states.

## Paired schema and Q32.32 co-simulation

The TOML and JSON schemas are structurally identical. Because the universal
schema port accepts one scalar input per edge, one physical Wong-Wang update is
serialised over six rising edges:

1. latch `stim1`;
2. latch `stim2`;
3. latch `xi1`;
4. latch `xi2` and evaluate both transfer arguments;
5. evaluate both rates through the generated `exprel` path;
6. commit both gating and both OU-current states.

Gaussian generation remains outside the datapath. The schema records
`sqrt(dt/tau_ampa)` as a derived constant for the enrolled default, avoiding a
coarse generic square-root lookup in the OU scale. The generated Q32.32
Verilog compiles with Icarus and preserves the varied 32-update trace within:

- maximum physical-state error `2.5e-4`;
- maximum rate error `0.30 Hz`.

The rate envelope reflects the generic `exprel` lookup spacing. It is an
explicit bounded co-simulation result, not exact transcendental equivalence.

## Controlled benchmark

`benchmarks/bench_wong_wang.py` measures the same complete deterministic-sample
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

- Firing rates are continuous observables, not binary spikes.
- Gaussian samples cross the accelerator and RTL boundaries explicitly.
- The paper timestep is the maintained default; the author-code discrepancy is
  preserved in the evidence.
- Q32.32 co-simulation reaches H1 only.
- No synthesis, timing closure, formal equivalence, device resource, or PPA
  result is claimed.

## Reproduction

The focused test invocation preloads the installed Rust extension before
pytest adds repository support paths to the import search order.

```bash
rustc --edition 2021 --test \
  src/sc_neurocore/accel/rust/safety/wong_wang.rs \
  -o /tmp/wong_wang_tests
/tmp/wong_wang_tests

PYTHONPATH=src:. .venv/bin/python - <<'PY'
import sc_neurocore_engine
import pytest

raise SystemExit(
    pytest.main(
        [
            "-q",
            "tests/test_model_wong_wang.py",
            "tests/test_wong_wang_dynamics.py",
            "tests/test_reference_wong_wang.py",
            "tests/test_cosim_wong_wang.py",
            "tests/test_wong_wang_backends.py",
            "tests/test_bench_wong_wang.py",
        ]
    )
)
PY

taskset -c 11 env PYTHONPATH=src:. .venv/bin/python \
  benchmarks/bench_wong_wang.py \
  --json benchmarks/results/bench_wong_wang.json
```
