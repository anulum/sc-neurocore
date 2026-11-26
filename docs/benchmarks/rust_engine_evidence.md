<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Rust engine and benchmark evidence
-->

# Rust Engine and Benchmark Evidence

This text was part of the repository README until the README became a short landing page.

The optional Rust engine provides SIMD-accelerated simulation, 208 Rust PyO3
model wrappers, a 180-model NetworkRunner dispatch list, and fused E-I network
simulation. Release automation builds pre-built `sc_neurocore_engine` wheels
with `maturin` for Python 3.11-3.14 on Linux x86_64/aarch64, macOS, and Windows.
Use a matching release wheel first:

```bash
python -m pip install sc_neurocore_engine
```

If no matching wheel exists for your platform or Python version, build the local
bridge from a source checkout with a local Rust toolchain:

```bash
python -m pip install --require-hashes -r requirements/maturin.txt
cd bridge
python -m maturin develop --release
```

The committed Brunel balanced-network scaling artefact
`benchmarks/results/rust_scaling_benchmark.json` records 39-202x wall-clock
ratios against Brian2 for its 10K-100K Rust rows — but those rows measure an
unconnected fixed-point LIF layer under constant external drive, not the
recurrent Brunel network, so they are not a dynamics-parity result (their
firing rates sit 3.2-17x above Brian2's at matched sizes). The
dynamics-parity comparison in the same artefact is the NumPy dense lane
(rates within 5% of Brian2): 2.4x faster at 1,000 neurons, 1.3x at 5,000,
and 0.75x (slower than Brian2) at 10,000. The artefact is one recording
(2026-03-08, one host, 300 ms simulated, two repetitions with one seed) and
compares mean firing rates only.
`tests/test_brunel_dynamics_parity_gate.py` checks that this text matches the
artefact; it does not rerun the simulation. The `brunel-parity.yml` workflow
reruns both lanes weekly on current code, with a seed per repetition, and
fails when the rates differ by more than the same tolerance. A rerun of that
kind on 2026-10-12 (one loaded host, 1 s simulated, eight seeds per lane,
`benchmarks/results/brunel_parity_rerun_2026-10-12.json`) gave rate ratios of
1.015 at 1,000 neurons and 0.996 at 5,000; the rate of a single seed varied
by about 1 to 4 %. Its timings are not a published comparison.

The AdEx model also exposes source-bound baseline-Euler simulation through
Python, the Rust engine, Julia, Go, and Mojo. Its committed
`benchmarks/results/bench_adex.json` artefact records exact event parity and a
`5e-12` complete `(v,w)` trace envelope. The separate source-fit receipt,
four-current Q16.16 co-simulation, Yosys synthesis receipt, and bounded formal
safety establish its honest H2 boundary; see the
[AdEx model page](../api/models/adex.md) for backend and evidence boundaries.

The ExpIF model exposes a deterministic zero-noise specialization of the
Fourcaud-Trocmé source protocol through the same five public paths: fitted
parameters, Heun RK2 below the paper's -30 mV handoff, the derived analytical
exponential-only tail duration, and the fitted 1.7 ms refractory interval. Its
committed `benchmarks/results/bench_expif.json` artefact records complete
voltage/refractory/event packets with exact event parity and a `5e-8` state
envelope. The historical SC candidate-first RK4 and Q32.32 RTL recurrence is
preserved as a separately named compatibility profile; the
[ExpIF model page](../api/models/expif.md) states both evidence boundaries.

`NonResettingLIFNeuron` now exposes the Kobayashi et al. one-timescale MAT(1)
source recurrence across Python, Rust, Julia, Go, and Mojo. The former project
exact-relaxation behavior is preserved separately as
`SCNonResettingAdaptiveLIFNeuron`; the [MAT(1) model page](../api/models/non_resetting_lif.md)
states the identity, benchmark, Q32.32, and bounded-formal evidence boundaries.

The Lapicque catalogue identity implements the 1907 polarization-threshold
circuit and its strength-duration law without inventing automatic reset or
repetitive spikes. Python, Rust, Julia, Go, and Mojo preserve the complete
polarization/event/latch packet; the committed source-hashed benchmark records
one identical first-attainment event in every lane with maximum state difference
`1.222e-15`. The former periodic hard-reset recurrence remains intact as
count-neutral `SCLapicqueLIFNeuron`; the [Lapicque model page](../api/models/lapicque.md)
states both identities and the Q32.32 source versus Q16.16 SC silicon boundaries.

The Perfect Integrator exposes the Naud-Gerstner strict source boundary through
five complete failure-atomic runtimes while retaining the former inclusive SC
profile under an explicit count-neutral identity. Its
[model page](../api/models/perfect_integrator.md) documents the DOI receipt,
source/SC schemas, controlled benchmark, and dedicated source RTL evidence.

The Quadratic IF source profile now uses Latham et al.'s actual normalized
numerical apex/reset/timestep (`31/3`, `-3`, `.05`) across Python, Rust/PyO3,
Julia, Go, and Mojo complete packets. The former symmetric `-1/+1/.01` profile
remains intact as count-neutral `SCSymmetricQuadraticIFNeuron`; the
[Quadratic IF model page](../api/models/quadratic_if.md) documents the exact
normalisation, DOI receipt, source/SC split, and silicon evidence boundary.

The Theta catalogue identity is bound to Ermentrout and Kopell's equation
(2.5): `current` is the dimensionless source parameter `a`, or a frozen slow
drive, and does not imply the full coupled parabolic-bursting system. Python,
Rust/PyO3, Julia, Go, and Mojo expose aligned failure-atomic phase/event
packets; the independent source receipt, controlled five-runtime benchmark,
Q16.16 Euler co-simulation, 6,203-cell Yosys receipt, and bounded formal job are
documented on the [Theta model page](../api/models/theta.md).

The threshold-linear rate model exposes its memoryless
`gain * max(0, current - theta)` transfer through Python, a configurable Rust
engine batch, independent Rust safety, Julia, Go, and Mojo. Its committed
source- and binary-bound benchmark requires bit-exact complete rate traces in
all five lanes and records only local non-exclusive regression timings. The
[model page](../api/models/threshold_linear_rate.md) states the continuous-
rate, no-spike, and no-RTL boundaries.

When installed, SC-NeuroCore automatically uses the Rust engine for:

- **NetworkRunner:** 180-model fused Rayon-parallel simulation loop
- **E-I network:** single Rust call for connectivity + Poisson + Euler + spike detection
- **Batch simulate:** model dispatch loop in compiled Rust
- **SIMD bitstream ops:** popcount dispatched to AVX2/AVX-512 when the CPU has them (measured figures with their host and size are in [the benchmark report](BENCHMARKS.md))

The pure Python package works without the engine — NumPy fallbacks are
used for all operations. Install or build the engine only when you need the
performance advantage. See [Install Profiles](../guides/install_profiles.md)
for the base install, optional extras, and source-build path.

`pip install sc-neurocore` publishes the Python suite under the public
`sc-neurocore` package name. The optional Rust engine remains part of the
repository / release-asset / source-build flow rather than a separate PyPI
runtime dependency. Source-only extended modules such as `analysis`, `viz`,
`audio`, `dashboard`, and `swarm` still require a source checkout.

## Rust Engine (208 PyO3 Wrappers, 180-Model NetworkRunner)

The `sc_neurocore_engine` crate provides 208 Rust PyO3 model wrappers callable
from Python (including ArcaneNeuron), a 180-model NetworkRunner with
Rayon-parallel population simulation (run at up to 100K neurons in the
committed scaling benchmark), and SIMD-accelerated primitives: AVX-512 or
AVX2 chosen at run time on x86-64 and NEON on AArch64, plus SVE and RISC-V V
code paths that are compiled only when a build enables those target
features, which no workflow or published wheel does. Rust test totals are maintained by the Rust workspace; public
release claims should cite the current `cargo test -- --list` output or CI
evidence before publication.

| Crate | Tests | Purpose |
|-------|------:|---------|
| `sc_neurocore_engine` | CI-listed | PyO3 SIMD engine, Rust model wrappers, NetworkRunner |
| `tinysc_riscv` | 83 | RISC-V SC instruction set simulator |
| `core_engine` | 22 | SC arithmetic core (standalone) |
| `autonomous_learning` | 22 | Self-modifying plasticity rules |
| `neuro_symbolic` | 28 | Hyperdimensional computing + predictive coding |
| `stochastic_doctor_core` | 34 | Bitstream diagnostics engine |

| Category | Scope |
|----------|-------|
| Primitives | Bernoulli + Sobol bitstream, pack/unpack, popcount, SIMD (AVX-512, AVX2, NEON; SVE and RISC-V V compiled on request) |
| Neurons | 208 PyO3 model wrappers; 180 canonical models wired into NetworkRunner |
| NetworkRunner | 180-model fused simulation loop with CSR projections and Rayon parallelism |
| Synapses | Static, STDP, Reward-STDP |
| Layers | Dense, Conv2D, Recurrent, Learning, Fusion, Memristive, Attention |
| Networks | Brunel, GNN, Spike recorder, Connectome, Fault injection |
| Compiler | IR builder/parser/verifier, SystemVerilog + MLIR emitters, IR bridge |
| Domain | HDC, Kuramoto, SSGF geometry |
| Training | 6 surrogate gradient functions + property tests |

The NetworkRunner inventory above and the formal inventory on the
[package architecture page](../architecture/package_architecture.md) are source-gated by
`tests/test_public_claims_metrics.py` against `supported_models()`, git-tracked
proof jobs, and HDL statements. The Studio figure is the current live
`list_models()` inventory. These are different inventories and are
intentionally not collapsed into one “model count”.

## Benchmarks

Run the benchmark suite:

```bash
python benchmarks/benchmark_suite.py           # quick mode
python benchmarks/benchmark_suite.py --full    # thorough (10x)
python benchmarks/benchmark_suite.py --markdown # output BENCHMARKS.md
```

Measured figures, each with its host, date and committed record, are in the
[benchmark report](BENCHMARKS.md); timings depend on the
machine and its load.
