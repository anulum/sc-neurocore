<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore - Package boundary decision
-->

# v4 Package Boundary Decision

This decision records the install and source-boundary state used by the current
v4 pruning work. It does not change package metadata by itself; it classifies
the package discovery excludes, optional dependency groups, and Rust workspace
excludes that already exist in `pyproject.toml` and `Cargo.toml`.

## Decision

The base wheel stays the core package surface. It carries the Python package,
NumPy/SciPy runtime dependencies, typed public facade, packaged HDL resources,
and documented CPU-side stochastic-computing workflows. Larger training,
hardware, web, quantum, biological, visualisation, and research integrations
remain opt-in extras or source-checkout research surfaces until they meet the
same packaging, evidence, documentation, and release gates as the core package.

The root Cargo workspace keeps `engine` as the only default Rust member. Other
Rust crates remain separate crate workspaces or retired candidates until their
release ownership and verification gates are explicit.

## v4 Product Boundary

`sc-neurocore-core` is the planned small product boundary for the v4 split. The
minimal install is focused on the surfaces below; everything outside this set is
kept as an optional, source-checkout, or separately owned research path until it
meets the same evidence, coverage, docs, and packaging gates as the core.

| Surface | Boundary decision | Required evidence before promotion |
| --- | --- | --- |
| `python package` | core package | Typed public facade, NumPy/SciPy runtime compatibility, packaged resources, and real-surface tests. |
| `rust engine` | core package | Maintained PyO3 `engine` workspace member with focused Rust and Python integration evidence. |
| `stochastic bitstreams` | core package | Documented bitstream APIs, deterministic fixtures, and cross-check tests against Python behavior. |
| `verilog/rtl export` | core package | Packaged HDL resources, export examples, and simulator/formal evidence that matches public claims. |
| `nir` | core-adjacent optional extra | Neuromorphic Intermediate Representation import/export remains optional until release fixtures and compatibility coverage are promoted. |

The following named extensions are explicitly outside `sc-neurocore-core` until
they satisfy the same promotion gate:

| Surface | Boundary decision | Required evidence before promotion |
| --- | --- | --- |
| `quantum` | research/experimental extension; optional extra | Reproducible backend fixtures, bounded public API, dependency gate, and benchmark evidence. |
| `bioware/MEA` | research/experimental extension | Biological-signal safety boundaries, data provenance, validation fixtures, and clinical-facing documentation controls. |
| `photonic` | research/experimental extension | Layout-tool dependency gate, hardware assumptions, generated artefact fixtures, and comparison evidence. |
| `robotics` | research/experimental extension | Hardware safety boundary, simulator or device fixtures, ownership split, and failure-mode documentation. |
| `world-model` | research/experimental extension | Public API selection, model/data provenance, bounded dependencies, and reproducible evaluation fixtures. |
| `audio` | research/experimental extension | Latency fixtures, dependency gate, sample-rate compatibility coverage, and documented runtime limits. |
| `sleep` | research/experimental extension | Protocol validation fixtures, safety boundary, bounded dependencies, and evidence traceability. |
| `swarm` | research/experimental extension | Runtime ownership split, coordination fixtures, safety invariants, and benchmark evidence for promoted workflows. |

## Optional Dependency Groups

| Metadata token | Decision | Rationale |
| --- | --- | --- |
| `minimal` | core package | Explicit v4 alias for the base dependency set; it adds no package beyond the base wheel. |
| `core` | core package | Alias for the base dependency set; it adds no package beyond the base wheel. |
| `dev` | contributor extra | Contributor lint, type, test, audit, and documentation tooling. |
| `dev-full` | contributor extra | Full contributor research stack for local validation, not a user install profile. |
| `accel` | optional extra | Numba acceleration for local CPU experiments. |
| `gpu` | optional extra | CuPy CUDA profile; depends on local driver and hardware compatibility. |
| `hdl` | optional extra | Unit-aware HDL workflows and packaged RTL resource access. |
| `license` | optional extra | HTTP client support for commercial-license status checks. |
| `full` | optional extra | Local research union for broad workflows; not the default install. |
| `research` | optional extra | General research dependencies for plotting, graph work, ONNX, and PyTorch paths. |
| `chemistry` | optional extra | Chemistry and molecular-tooling experiments. |
| `training` | optional extra | PyTorch-backed training workflows. |
| `jax` | optional extra | JAX acceleration and autodiff research paths. |
| `julia` | optional extra | Julia bridge support through `juliacall`. |
| `quantum` | optional extra | Qiskit and PennyLane quantum-circuit experiments. |
| `quantum-cognition` | optional extra | Alias over the quantum stack for Fisher-Posner experiments. |
| `nir` | optional extra | Neuromorphic Intermediate Representation import/export. |
| `mpi` | optional extra | Distributed simulation through `mpi4py`. |
| `lava` | optional extra | Intel Lava bridge profile, constrained to Python versions supported by Lava. |
| `studio` | optional extra | Local web studio runtime dependencies. |
| `federation` | optional extra | SCPN Studio platform federation vertical. |
| `docs` | contributor extra | Documentation build dependencies. |
| `compression` | optional extra | Wavelet and compressed-raster codec support. |
| `optics` | optional extra | Photonic-layout experiments through `gdsfactory`. |
| `bioware` | optional extra | Spike-sorting and biological closed-loop prototype dependencies. |
| `units` | optional extra | Pint-backed dimensional analysis. |
| `symbolic` | optional extra | SymPy symbolic differentiation for the exponential-Euler diagonal Jacobian; only `exp_euler` models pull it in. |

## Python Package Discovery Excludes

These `setuptools` excludes keep broad research modules out of the base wheel.
They stay available from a source checkout.

| Metadata token | Decision | Rationale |
| --- | --- | --- |
| `sc_neurocore.analysis` | source-checkout research surface | Analysis tooling remains broad and optional; package promotion requires a bounded public API and dependency gate. |
| `sc_neurocore.audio` | source-checkout research surface | Audio engines remain research-only until latency and dependency evidence justify packaging. |
| `sc_neurocore.bio` | source-checkout research surface | Biological circuit experiments stay source-only until validation and safety boundaries are explicit. |
| `sc_neurocore.chaos` | source-checkout research surface | Chaotic-RNG research utilities stay source-only until a packaged API boundary is selected. |
| `sc_neurocore.dashboard` | source-checkout research surface | Terminal and dashboard tooling remains source-only; web Studio is the packaged UI dependency profile. |
| `sc_neurocore.drivers` | source-checkout research surface | Hardware-driver interfaces require per-device safety and install gates before packaging. |
| `sc_neurocore.experiments` | source-checkout research surface | Experiment management remains a checkout workflow, not a base-wheel API. |
| `sc_neurocore.generative` | source-checkout research surface | Generative models remain research-only until evidence and dependencies are bounded. |
| `sc_neurocore.interfaces` | source-checkout research surface | External interfaces require dedicated compatibility and safety gates before packaging. |
| `sc_neurocore.optics` | source-checkout research surface | Photonic workflows remain source-only even though the dependency group is opt-in. |
| `sc_neurocore.physics` | source-checkout research surface | Physics experiments remain source-only until the validated subset is promoted. |
| `sc_neurocore.robotics` | source-checkout research surface | Robotics and swarm-control hooks require hardware and safety evidence before packaging. |
| `sc_neurocore.sleep` | source-checkout research surface | Sleep-protocol optimization remains a research checkout path. |
| `sc_neurocore.swarm` | source-checkout research surface | Swarm tooling stays source-only until ownership and runtime boundaries are split. |
| `sc_neurocore.viz` | source-checkout research surface | Visualisation remains a source-checkout feature until the plotting API and dependencies are bounded. |
| `sc_neurocore.world_model` | source-checkout research surface | World-model experiments stay outside the base wheel until a separate package or public API is chosen. |

## Cargo Workspace Excludes

The root Cargo workspace builds the maintained `engine` package by default.
Excluded crates are not part of the default wheel build.

| Metadata token | Decision | Rationale |
| --- | --- | --- |
| `crates/autonomous_learning` | separate crate | Autonomous-learning FFI work stays outside the default engine workspace until release ownership is explicit. |
| `crates/core_engine` | separate crate | Core-engine experiments remain split from the maintained PyO3 engine. |
| `crates/evo_substrate_core` | separate crate | Evolution-substrate work remains an independent research crate. |
| `crates/neuro_symbolic` | separate crate | Neuro-symbolic Rust work remains independent until its Python and benchmark contracts are selected. |
| `crates/spike_stats_core` | separate crate | Spike-statistics Rust work remains independent until the codec/stat API is promoted. |
| `crates/stochastic_doctor_core` | separate crate | Diagnostic tooling remains an independent audit crate. |
| `crates/tinysc_riscv` | separate crate | Embedded RISC-V work keeps separate target assumptions from the host PyO3 engine. |
| `fuzz` | separate crate | Fuzzing harnesses are kept out of the default workspace build. |
| `src/sc_neurocore/accel/rust` | retired candidate | Legacy embedded Rust acceleration stays excluded until removed or rehomed under the maintained engine path. |

## Promotion Gate

An excluded Python module or Rust crate can move into the package only after it
has a bounded public API, strict typing or Rust lint coverage, module-specific
real-surface tests, dependency documentation, and release evidence that matches
the package maturity classifier.
