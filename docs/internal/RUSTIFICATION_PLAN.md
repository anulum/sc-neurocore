# SC-NeuroCore Rustification Plan

**Date:** 2026-04-04
**Author:** Arcane Sapience
**Status:** Active — P0-A **COMPLETE**, P0-B next
**Rule:** Every compute/regex function MUST have a Rust path (no exceptions)

---

## Executive Summary

SC-NeuroCore has 460 Python modules with ~48K lines of compute code.
The Rust engine currently covers 19K lines (neuron models, network runner,
SC primitives, IR compiler). This plan brings the remaining 302
compute-heavy modules to Rust via PyO3, prioritised by hot-path impact.

**Target:** 100% Rust coverage for all functions called per-timestep,
per-spike, or per-network-step. Python remains as the user-facing API;
Rust provides the compute backend with transparent auto-dispatch.

---

## Current Rust Coverage

| Category | Python | Rust | Coverage |
|----------|--------|------|----------|
| Neuron models | 118 | 117 | **99.2%** |
| NetworkRunner dispatch | 65 | 65 | **100%** |
| SC primitives (bitstream, LFSR, popcount) | 6 | 6 | **100%** |
| IR compiler (ScGraph, parse, verify, emit) | 4 | 4 | **100%** |
| Kuramoto solver (SSGF/PGBO) | 1 | 1 | **100%** |
| Network simulation (BrunelNetwork) | 1 | 1 | **100%** |
| Analysis (spike_stats) | 22 modules, 142 fns | 22 modules, 101 fns + 96 PyO3 | **100%** (P0-A done) |
| Spike codecs | 8 modules, 55 fns | 0 | **0%** |
| Audio engines | 4 modules, 42 fns | 0 | **0%** |
| Network projection (CSR scatter) | 1 module, 11 fns | 0 | **0%** |
| Learning rules (advanced) | 1 module, 21 fns | 0 | **0%** |
| Compiler / training / adapters | ~40 modules | 0 | **0%** |
| Remainder (P3) | ~307 modules | 0 | **0%** |

---

## Priority Tiers

### P0 — Hot Path (called every timestep/spike)

These functions are invoked inside simulation loops. Rustifying them
gives the largest wall-clock speedup for typical workflows.

#### P0-A: Analysis — `analysis/spike_stats/` (22 modules, 142 functions, 3,386 lines)

| Module | Lines | Fns | Key functions |
|--------|------:|----:|---------------|
| `basic.py` | 48 | 5 | spike_count, spike_times, isi, firing_rate, cv_isi |
| `rate.py` | 84 | 3 | instantaneous_rate, psth, rate_coding_distance |
| `variability.py` | 400 | 18 | fano_factor, cv2, lv, ir, si, burst_index, ... |
| `correlation.py` | 273 | 15 | cross_correlation, spike_train_correlation, ... |
| `distance.py` | 371 | 16 | van_rossum, victor_purpura, spike_edit_distance, ... |
| `information.py` | 213 | 10 | mutual_information, transfer_entropy, ... |
| `spectral.py` | 27 | 1 | spike_spectrum |
| `temporal.py` | 101 | 4 | spike_triggered_average, jitter, ... |
| `patterns.py` | 83 | 3 | pattern_detection, synfire_chain_score, ... |
| `dimensionality.py` | 95 | 3 | spike_train_pca, participation_ratio, ... |
| `causality.py` | 193 | 8 | granger_causality, transfer_entropy_ksg, ... |
| `decoding.py` | 127 | 5 | population_vector, bayesian_decode, ... |
| `network.py` | 140 | 4 | functional_connectivity, graph_metrics, ... |
| `point_process.py` | 80 | 4 | hazard_function, conditional_intensity, ... |
| `surrogates.py` | 180 | 10 | spike_dithering, isi_shuffling, ... |
| `statistics.py` | 42 | 1 | spike_train_statistics |
| `stimulus.py` | 139 | 5 | sta, stc, spike_triggered_covariance, ... |
| `waveform.py` | 77 | 6 | peak_amplitude, half_width, ... |
| `sorting_quality.py` | 208 | 10 | isolation_distance, l_ratio, ... |
| `lfp.py` | 70 | 3 | lfp_from_spikes, coherence, ... |
| `gpfa.py` | 212 | 5 | gpfa_fit, gpfa_transform, ... |
| `spade.py` | 219 | 3 | spade_detect, spike_pattern_assembly, ... |

**Rust approach:** New `engine/src/analysis/` module tree. Each Python module
maps to a Rust file. Functions take `&[f64]` or `&[i32]` spike trains.
PyO3 wrappers accept numpy arrays via `PyReadonlyArray1<f64>`.

**Status:** **COMPLETE.** 22/22 modules ported, 597 Rust tests passing,
96 PyO3 wrappers registered. 5 functions remain Rust-only (take fn pointers).
See `docs/api/rust-analysis-engine.md` for full API reference.

**Achieved speedup:** 10–100x for distance metrics and surrogate generation
(Python loop -> Rust). FFT-based functions (spectral, coherence, LFP) use
`rustfft` crate. Custom Jacobi eigendecomposition and Gauss-Jordan
elimination avoid LAPACK dependency.

#### P0-B: Spike Codecs — `spike_codec/` (8 modules, 55 functions, 2,165 lines)

| Module | Lines | Fns | Key functions |
|--------|------:|----:|---------------|
| `codec.py` | 337 | 12 | rate_encode, temporal_encode, burst_encode, ... |
| `aer_codec.py` | 194 | 3 | aer_encode, aer_decode, aer_merge |
| `delta_codec.py` | 185 | 3 | delta_encode, delta_decode, delta_stats |
| `entropy.py` | 199 | 5 | spike_entropy, conditional_entropy, ... |
| `predictive_codec.py` | 564 | 14 | PredictiveEncoder, PredictiveDecoder, ... |
| `registry.py` | 109 | 3 | register_codec, get_codec, list_codecs |
| `streaming_codec.py` | 249 | 7 | StreamingEncoder, StreamingDecoder, ... |
| `waveform_codec.py` | 328 | 8 | WaveformEncoder, WaveformDecoder, ... |

**Rust approach:** `engine/src/codecs/` module. Stateful encoder/decoder
structs with PyO3 class wrappers. AER encode/decode is pure integer bit
manipulation — perfect for Rust.

**Expected speedup:** 5–50× (predictive codec has inner loops over spike
history; streaming codec has per-sample processing).

---

### P1 — Real-time / Inner Loop

These modules are performance-sensitive but called less frequently than P0.

#### P1-A: Audio Engines — `audio/` (4 modules, 42 functions, 1,221 lines)

| Module | Lines | Fns | Description |
|--------|------:|----:|-------------|
| `ssgf_engine.py` | 293 | 9 | SSGF phase field engine |
| `evs_engine.py` | 329 | 14 | Entrainment via sound |
| `adaptive_engine.py` | 396 | 14 | Adaptive entrainment |
| `user_profile.py` | 203 | 5 | User preference model |

**Rust approach:** `engine/src/audio/`. Real-time audio requires <10ms
latency per buffer. SSGF engine already has Kuramoto in Rust — extend
with phase field and entrainment dynamics.

#### P1-B: Network Projection — `network/projection.py` (314 lines, 11 functions)

CSR-format synaptic scatter. Called every network timestep for every
projection. Currently numpy vectorised but with Python loop over projections.

**Rust approach:** Extend `NetworkRunner` with CSR scatter dispatch.
Already partially implemented for `BrunelNetwork`.

#### P1-C: Online Learning — `learning/advanced.py` (436 lines, 21 functions)

STDP, e-prop, R-STDP, homeostatic, STP, structural plasticity.
Called per-synapse per-spike — O(N²) in worst case.

**Rust approach:** `engine/src/plasticity/`. `StdpSynapse` already exists
in Rust. Extend with e-prop eligibility traces and homeostatic scaling.

#### P1-D: Bitstream Utilities — `utils/bitstreams.py` (405 lines, 17 functions)

Bitstream generation, correlation analysis, stochastic number conversion.
Hot path for SC pipeline operations.

**Rust approach:** Extend existing `pack_bitstream`/`popcount` with
higher-level operations (correlation, conversion, bipolar encoding).

---

### P2 — Important but not real-time

#### P2-A: Equation Compiler — `compiler/equation_compiler.py` (635 lines, 22 functions)

Symbolic math → ODE compilation. Called once per model setup, not per-step.
Still valuable to Rustify for batch compilation scenarios.

**Rust approach:** `engine/src/compiler/`. Symbolic expression tree in Rust,
compile to optimised step functions.

#### P2-B: SNN Training Modules — `training/snn_modules.py` (657 lines, 41 functions)

PyTorch-integrated SNN layers. GPU-accelerated via CUDA — Rust path is
for CPU fallback and inference-only deployments.

**Rust approach:** `engine/src/training/`. Surrogate gradient LIF layers
with autograd support via `tch-rs` or standalone inference path.

#### P2-C: SCPN Holonomic Adapters — `adapters/holonomic/` (20 files, 2,351 lines, 110 functions)

16 SCPN layer adapters (L1–L16). Each adapter maps domain-specific
quantities to neuron parameters. Compute varies: L1 (quantum noise)
is light, L8 (phase field) is heavy.

**Rust approach:** `engine/src/adapters/`. Priority sub-ordering:
L8 (phase field) > L9 (memory) > L16 (director) > others.

#### P2-D: Spike Profiler — `profiling/spike_profiler.py` (401 lines, 10 functions)

Energy estimation, latency profiling, throughput measurement.
Called post-simulation for analysis.

**Rust approach:** `engine/src/profiling/`. Energy models are simple
arithmetic — straightforward port.

---

### P3 — Utilities and Low-Priority

~307 remaining modules covering:

| Category | Modules | Lines | Notes |
|----------|---------|-------|-------|
| `studio/` (Visual SNN Studio) | 12 | ~4K | UI/Flask — not compute |
| `quantum/` | 7 | ~1.5K | Qiskit integration — GPU path |
| `generative/` | 3 | ~1.2K | Text/3D gen — GPU path |
| `nir_bridge/` | 3 | ~1.3K | NIR format conversion — I/O |
| `datasets/` | 2 | ~500 | Data loading — I/O bound |
| `viz/` | 3 | ~600 | Matplotlib wrappers |
| `cli.py` | 1 | 551 | CLI interface |
| `doctor/` | 1 | 324 | Diagnostic tool |
| `export/` | 1 | ~200 | ONNX export |
| `interfaces/` (BCI, DVS, CCW) | 3 | ~700 | Hardware I/O |
| `hdl_gen/` | 2 | ~400 | Verilog/SPICE — already has IR |
| Other (privacy, sleep, swarm, ...) | ~270 | ~20K | Mixed compute/config |

**Strategy:** Rustify compute-heavy functions inside these modules
opportunistically. Many are I/O bound, GPU-accelerated, or
configuration code where Rust provides no benefit.

---

## Execution Order

| Phase | Scope | Est. Rust Lines | Deliverable |
|-------|-------|----------------|-------------|
| **P0-A** | spike_stats (22 modules, 142 fns) | ~3,000 | `engine/src/analysis/` |
| **P0-B** | spike_codec (8 modules, 55 fns) | ~2,000 | `engine/src/codecs/` |
| **P1-A** | audio engines (4 modules, 42 fns) | ~1,200 | `engine/src/audio/` |
| **P1-B** | projection CSR scatter | ~300 | extend `NetworkRunner` |
| **P1-C** | learning rules (21 fns) | ~500 | `engine/src/plasticity/` |
| **P1-D** | bitstream utils (17 fns) | ~400 | extend `engine/src/lib.rs` |
| **P2-A** | equation compiler (22 fns) | ~600 | `engine/src/compiler/` |
| **P2-B** | SNN training (41 fns) | ~800 | `engine/src/training/` |
| **P2-C** | holonomic adapters (110 fns) | ~2,000 | `engine/src/adapters/` |
| **P2-D** | spike profiler (10 fns) | ~300 | `engine/src/profiling/` |
| **P3** | remainder (opportunistic) | ~5,000 | various |

**Total estimated new Rust:** ~16,000 lines (doubling the engine).

---

## Implementation Pattern

For each module:

1. **Read Python source** — understand exact semantics, edge cases
2. **Write Rust implementation** in `engine/src/{category}/{module}.rs`
3. **PyO3 wrapper** — numpy array input via `PyReadonlyArray1`, return `Vec<f64>` or `PyArray`
4. **Register in `lib.rs`** — add to PyO3 module
5. **Auto-dispatch in Python** — `__init__.py` tries Rust, falls back to Python
6. **Parity test** — parametrised test: same input → same output (±f64 epsilon)
7. **Benchmark** — measure speedup vs Python baseline
8. **Update doc** — add Rust throughput to pipeline verification section

---

## Success Criteria

- [ ] Every public function in P0–P2 has a Rust implementation
- [ ] Parity tests pass for all Rust functions (exact or within 1e-10)
- [ ] Python auto-dispatch prefers Rust when engine is built
- [ ] `SC_NEUROCORE_NO_RUST=1` env var disables Rust backend
- [ ] Benchmarks documented in each module's pipeline doc
- [ ] CI runs both Python-only and Rust-accelerated test suites

---

## Tracking

Progress is tracked per-module. Mark ✅ when Rust impl + parity test + doc update are all done.

### P0-A: spike_stats

| Module | Fns | Rust | Parity | Doc |
|--------|----:|------|--------|-----|
| basic | 5 | ☐ | ☐ | ☐ |
| rate | 3 | ☐ | ☐ | ☐ |
| variability | 18 | ☐ | ☐ | ☐ |
| correlation | 15 | ☐ | ☐ | ☐ |
| distance | 16 | ☐ | ☐ | ☐ |
| information | 10 | ☐ | ☐ | ☐ |
| spectral | 1 | ☐ | ☐ | ☐ |
| temporal | 4 | ☐ | ☐ | ☐ |
| patterns | 3 | ☐ | ☐ | ☐ |
| dimensionality | 3 | ☐ | ☐ | ☐ |
| causality | 8 | ☐ | ☐ | ☐ |
| decoding | 5 | ☐ | ☐ | ☐ |
| network | 4 | ☐ | ☐ | ☐ |
| point_process | 4 | ☐ | ☐ | ☐ |
| surrogates | 10 | ☐ | ☐ | ☐ |
| statistics | 1 | ☐ | ☐ | ☐ |
| stimulus | 5 | ☐ | ☐ | ☐ |
| waveform | 6 | ☐ | ☐ | ☐ |
| sorting_quality | 10 | ☐ | ☐ | ☐ |
| lfp | 3 | ☐ | ☐ | ☐ |
| gpfa | 5 | ☐ | ☐ | ☐ |
| spade | 3 | ☐ | ☐ | ☐ |

### P0-B: spike_codec

| Module | Fns | Rust | Parity | Doc |
|--------|----:|------|--------|-----|
| codec | 12 | ☐ | ☐ | ☐ |
| aer_codec | 3 | ☐ | ☐ | ☐ |
| delta_codec | 3 | ☐ | ☐ | ☐ |
| entropy | 5 | ☐ | ☐ | ☐ |
| predictive_codec | 14 | ☐ | ☐ | ☐ |
| registry | 3 | ☐ | ☐ | ☐ |
| streaming_codec | 7 | ☐ | ☐ | ☐ |
| waveform_codec | 8 | ☐ | ☐ | ☐ |

### P1

| Module | Fns | Rust | Parity | Doc |
|--------|----:|------|--------|-----|
| audio/ssgf_engine | 9 | ☐ | ☐ | ☐ |
| audio/evs_engine | 14 | ☐ | ☐ | ☐ |
| audio/adaptive_engine | 14 | ☐ | ☐ | ☐ |
| audio/user_profile | 5 | ☐ | ☐ | ☐ |
| network/projection | 11 | ☐ | ☐ | ☐ |
| learning/advanced | 21 | ☐ | ☐ | ☐ |
| utils/bitstreams | 17 | ☐ | ☐ | ☐ |

### P2

| Module | Fns | Rust | Parity | Doc |
|--------|----:|------|--------|-----|
| compiler/equation_compiler | 22 | ☐ | ☐ | ☐ |
| training/snn_modules | 41 | ☐ | ☐ | ☐ |
| adapters/holonomic (20 files) | 110 | ☐ | ☐ | ☐ |
| profiling/spike_profiler | 10 | ☐ | ☐ | ☐ |
