<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Rust benchmark documentation -->
# Rust Engine Benchmarks

All measurements via Criterion 0.8, single-threaded, pure CPU.
Hardware: 11th Gen Intel Core i5-11600K @ 3.90 GHz (6C/12T), DDR4-2400, Ubuntu 24.04.
Verified via `lscpu` on 2026-05-31.

Last updated: 2026-07-13 for the DPI row; other rows retain their stated
historical measurement dates.

## How to Run

```bash
# Full benchmark suite
cargo bench --bench full_bench
cargo bench --bench analysis_bench

# Quick (fewer iterations)
cargo bench --bench full_bench -- --quick
cargo bench --bench analysis_bench -- --quick
```

---

## Core Engine (`full_bench`)

### Bitstream Operations

| Benchmark | Median |
|-----------|-------:|
| pack_1m | 811 µs |
| pack_fast_1m | 477 µs |
| pack_dispatch_1m (SIMD) | 34.4 µs |
| popcount_portable_1m | 29.6 µs |
| popcount_simd_1m | 6.13 µs |
| bernoulli_stream_1024 | 4.25 µs |
| bernoulli_packed_1024 | 3.97 µs |
| encoder_64k_steps | 281 µs |

### Fixed-Point LIF Neuron

| Benchmark | Median |
|-----------|-------:|
| lif_10k_steps | 56.3 µs |
| lif_100k_steps | 758 µs |

### Dense Layer / Graph / Attention

| Benchmark | Median |
|-----------|-------:|
| dense_forward_64x32 | 993 µs |
| mixed_dense_q88_q1616_64x32 | 2.245 µs |
| block_floating_dense_q16_64x32 | 10.056 µs |
| mixed_dense_q88_q1616_trap_64x32 | 2.325 µs |
| block_floating_dense_q16_trap_64x32 | 8.669 µs |
| mixed_dense_q88_q1616_envelope_64x32 | 2.322 µs |
| block_floating_dense_q16_envelope_64x32 | 8.748 µs |
| dcls_q88_tent_kernel_16tap | 40.184 ns/sample |
| ultrascale_plus_target_contract_64x32 | 130.836 µs/emit |
| ultrascale_plus_dense_fold_plan_64x32 | 6.661 ns/plan |
| attention_10x16_20x32 | 88.5 µs |
| gnn_20x8_forward | 85.3 µs |

The 2026-06-04 dense precision rows were captured on a workstation under
concurrent load and without exclusive CPU core isolation.  Use the medians as
local regression context and parity evidence.  Before citing them as production
throughput, rerun the benchmark on isolated cores and record CPU affinity,
host-load, governor, and frequency evidence in the raw JSON artefact.

`mixed_dense_q88_q1616_64x32` was measured on 2026-06-04 with
`taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_mixed_dense`.
The committed raw artefact is
`benchmarks/results/local_rust_2026-06-04_mixed_dense.json`; the matching Python
reference artefact is `benchmarks/results/local_python_2026-06-04_mixed_dense.json`.
Both artefacts record safe-workload overflow count `0` and saturating-probe
overflow count `32` for parity with the mixed-dense HDL `overflow_vector`.
They also record conservative envelope telemetry for comparison with the HDL
`abs_bounds_q1616` lanes: Python and Rust safe max absolute bound `531400`,
and saturating-probe max bound `17454214414336`.  The Python benchmark uses
`QFormatMixed(scale_per_tensor=False)` here so the comparison is the same raw
Q8.8/Q16.16 physical contract implemented by Rust and HDL.

`block_floating_dense_q16_64x32` was measured on 2026-06-04 with
`taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_block_floating_dense`.
The committed raw artefact is
`benchmarks/results/local_rust_2026-06-04_block_floating_dense.json`; the
matching Python reference artefact is
`benchmarks/results/local_python_2026-06-04_block_floating_dense.json`.
Both artefacts record safe-workload overflow count `0` and saturating-probe
overflow count `32` for parity with the block-floating HDL `overflow_vector`.
They also record conservative envelope telemetry for comparison with the HDL
`abs_bounds_q1616` lanes.  The Python and Rust block-floating artefacts now use
the same physical BFP workload: mantissa checksum `-15`, exponent checksum `0`,
safe max absolute bound `610816`, and saturating-probe max bound
`1125865547104256`.

The precision trap rows were measured on 2026-06-04 with
`taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_precision_traps`.
The committed raw artefact is
`benchmarks/results/local_rust_2026-06-04_precision_traps.json`; the matching
Python reference artefact is
`benchmarks/results/local_python_2026-06-04_precision_traps.json`.  Both Rust
trap workloads reported 32 saturated outputs on the deterministic 64×32
overflow workload.

The precision envelope rows were measured on 2026-06-04 with
`taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_precision_envelopes`.
The committed raw artefact is
`benchmarks/results/local_rust_2026-06-04_precision_envelopes.json`; the
matching Python reference artefact is
`benchmarks/results/local_python_2026-06-04_precision_envelopes.json`.  Both
Rust envelope workloads reported `conservative_overflow_free=true` and matched
the Python maximum absolute bounds for the deterministic 64×32 safe workload.

`dcls_q88_tent_kernel_16tap` was measured on 2026-06-04 with
`taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_dcls_q88`.
The committed raw artefact is
`benchmarks/results/local_rust_2026-06-04_dcls_q88.json`; the matching
Python/PyTorch/SystemVerilog artefact is
`benchmarks/results/local_python_2026-06-04_dcls_q88.json`.  The Rust path
reported `overflow_count=0` and `active_tap_total=2808700`, matching the
Python reference workload and DCLS Q8.8 saturation contract.

`ultrascale_plus_target_contract_64x32` was measured on 2026-06-04 with the
Rust benchmark process pinned to CPUs 10-11 inside a runtime cpuset shield. The
committed raw artefact is
`benchmarks/results/local_rust_2026-06-04_ultrascale_plus_target.json`; the
matching Python/Vivado-Tcl artefact is
`benchmarks/results/local_python_2026-06-04_ultrascale_plus_target.json`. The
Rust target report deliberately records `fits_dsp_budget=false`: a 64x32
one-DSP-per-MAC dense graph estimates `2048` DSPs against the ZU3EG budget of
`360`, so ZU3EG deployment needs a folded/time-multiplexed implementation or a
larger target before any timing-closure claim is valid.

`ultrascale_plus_dense_fold_plan_64x32` was measured on 2026-06-04 with the
same runtime cpuset shield. The committed raw artefact is
`benchmarks/results/local_rust_2026-06-04_ultrascale_dense_folding.json`; the
matching Python/SystemVerilog artefact is
`benchmarks/results/local_python_2026-06-04_ultrascale_dense_folding.json`.
Both surfaces report the same ZU3EG fold plan: 320 DSPs per compute cycle,
five output rows per cycle, one input fold, seven output groups, and seven
compute cycles for the 64x32 dense contract.

### PRNG

| Benchmark | Median |
|-----------|-------:|
| prng_chacha_fill_1024 | 299 ns |
| prng_xoshiro_fill_1024 | 194 ns |

---

## Neuron Models (`full_bench`)

### Legacy Neurons (`neuron.rs`)

| Model | 1k steps | 10k steps | Per step |
|-------|----------|-----------|----------|
| Lapicque | 2.99 µs | 19.5 µs | **2.0 ns** |
| ExpIF | 25.0 µs | 237 µs | **24 ns** |
| AdEx | 29.1 µs | 291 µs | **29 ns** |

### Interneurons (`neurons/interneurons.rs`) — Interneuron model group

| Model | 1k steps | Per step | Sub-steps | Notes |
|-------|----------|----------|-----------|-------|
| VIP | 365 µs | **365 ns** | 4 | HH + A-type K+ |
| Martinotti | 530 µs | **530 ns** | 4 | Pospischil + M-current |
| SST+ | 552 µs | **552 ns** | 4 | Pospischil LTS + T-type + Ih |
| PV+ FS | 4.25 ms | **4.25 µs** | 50 | Wang-Buzsáki + Kv3.1 |
| Chandelier | 4.29 ms | **4.29 µs** | 50 | WB + Kv1 + Kv3.1 |
| Basket (cerebellar) | 5.60 ms | **5.60 µs** | 50 | WB + A-type + KCa |

> PV+/Chandelier/Basket use 50 sub-steps (dt=0.01 ms, 0.5 ms per call)
> for Wang-Buzsáki gating stability. SST/VIP/Martinotti use 4 sub-steps
> (dt=0.025 ms, 0.1 ms per call) with Pospischil-style gating.

### Sensory Neurons (`neurons/sensory.rs`) — Sensory model group

| Model | 10k steps | Per step | Type | Notes |
|-------|-----------|----------|------|-------|
| Retinal ganglion | 1.08 ms | **108 ns** | spiking | Pillow 2005 GLM (stim+history filters) |
| Inner hair cell | 407 µs | **40.7 ns** | graded | Meddis vesicle pool + CaV1.3 |
| Merkel cell | 312.93 µs | **31.29 ns** | spiking | exact slow adaptation |
| Rod photoreceptor | 663 µs | **66.3 ns** | graded | cGMP cascade + Ca²⁺-GC feedback |
| Nociceptor | 259.48 µs | **25.95 ns** | spiking | exact membrane/sensitisation relaxation |
| Pacinian corpuscle | 489.26 µs | **48.93 ns** | spiking | exact derivative-pressure adaptation |
| Olfactory receptor | 411 µs | **41.1 ns** | spiking | cAMP + Ca²⁺/CaM + PDE4 |

> Merkel cell remeasured 2026-05-31 after exact membrane/adaptation relaxation
> and fail-closed state guards.
> Pacinian corpuscle remeasured 2026-05-31 after exact derivative-pressure
> membrane/adaptation relaxation and fail-closed state guards.
> Nociceptor remeasured 2026-05-31 after exact membrane/sensitisation
> relaxation and fail-closed state guards.
> Remaining sensory rows retain their previously documented measurement date
> until each model is remeasured after its own hardening pass.

### Motor Neurons (`neurons/motor.rs`) — Motor model group

| Model | 1k steps | Per step | Sub-steps | Notes |
|-------|----------|----------|-----------|-------|
| Alpha motor | 6.48 ms | **6.48 µs** | 50 | WB + PIC (h_pic) + AHP + Ca²⁺ |
| Gamma motor | 161 µs (10k) | **16.1 ns** | 1 | LIF + adaptation |
| Upper motor | 601.68 µs | **601.68 ns** | 4 | Pospischil RS + Ca²⁺, exact gates + conductance membrane |
| Renshaw cell | 4.92 ms | **4.92 µs** | 50 | exact WB gates + adaptation |

> Renshaw cell remeasured 2026-05-31 after exact gate, adaptation, conductance integration, and fail-closed state guards.
| Motor unit | 327 µs (10k) | **32.7 ns** | 1 | exact LIF + force model |

> Motor unit remeasured 2026-05-31 after exact membrane, adaptation, force-decay, subtype-parity, and fail-closed state guards.

> Alpha motor is the most expensive per-step model due to WB gating (50 sub-steps),
> PIC evaluation, Ca2+ dynamics, and AHP computation at each sub-step.

### Cerebellar Neurons (`neurons/cerebellar.rs`) — Cerebellar model group

| Model | Steps | Median | Per step | Sub-steps | Notes |
|-------|-------|--------|----------|-----------|-------|
| Granule cell (D'Angelo 2001) | 10k | 7.64 ms | **764 ns** | 4 | exact gates + 8 conductances: Na, K_dr, K_A, Ca_T, K_Ca, Ih, leak, tonic GABA |
| Golgi cell (Solinas 2007) | 1k | 2.96 ms | **2.96 µs** | 10 | exact gates + 11 currents: Na_t, Na_p, K_dr, K_A, K_M, Ca_T, Ca_N, BK, SK, Ih, leak |

> Granule cell remeasured 2026-05-31 after exact gate, calcium, conductance integration, full polyglot parity, and fail-closed state guards.

> Golgi cell remeasured 2026-05-31 after exact gate, calcium, conductance integration, full polyglot parity, and fail-closed state guards.
| Stellate cell | 1k | 6.08 ms | **6.08 µs** | 50 | exact WB gates + exact Kv3.1 + conductance membrane |
| Lugaro cell | 10k | 230.56 µs | **23.06 ns** | 1 | exact LIF + adaptation + 5-HT |
| Unipolar brush cell | 10k | 211.47 µs | **21.15 ns** | 1 | LIF + persistent NMDA-like, exact relaxation |
| DCN neuron | 1k | 2.71 ms | **2.71 µs** | 20 | exact gates + 7 currents: Na_t, Na_p, K_dr, Ca_T, AHP, Ih, leak |

> Lugaro cell remeasured 2026-05-31 after exact membrane/adaptation relaxation,
> Python/Rust/Go/Julia/Rust-safety parity, Mojo metadata alignment, and
> fail-closed state guards.

> Stellate cell remeasured 2026-05-31 after exact Wang-Buzsáki h/n gates,
> exact Kv3.1 p-gate relaxation, exact conductance-form membrane integration,
> Python/Rust/Go/Julia/Rust-safety parity, and fail-closed state guards.

> DCN neuron remeasured 2026-05-31 after exact gate, calcium, conductance integration, full polyglot parity, and fail-closed state guards.

> Granule cell uses four exact-integration sub-steps for D'Angelo-style
> conductance dynamics and tonic GABA inhibition.

### Ion Channel Variant Neurons (`neurons/channels.rs`) — Ion-channel variant group

| Model | Steps | Median | Per step | Sub-steps | Notes |
|-------|-------|--------|----------|-----------|-------|
| Persistent Na+ | 1k | 4.61 ms | **4.61 µs** | 50 | WB + INaP subthreshold amplification |
| Ih (HCN) | 1k | 5.17 ms | **5.17 µs** | 50 | WB + Ih sag/rebound |
| T-type Ca²⁺ | 1k | 9.17 ms | **9.17 µs** | 50 | WB + IT rebound bursting |
| A-type K+ | 1k | 4.92 ms | **4.92 µs** | 50 | WB + IA onset delay |
| BK (Ca²⁺-K+) | 1k | 5.54 ms | **5.54 µs** | 50 | WB + BK fast AHP |
| SK (Ca²⁺-K+) | 1k | 4.35 ms | **4.35 µs** | 50 | WB + SK medium AHP |
| NMDA receptor | 1k | 4.81 ms | **4.81 µs** | 50 | WB + NMDA + Mg²⁺ block |

### Map Neurons (`neurons/maps.rs`) — Map-neuron model group

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Aihara map | 100k | 3.38 ms | **33.8 ns** | Chaotic sigmoid map |
| Kilinc-Bhatt map | 100k | 2.45 ms | **24.5 ns** | Adaptive threshold map |
| Ermentrout-Kopell | 100k | 2.90 ms | **29.0 ns** | Canonical Type I (theta) |

### Population / Mean-Field (`neurons/population.rs`) — Population and mean-field model group

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Montbrio-Pazo-Roxin | 100k | 1.57 ms | **15.7 ns** | Exact mean-field of QIF |
| Brunel balanced | 100k | 1.62 ms | **16.2 ns** | E/I balance, 2 rate ODEs |
| TUM (STP) | 100k | 3.03 ms | **30.3 ns** | Rate + depression + facilitation, 3 ODEs |
| El Boustani (NMDA) | 100k | 2.74 ms | **27.4 ns** | E/I + NMDA gating, 3 ODEs |

### Miscellaneous (`neurons/misc.rs`) — Miscellaneous model group

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Graded synapse | 100k | 1.23 ms | **12.3 ns** | Non-spiking, passive RC + release sigmoid |
| Gap junction | 100k | 2.96 ms | **29.6 ns** | LIF + electrical synapse + Cx36 rectification |
| FH axon (GHK) | 1k | 5.84 ms | **5.84 µs** | Myelinated nerve, GHK driving force, 50 sub-steps |
| Node of Ranvier (MRG) | 1k | 1.46 ms | **1.46 µs** | Nav1.6 + INaP + Kv7, 20 sub-steps |
| Myelinated axon | 1k | 1.40 ms | **1.40 µs** | MRG node + internode cable |
| Cardiac Purkinje | 1k | 1.05 ms | **1.05 µs** | DiFrancesco-Noble, 6 currents, 10 sub-steps |
| Smooth muscle | 1k | 198 µs | **198 ns** | CaL + BK + IP3R/SERCA, 4 sub-steps |
| Beta cell | 1k | 196 µs | **196 ns** | CaL + K_dr + K_ATP + K_Ca, 4 sub-steps |

### Biophysical Models (`neurons/biophysical/`)

| Model | Steps | Median | Per step | Sub-steps | Notes |
|-------|-------|--------|----------|-----------|-------|
| Hodgkin-Huxley 1952 | 1k | 13.3 ms | **13.3 µs** | 100 | Full 4-ODE HH with safe_rate kinetics |
| Wang-Buzsáki 1996 | 1k | 6.94 ms | **6.94 µs** | 50 | FS interneuron, m_inf (no m state) |
| Connor-Stevens 1977 | 1k | 71.13 ms | **71.13 µs** | 100 | Candidate-first RK4 A-type K+ model |
| Traub-Miles 1991 | 1k | 1.80 ms | **1.80 µs** | 10 | CA3 pyramidal + M-current |
| Mainen-Sejnowski 1996 | 1k | 1.86 ms | **1.86 µs** | 20 | Two-compartment (soma+axon) |
| Plant R15 1976 | 1k | 1.11 ms | **1.11 µs** | 5 | Aplysia parabolic burster + Ca²⁺/KCa |
| De Schutter-Bower 1994 | 1k | 775 µs | **775 ns** | 5 | Purkinje cell, Ca²⁺-dependent K |
| Golomb FS 2007 | 1k | 711 µs | **711 ns** | 10 | Kv3 fast-spiking interneuron |
| Pospischil 2008 | 1k | 686 µs | **686 ns** | 4 | Minimal HH, 5 cortical cell types |
| Destexhe 1993 | 1k | 654 µs | **654 ns** | 5 | Thalamocortical + T-current rebound |
| Hill-Tononi 2005 | 1k | 350 µs | **350 ns** | 1 | Na-dependent K, sleep/wake |
| Durstewitz 2000 | 1k | 149 µs | **149 ns** | 1 | PFC + D1 dopamine + NMDA Mg²⁺ block |
| Bertram 2000 | 1k | 132 µs | **132 ns** | 1 | Phantom burster, dual slow K |
| Av-Ron 1991 | 1k | 120 µs | **120 ns** | 1 | Cardiac ganglion, Type III burst |
| Yamada 1989 | 1k | 105 µs | **105 ns** | 1 | Subcritical Hopf burster |
| Huber-Braun 1998 | 1k | 71.4 µs | **71.4 ns** | 1 | Temperature-sensitive cold receptor |
| Prescott 2008 | 10k | 537 µs | **53.7 ns** | 1 | Type I/II/III excitability tuning |
| GLIF (Allen) | 10k | 363 µs | **36.3 ns** | 1 | LIF + threshold adapt + ASC |
| GIF population | 10k | 368 µs | **36.8 ns** | 1 | Escape-rate generalized IF |
| Mihalas-Niebur 2009 | 10k | 123 µs | **12.3 ns** | 1 | Generalized IF, 20 spike patterns |

> Models with sub-steps have larger per-step cost due to inner ODE integration
> loops. HH is most expensive (100 sub-steps at dt=0.01 ms).
> Measured 2026-04-05 on i5-11600K @ 3.90 GHz.

### Simple Spiking (`neurons/simple_spiking.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| SuperSpike | 10k | 17.8 µs | **1.8 ns** | Surrogate gradient LIF |
| Brunel-Wang (LIF+NMDA) | 10k | 23.4 µs | **2.34 ns** | AMPA + NMDA Mg²⁺ block + GABA |
| e-prop ALIF | 10k | 28.3 µs | **2.8 ns** | Adaptive LIF for e-prop learning |
| Resonate-and-Fire | 10k | 41.8 µs | **4.2 ns** | 2D subthreshold oscillator |
| Alpha synapse LIF | 10k | 53.1 µs | **5.3 ns** | Excitatory + inhibitory alpha |
| McKean | 10k | 320.24 µs | **32.0 ns** | RK4 piecewise-linear McKean |
| Hindmarsh-Rose | 10k | 77.4 µs | **7.7 ns** | 3D burster (x,y,z) |
| COBA LIF | 10k | 104 µs | **10.4 ns** | Conductance-based with g_e, g_i |
| FitzHugh-Nagumo | 10k | 508.24 µs | **50.824 ns** | RK4 two-state qualitative spike model |
| FitzHugh-Rinzel | 10k | 572.65 µs | **57.265 ns** | RK4 three-state burster with slow y |
| Wilson HR | 10k | 506.27 µs | **51.4 ns** | RK4 polynomial cortical |
| Benda-Herz | 10k | 218 µs | **21.8 ns** | Stochastic rate + adaptation |
| Learnable neuron | 10k | 224 µs | **22.4 ns** | Learnable parameters (tau, beta) |
| Pernarowski | 10k | 632.87 µs | **63.287 ns** | RK4 three-state beta-cell burster |
| Terman-Wang | 10k | 1.2380 ms | **123.80 ns** | RK4 LEGION relaxation oscillator |
| Gutkin-Ermentrout | 10k | 289 µs | **28.9 ns** | Type I excitability |
| Sherman-Rinzel-Keizer | 1k | 29.0 µs | **29.0 ns** | Beta cell burster |
| Chay-Keizer | 1k | 30.5 µs | **30.5 ns** | Beta cell with KCa |
| Chay | 1k | 32.8 µs | **32.8 ns** | Beta cell 3-variable |
| Morris-Lecar | 10k | 561 µs | **56.1 ns** | Ca/K 2D model |
| Butera respiratory | 1k | 341.62 µs | **341.62 ns** | bounded RK4 Pre-Bötzinger + INaP |

> Most simple spiking models in this table retain their previously documented
> integration contract. Butera respiratory was remeasured on 2026-05-31 after
> bounded RK4 hardening across Python, Rust, Go, Julia, and Rust safety
> surfaces. Older rows in this section retain their 2026-04-05 measurement
> date until their own model-specific hardening pass is completed.

### Rate / Mean-Field (`neurons/rate.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Threshold linear rate | 100k | 39.4 µs | **0.4 ns** | Rectified linear |
| Sigmoid rate | 100k | 921 µs | **9.2 ns** | Sigmoidal firing rate |
| TsodyksMarkram STP | 10k | 86.4 µs | **8.6 ns** | Short-term plasticity |
| Parallel spiking | 10k | 60.9 µs | **6.1 ns** | Multi-subunit LIF |
| Astrocyte | 10k | 190 µs | **19.0 ns** | Ca²⁺/IP3/SERCA dynamics |
| Compte WM | 10k | 232 µs | **23.2 ns** | Working memory NMDA |
| LTC | 10k | 409 µs | **40.9 ns** | Liquid time constant |
| FractionalLIF | 10k | 739 µs | **73.9 ns** | Fractional-order memory kernel |
| LeakyCompeteFire | 10k | 972 µs | **97.2 ns** | WTA lateral inhibition (4 units) |
| AmariNeuralField | 10k | 24.2 ms | **2.42 µs** | 32-unit neural field |
| Siegert | 100k | 44.4 ms | **444 ns** | Transfer function with erf() |

### Hardware Neuromorphic (`neurons/hardware.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| TrueNorth | 100k | 94.2 µs | **0.9 ns** | IBM integer LIF |
| SpiNNaker2 | 100k | 103 µs | **1.0 ns** | Fixed-point 3-compartment |
| Akida | 100k | 60.1 µs | **0.6 ns** | Event-driven threshold |
| Loihi CUBA | 100k | 342 µs | **3.4 ns** | Intel 2-variable integer IF |
| Loihi2 | 100k | 416 µs | **4.2 ns** | Intel 3-compartment integer |
| SpiNNaker LIF | 10k | 44.5 µs | **4.5 ns** | ARM LIF emulation |
| DPI | 100k | 5.28 ms | **52.8 ns** | Coupled membrane/AHP DPI, Eq. (2)–(3) feedback and pulse |
| BrainScaleS AdEx | 1k | 31.4 µs | **31.4 ns** | Analog accelerated AdEx |
| NeuroGrid | 1k | 43.6 µs | **43.6 ns** | Subthreshold analog, 2-comp |

The DPI row was remeasured after the 2010 coupled-circuit fidelity pass with
`cargo bench --no-default-features --bench full_bench dpi_100k_steps`. Criterion
reported a 95% interval of 5.1153–5.4610 ms for 100,000 steps on the loaded,
non-isolated workstation. It is a local regression measurement, not a reserved-
core throughput claim.

### AI-Optimised (`neurons/ai_optimized.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Meta-plastic | 10k | 20.9 µs | **2.1 ns** | Learning rate adaptation |
| Differentiable surrogate | 10k | 38.4 µs | **3.8 ns** | Surrogate gradient SNN |
| Multi-timescale | 10k | 99.1 µs | **9.9 ns** | Fast + slow dynamics |
| Predictive coding | 10k | 126 µs | **12.6 ns** | Prediction error driven |
| Attention-gated | 10k | 175 µs | **17.5 ns** | Gate modulates response |
| Compositional binding | 10k | 184 µs | **18.4 ns** | Phase-based variable binding |
| Self-referential | 10k | 404 µs | **40.4 ns** | Self-modifying tau |
| Continuous attractor | 10k | 6.58 ms | **658 ns** | 16-unit bump attractor |
| Arcane | 10k | 1.37 ms | **137 ns** | Deep accumulator + novelty |

### Multi-Compartment (`neurons/multi_compartment.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Dendrify | 1k | 20.2 µs | **20.2 ns** | Simplified dendritic |
| Two-compartment LIF | 10k | 26.9 µs | **2.7 ns** | Soma + dendrite LIF |
| Rall cable | 1k | 75.6 µs | **75.6 ns** | 5-compartment cable |
| Pinsky-Rinzel | 1k | 384 µs | **384 ns** | 2-comp pyramidal, 8-state RK4 † |
| Marder STG | 1k | 784 µs | **784 ns** | Stomatogastric, 7 currents, 13-state RK4 † |
| Booth-Rinzel | 1k | 253 µs | **253 ns** | Motoneuron soma+dendrite |
| Hay L5 pyramidal | 1k | 591 µs | **591 ns** | 3-comp (soma+trunk+apical) |

† Pinsky-Rinzel (eight-state) and Marder STG (thirteen-state) re-measured
2026-06-29 after their faithful RK4 rebuilds via a local release microbenchmark
(`sc-neurocore-safety` crate, identical kinetics), non-isolated on a loaded
workstation; an isolated criterion-harness rerun remains the production
reference. The RK4 step costs more than the prior single-step Euler (four
derivative evaluations per step, plus the larger faithful state vectors and
voltage-dependent rate functions). Other rows are earlier criterion medians.

### Maps — Additional (`neurons/maps.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Ibarz-Tanaka | 1k | 0.079827 ms | **79.8 ns** | Controlled PyO3 batch; source four-branch map |
| Cazelles | 100k | 955 µs | **9.6 ns** | Coupled map lattice |
| Medvedev | 500k | 10.799 ms | **21.6 ns** | Controlled PyO3 batch; slow-calcium first-return map |
| Courage-Nekorkin | 100k | 1.34 ms | **13.4 ns** | FHN-like map |
| Rulkov | 100k | 1.67 ms | **16.7 ns** | Slow-fast bursting map |
| Chialvo | 100k | 1.75 ms | **17.5 ns** | 2D excitable map |

The Medvedev row supersedes the old tent-map Criterion result. It comes from
the committed five-repeat controlled batch artefact rather than the older
Criterion run, so it must not be compared as if the harnesses were identical.
The source-derived recurrence produces 375,000 events in every Python, Rust,
Julia, Go, and Mojo lane over 500,000 iterations at `I=2`. Rust's recorded
median is 10.799 ms; the complete host metadata, source hashes, measured order,
and parity fields are in `benchmarks/results/bench_medvedev_map.json`.

The corrected Ibarz-Tanaka row supersedes the earlier rational/linear-kernel
Criterion result. It comes from the 21-call controlled parity artefact at
`I=0.2`, where all five lanes report 33 source reset events over 1,000
iterations. Rust's median is 0.079827 ms; the recorded process is CPU-pinned,
but the host is loaded and has no kernel-isolated CPU set. Source hashes and
the complete environment are in
`benchmarks/results/bench_ibarz_tanaka_map.json`.

### Sensory — Additional (`neurons/sensory.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Outer hair cell | 10k | 106 µs | **10.6 ns** | Prestin electromotility |
| Taste receptor | 10k | 120 µs | **12.0 ns** | Gustatory transduction |
| Cone photoreceptor | 10k | 135 µs | **13.5 ns** | Colour vision, cGMP cascade |

### Trivial IF Variants (`neurons/trivial.rs`)

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Perfect integrator | 100k | 206 µs | **2.1 ns** | Pure integration, simplest IF |
| KLIF | 100k | 224 µs | **2.2 ns** | Kernel-based LIF |
| Complementary LIF | 100k | 265 µs | **2.7 ns** | Dual-pathway (v_pos + v_neg) |
| Sigma-delta | 100k | 311 µs | **3.1 ns** | Spike-based A/D encoder |
| Integer QIF | 100k | 368 µs | **3.7 ns** | Integer arithmetic QIF |
| Inhibitory LIF | 100k | 447 µs | **4.5 ns** | Self-inhibiting trace |
| Gated LIF | 100k | 555 µs | **5.6 ns** | Input gating mechanism |
| Parametric LIF | 100k | 810 µs | **8.1 ns** | Learnable alpha, beta |
| Stochastic LIF | 10k | 85.1 µs | **8.5 ns** | Gaussian noise injection |
| Quadratic IF | 100k | 875 µs | **8.7 ns** | Quadratic subthreshold |
| Non-resetting LIF | 10k | 165 µs | **16.5 ns** | Threshold adaptation, no reset |
| Energy LIF | 10k | 183 µs | **18.3 ns** | Energy-budget constrained |
| MAT | 10k | 187 µs | **18.7 ns** | Multi-timescale adaptive threshold |
| SFA | 10k | 196 µs | **19.6 ns** | Spike frequency adaptation |
| NLIF | 10k | 261 µs | **26.1 ns** | Nonlinear leak + recovery |
| Adaptive threshold IF | 10k | 288 µs | **28.8 ns** | Dynamic threshold |
| Escape rate | 10k | 517 µs | **51.7 ns** | Stochastic, exp() hazard |
| Theta neuron | 100k | 6.97 ms | **69.7 ns** | Phase model, cos()/sin() |
| CFC | 100k | 11.6 ms | **116 ns** | Closed-form continuous, exp() |

> Trivial models are the fastest — no sub-stepping, minimal state.
> Measured 2026-04-05 on i5-11600K @ 3.90 GHz.

---

## Analysis Modules (`analysis_bench`)

22 modules, 84 benchmark points. Full results in
[rust-analysis-engine.md](rust-analysis-engine.md#benchmark-results).

### Highlights (fastest per category)

| Category | Function | Input | Median |
|----------|----------|-------|-------:|
| Basic | firing_rate | 100 spikes | 24 ns |
| Waveform | waveform_amplitude | 64 samples | 39 ns |
| Variability | fano_factor | 100 spikes | 98 ns |
| Basic | spike_times | 100 spikes | 142 ns |
| Distance | isi_distance | 100 spikes | 254 ns |
| Temporal | change_point_detection | 1K spikes | 347 ns |
| Decoding | bayesian_decode | 20n, 8 stim | 982 ns |
| Patterns | spike_directionality | 5K spikes | 68 µs |
| Spectral | power_spectrum | 100K samples | 5.7 ms |
| GPFA | gpfa | 4n, 500t, 5 iter | 5.8 ms |

### Scaling Characteristics

| Function | 100 → 100K | Scaling |
|----------|-----------|---------|
| spike_times | 142 ns → 102 µs | O(n) |
| power_spectrum | 4.0 µs → 5.7 ms | O(n log n) |
| sample_entropy | 46 µs → (n/a) | O(n²) |
| functional_connectivity | — | O(n² × T) |

---

## Benchmark Files

| File | Harness | Content |
|------|---------|---------|
| `engine/benches/full_bench.rs` | Criterion | Core + neurons (31 benchmarks) |
| `engine/benches/analysis_bench.rs` | Criterion | 22 analysis modules (84 benchmarks) |
| `engine/benches/bitstream_bench.rs` | Criterion | Bitstream-specific deep dive |
| `engine/benches/scaling_bench.rs` | Criterion | Network scaling tests |

## Criterion Output

JSON results stored in `engine/target/criterion/*/new/estimates.json`
after each run. Use `cargo-criterion` or the HTML reports in
`engine/target/criterion/report/` for trend analysis.
