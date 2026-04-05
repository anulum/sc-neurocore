# Rust Engine Benchmarks

All measurements via Criterion 0.8, single-threaded, pure CPU.
Hardware: 11th Gen Intel Core i5-11600K @ 3.90 GHz (6C/12T), DDR4-2400, Ubuntu 24.04.
Verified via `lscpu` on 2026-04-04.

Last updated: 2026-04-05.

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
| attention_10x16_20x32 | 88.5 µs |
| gnn_20x8_forward | 85.3 µs |

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

### Interneurons (`neurons/interneurons.rs`) — Phase 3A

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

### Sensory Neurons (`neurons/sensory.rs`) — Phase 3B

| Model | 10k steps | Per step | Type | Notes |
|-------|-----------|----------|------|-------|
| Retinal ganglion | 1.08 ms | **108 ns** | spiking | Pillow 2005 GLM (stim+history filters) |
| Inner hair cell | 407 µs | **40.7 ns** | graded | Meddis vesicle pool + CaV1.3 |
| Merkel cell | 202 µs | **20.2 ns** | spiking | Slow adapting |
| Rod photoreceptor | 663 µs | **66.3 ns** | graded | cGMP cascade + Ca²⁺-GC feedback |
| Nociceptor | 68.6 µs | **6.9 ns** | spiking | Sensitisation |
| Pacinian corpuscle | 240 µs | **24.0 ns** | spiking | sin() input, fast adapting |
| Olfactory receptor | 411 µs | **41.1 ns** | spiking | cAMP + Ca²⁺/CaM + PDE4 |

> Sensory models use simple Euler integration (no sub-stepping).
> Measured 2026-04-05 on i5-11600K @ 3.90 GHz.

### Motor Neurons (`neurons/motor.rs`) — Phase 3C

| Model | 1k steps | Per step | Sub-steps | Notes |
|-------|----------|----------|-----------|-------|
| Alpha motor | 6.48 ms | **6.48 µs** | 50 | WB + PIC (h_pic) + AHP + Ca²⁺ |
| Gamma motor | 161 µs (10k) | **16.1 ns** | 1 | LIF + adaptation |
| Upper motor | 475 µs | **475 ns** | 4 | Pospischil RS + Ca²⁺ |
| Renshaw cell | 4.32 ms | **4.32 µs** | 50 | WB + adaptation |
| Motor unit | 180 µs (10k) | **18.0 ns** | 1 | LIF + force model |

> Alpha motor is the most expensive per-step model due to WB gating (50 sub-steps),
> PIC evaluation, Ca2+ dynamics, and AHP computation at each sub-step.

### Cerebellar Neurons (`neurons/cerebellar.rs`) — Phase 3D

| Model | Steps | Median | Per step | Sub-steps | Notes |
|-------|-------|--------|----------|-----------|-------|
| Granule cell (D'Angelo 2001) | 10k | 4.92 ms | **492 ns** | 4 | Full HH: 7 currents (Na, K_dr, K_A, Ca_T, K_Ca, Ih, leak) |
| Golgi cell (Solinas 2007) | 1k | 2.57 ms | **2.57 µs** | 10 | 11 currents: Na_t, Na_p, K_dr, K_A, K_M, Ca_T, Ca_N, BK, SK, Ih, leak |
| Stellate cell | 1k | 5.15 ms | **5.15 µs** | 50 | WB + Kv3.1 |
| Lugaro cell | 10k | 196 µs | **19.6 ns** | 1 | LIF + adaptation + 5-HT |
| Unipolar brush cell | 10k | 128 µs | **12.8 ns** | 1 | LIF + persistent NMDA-like |
| DCN neuron | 1k | 2.68 ms | **2.68 µs** | 20 | 7 currents: Na_t, Na_p, K_dr, Ca_T, AHP, Ih, leak |

> Granule cell uses simple Euler integration with T-type Ca2+ gating for
> rebound bursting. No sub-stepping needed.

### Ion Channel Variant Neurons (`neurons/channels.rs`) — Phase 3E

| Model | Steps | Median | Per step | Sub-steps | Notes |
|-------|-------|--------|----------|-----------|-------|
| Persistent Na+ | 1k | 4.61 ms | **4.61 µs** | 50 | WB + INaP subthreshold amplification |
| Ih (HCN) | 1k | 5.17 ms | **5.17 µs** | 50 | WB + Ih sag/rebound |
| T-type Ca²⁺ | 1k | 9.17 ms | **9.17 µs** | 50 | WB + IT rebound bursting |
| A-type K+ | 1k | 4.92 ms | **4.92 µs** | 50 | WB + IA onset delay |
| BK (Ca²⁺-K+) | 1k | 5.54 ms | **5.54 µs** | 50 | WB + BK fast AHP |
| SK (Ca²⁺-K+) | 1k | 4.35 ms | **4.35 µs** | 50 | WB + SK medium AHP |
| NMDA receptor | 1k | 4.81 ms | **4.81 µs** | 50 | WB + NMDA + Mg²⁺ block |

### Map Neurons (`neurons/maps.rs`) — Phase 3F

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Aihara map | 100k | 3.38 ms | **33.8 ns** | Chaotic sigmoid map |
| Kilinc-Bhatt map | 100k | 2.45 ms | **24.5 ns** | Adaptive threshold map |
| Ermentrout-Kopell | 100k | 2.90 ms | **29.0 ns** | Canonical Type I (theta) |

### Population / Mean-Field (`neurons/population.rs`) — Phase 3G

| Model | Steps | Median | Per step | Notes |
|-------|-------|--------|----------|-------|
| Montbrio-Pazo-Roxin | 100k | 1.57 ms | **15.7 ns** | Exact mean-field of QIF |
| Brunel balanced | 100k | 1.62 ms | **16.2 ns** | E/I balance, 2 rate ODEs |
| TUM (STP) | 100k | 3.03 ms | **30.3 ns** | Rate + depression + facilitation, 3 ODEs |
| El Boustani (NMDA) | 100k | 2.74 ms | **27.4 ns** | E/I + NMDA gating, 3 ODEs |

### Miscellaneous (`neurons/misc.rs`) — Phase 3H

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

### Biophysical Models (`neurons/biophysical.rs`)

| Model | Steps | Median | Per step | Sub-steps | Notes |
|-------|-------|--------|----------|-----------|-------|
| Hodgkin-Huxley 1952 | 1k | 13.3 ms | **13.3 µs** | 100 | Full 4-ODE HH with safe_rate kinetics |
| Wang-Buzsáki 1996 | 1k | 6.94 ms | **6.94 µs** | 50 | FS interneuron, m_inf (no m state) |
| Connor-Stevens 1977 | 1k | 3.56 ms | **3.56 µs** | 10 | A-type K+ for delay tuning |
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
| Brunel-Wang (LIF+NMDA) | 10k | 23.4 µs | **2.34 ns** | LIF + AMPA + NMDA Mg²⁺ block + GABA, 1 exp() |

> Brunel-Wang uses single Euler step with conductance-based synaptics.
> The exp() call for the Mg²⁺ block factor is the dominant cost.

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
