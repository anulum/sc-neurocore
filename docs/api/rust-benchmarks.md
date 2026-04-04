# Rust Engine Benchmarks

All measurements via Criterion 0.8, single-threaded, pure CPU.
Hardware: i5-11600K (6C/12T, 3.9 GHz), DDR4-2400, Ubuntu 24.04.

Last updated: 2026-04-04.

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
| pack_1m | 748 µs |
| pack_fast_1m | 423 µs |
| pack_dispatch_1m (SIMD) | 35.3 µs |
| popcount_portable_1m | 19.7 µs |
| popcount_simd_1m | 3.57 µs |
| bernoulli_stream_1024 | 4.15 µs |
| bernoulli_packed_1024 | 3.77 µs |
| encoder_64k_steps | 163 µs |

### Fixed-Point LIF Neuron

| Benchmark | Median |
|-----------|-------:|
| lif_10k_steps | 74.6 µs |
| lif_100k_steps | 569 µs |

### Dense Layer / Graph / Attention

| Benchmark | Median |
|-----------|-------:|
| dense_forward_64x32 | 869 µs |
| attention_10x16_20x32 | 62.8 µs |
| gnn_20x8_forward | 80.0 µs |

### PRNG

| Benchmark | Median |
|-----------|-------:|
| prng_chacha_fill_1024 | 323 ns |
| prng_xoshiro_fill_1024 | 197 ns |

---

## Neuron Models (`full_bench`)

### Legacy Neurons (`neuron.rs`)

| Model | 1k steps | 10k steps | Per step |
|-------|----------|-----------|----------|
| Lapicque | 2.13 µs | 21.4 µs | **2.1 ns** |
| ExpIF | 26.8 µs | 250 µs | **25 ns** |
| AdEx | 32.3 µs | 305 µs | **30 ns** |

### Interneurons (`neurons/interneurons.rs`) — Phase 3A

| Model | 1k steps | Per step | Sub-steps | Notes |
|-------|----------|----------|-----------|-------|
| VIP | 351 µs | **351 ns** | 4 | HH + A-type K+ |
| Martinotti | 505 µs | **505 ns** | 4 | Pospischil + M-current |
| SST+ | 586 µs | **586 ns** | 4 | Pospischil LTS + T-type + Ih |
| PV+ FS | 4.35 ms | **4.35 µs** | 50 | Wang-Buzsáki + Kv3.1 |
| Chandelier | 4.91 ms | **4.91 µs** | 50 | WB + Kv1 + Kv3.1 |
| Basket (cerebellar) | 4.91 ms | **4.91 µs** | 50 | WB + A-type + KCa |

> PV+/Chandelier/Basket use 50 sub-steps (dt=0.01 ms, 0.5 ms per call)
> for Wang-Buzsáki gating stability. SST/VIP/Martinotti use 4 sub-steps
> (dt=0.025 ms, 0.1 ms per call) with Pospischil-style gating.

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
