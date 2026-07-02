# Rust Analysis Engine -- 22 Modules, 597 Tests

Pure Rust implementations of all 22 `spike_stats` analysis modules,
exposed to Python via PyO3 bindings. Zero NumPy or SciPy dependency on the
Rust side -- all linear algebra (eigendecomposition, matrix inversion,
Gaussian elimination), FFT (via `rustfft`), and stochastic processes
(via `rand_distr`) are implemented natively.

## Architecture

```
engine/src/analysis/
  mod.rs              -- 22 pub mod declarations
  basic.rs            -- spike_times, isi, firing_rate, spike_count, bin_spike_train
  rate.rs             -- instantaneous_rate, psth, population_rate
  variability.rs      -- cv_isi, cv2, local_variation, fano_factor, entropies, Hurst
  correlation.rs      -- cross_correlation, STTC, coherence, covariance
  distance.rs         -- van_rossum, victor_purpura, ISI/SPIKE distances
  information.rs      -- mutual_information, transfer_entropy, KL MI
  causality.rs        -- pairwise/conditional/spectral Granger, PDC, DTF
  decoding.rs         -- population_vector, Bayesian, MLE, LDA, naive Bayes
  network.rs          -- functional_connectivity, unitary_events, assemblies
  surrogates.rs       -- ISI shuffle, dither, Poisson, gamma, compound Poisson
  temporal.rs         -- burst_detection, latency, onset, change points
  patterns.rs         -- spike_directionality, spike_train_order, C3
  spectral.rs         -- power_spectrum (FFT via rustfft)
  waveform.rs         -- width, amplitude, slopes, halfwidth, PT ratio
  point_process.rs    -- conditional_intensity, hazard, survivor, renewal
  statistics.rs       -- significance_bootstrap (Rust-only, takes fn pointer)
  stimulus.rs         -- STA, STC, spatial_information, place fields, tuning
  lfp.rs              -- phase_locking_value, spike_field_coherence, phase histogram
  sorting_quality.rs  -- isolation_distance, L-ratio, silhouette, d', SNR, drift
  dimensionality.rs   -- PCA, demixed PCA, factor analysis (Jacobi eigendecomp)
  gpfa.rs             -- GPFA via EM with squared-exponential GP priors
  spade.rs            -- SPADE: Apriori itemset mining + surrogate significance
```

## Dependency Map

External crates used by analysis modules:

| Crate | Version | Used by |
|-------|---------|---------|
| `rustfft` | 6.2 | `spectral.rs`, `lfp.rs`, `correlation.rs` |
| `rand_distr` | 0.6 | `surrogates.rs` |

All other computation (eigendecomposition, matrix inversion, complex
arithmetic, Hilbert transforms, histogram binning) is self-contained.

## Custom Linear Algebra

Several modules implement their own numerical primitives to avoid
external LAPACK/BLAS dependencies:

- **Jacobi eigendecomposition** (`dimensionality.rs`): symmetric
  eigenvalues + eigenvectors via iterative rotations, O(n^3) per sweep.
- **Gauss-Jordan inverse** (`sorting_quality.rs`, `gpfa.rs`,
  `dimensionality.rs`): partial-pivot elimination on
  augmented [A|I] matrix.
- **Gaussian elimination solver** (`gpfa.rs`, `decoding.rs`): solves Ax = b
  with partial pivoting.

The current Rust analysis crate does not export a benchmark-dispatched
causality backend. The generated `accel/rust/safety/causality.rs` file is a
non-authoritative safety-note surface; Granger, PDC, and DTF runtime behavior is
owned by the Python reference in `analysis/spike_stats/causality.py`.

## PyO3 Bindings

96 functions are exposed to Python via `#[pyfunction]` wrappers in
`engine/src/lib.rs`. The sole exception is `significance_bootstrap`
in `statistics.rs` which takes a `Fn(&[f64], &[f64]) -> f64` pointer
and cannot cross the PyO3 boundary.

### Functions NOT exposed via PyO3

| Function | Reason |
|----------|--------|
| `significance_bootstrap` | Takes `Fn` pointer |
| `generalized_victor_purpura` | Takes `fn(f64) -> f64` |
| `time_rescaling_ks_test` | Takes `fn(f64) -> f64` |
| `inhomogeneous_poisson` | Takes `fn(f64) -> f64` |
| `surrogate_trial_shuffle` | Returns permutation indices only |

These are callable from Rust code and tests but require Python-side
wrappers with closures.

## Module Reference

### basic

Core spike train operations used by most other modules.

| Function | Signature | Description |
|----------|-----------|-------------|
| `spike_times` | `(&[i32], f64) -> Vec<f64>` | Extract spike times (s) from binary array |
| `isi` | `(&[i32], f64) -> Vec<f64>` | Inter-spike intervals (s) |
| `firing_rate` | `(&[i32], f64) -> f64` | Mean firing rate (Hz) |
| `spike_count` | `(&[i32]) -> i64` | Total spike count |
| `bin_spike_train` | `(&[i32], usize) -> Vec<i64>` | Bin into spike counts |

### spectral

| Function | Signature | Description |
|----------|-----------|-------------|
| `power_spectrum` | `(&[i32], f64) -> (Vec<f64>, Vec<f64>)` | PSD via FFT, returns (psd, freqs_hz) |

### waveform

Spike waveform shape analysis. All functions take `&[f64]` waveform
samples and `dt` (sample interval, default 1/30000 s).

| Function | Returns | Reference |
|----------|---------|-----------|
| `waveform_width` | `f64` | Trough-to-peak width (s). Bartho et al. 2004 |
| `waveform_amplitude` | `f64` | Peak-to-trough amplitude |
| `waveform_repolarization_slope` | `f64` | Max dV/dt after trough. Bean 2007 |
| `waveform_recovery_slope` | `f64` | Min dV/dt after peak. Bean 2007 |
| `waveform_halfwidth` | `f64` | Duration at half-minimum. Bartho et al. 2004 |
| `waveform_pt_ratio` | `f64` | Post-trough peak / trough amplitude |

### point_process

Point process models and hazard functions.

| Function | Returns | Reference |
|----------|---------|-----------|
| `conditional_intensity` | `Vec<f64>` | Moving-window Poisson rate (Hz). Brown et al. 2004 |
| `isi_hazard_function` | `(Vec<f64>, Vec<f64>)` | h(t) = f(t)/S(t). Tuckwell 1988 |
| `isi_survivor_function` | `(Vec<f64>, Vec<f64>)` | S(t) = P(ISI > t) |
| `renewal_density` | `(Vec<f64>, Vec<f64>)` | Normalised by mean rate. Cox 1962 |

### statistics

| Function | Signature | Notes |
|----------|-----------|-------|
| `significance_bootstrap` | `(F, &[f64], &[f64], usize, u64) -> (f64, f64)` | **Rust-only.** Permutation test with splitmix64 PRNG |

### stimulus

Spike-triggered analysis and receptive field estimation.

| Function | Returns | Reference |
|----------|---------|-----------|
| `spike_triggered_average` | `Vec<f64>` | Mean pre-spike stimulus snippet |
| `spike_triggered_covariance` | `Vec<f64>` (flat matrix) | Covariance of pre-spike stimulus. Schwartz et al. 2006 |
| `spatial_information` | `f64` (bits/spike) | Skaggs et al. 1993 |
| `place_field_detection` | `Vec<(f64, f64)>` | Contiguous high-rate bins. O'Keefe & Dostrovsky 1971 |
| `tuning_curve` | `(Vec<f64>, Vec<f64>)` | Rate vs stimulus value. Dayan & Abbott 2001 |

### lfp

Spike-LFP coupling via analytic signal (Hilbert transform with `rustfft`).

| Function | Returns | Description |
|----------|---------|-------------|
| `phase_locking_value` | `f64` | PLV = |mean(exp(j * phase_at_spikes))| |
| `spike_field_coherence` | `(Vec<f64>, Vec<f64>)` | SFC = |S_xy|^2 / (S_xx * S_yy), plus freqs |
| `spike_phase_histogram` | `(Vec<i64>, Vec<f64>)` | Phase histogram in [-pi, pi] |

### sorting_quality

Spike sorting quality metrics. Cluster/noise inputs are row-major
`(n_points, n_features)` flat arrays.

| Function | Returns | Reference |
|----------|---------|-----------|
| `isolation_distance` | `f64` | Mahalanobis at rank n_cluster. Harris et al. 2001 |
| `l_ratio` | `f64` | Normalised chi2-approximation. Schmitzer-Torbert et al. 2005 |
| `silhouette_score` | `f64` | Mean (b-a)/max(a,b). Rousseeuw 1987 |
| `d_prime` | `f64` | Projected sensitivity index. Green & Swets 1966 |
| `isi_violation_rate` | `f64` | Fraction below refractory. Hill et al. 2011 |
| `presence_ratio` | `f64` | Fraction of occupied bins. IBL 2019 |
| `amplitude_cutoff` | `f64` | Estimated missing spikes. Hill et al. 2011 |
| `snr` | `f64` | Peak / noise_std. Suner et al. 2005 |
| `nn_hit_rate` | `f64` | k-NN cluster purity. Chung et al. 2017 |
| `drift_metric` | `f64` | Amplitude drift over time. IBL 2019 |

### dimensionality

Dimensionality reduction with built-in Jacobi eigendecomposition
(no LAPACK required).

| Function | Returns | Reference |
|----------|---------|-----------|
| `spike_train_pca` | `(Vec<f64>, Vec<f64>)` | (projected, explained_variance_ratio) |
| `demixed_pca` | `(Vec<f64>, Vec<f64>)` | Condition-dependent variance. Kobak et al. 2016 |
| `factor_analysis` | `(Vec<f64>, Vec<f64>)` | (loadings, uniquenesses). Rubin & Thayer 1982 |

### gpfa

Gaussian Process Factor Analysis. Full EM implementation with
squared-exponential GP kernels, block-structured precision matrices,
and approximate log-likelihood monitoring.

| Function | Returns | Reference |
|----------|---------|-----------|
| `gpfa` | `GpfaResult` struct | Trajectories, C, d, R, tau, log-likelihoods. Yu et al. 2009 |
| `gpfa_transform` | `Vec<f64>` | Project new data with learned parameters |

`GpfaResult` fields: `trajectories`, `c`, `d`, `r`, `tau`,
`log_likelihoods`, `n_latents`, `n_bins`, `n_neurons`.

### spade

Spike Pattern Detection and Evaluation. Apriori-style frequent
itemset mining extended to spatiotemporal patterns with lag search
and surrogate significance testing.

| Function | Returns | Reference |
|----------|---------|-----------|
| `spade_detect` | `Vec<SpadePattern>` | Significant patterns with p-values. Torre et al. 2013 |

`SpadePattern` fields: `neurons`, `lags`, `count`, `p_value`.

## Test Coverage

597 tests across all modules. Every public function has at least:

- Positive case (typical input producing expected output)
- Edge case (empty input, single element, degenerate dimensions)
- Numerical accuracy check (comparison against known values or bounds)

Run tests:

```bash
export PATH="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
cd engine && cargo test --lib
```

Run a single module's tests:

```bash
cargo test --lib analysis::spectral
cargo test --lib analysis::gpfa
```

## Benchmark Results

Measured with Criterion on i5-11600K @ 3.90 GHz, DDR4-3200.
Values are median latency. Pure CPU benchmarks.

Last measured: **2026-05-02** (`cargo bench --bench analysis_bench`)

### basic

| Function | 100 | 10K | 100K |
|----------|----:|----:|-----:|
| spike_times | 86 ns | 5.4 µs | 85.0 µs |
| isi | 106 ns | 5.6 µs | 86.3 µs |
| firing_rate | 13 ns | 1.2 µs | 15.3 µs |
| bin_spike_train | 36 ns | 2.5 µs | 29.0 µs |

### rate

| Function | 100 | 10K | 100K |
|----------|----:|----:|-----:|
| instantaneous_rate | 4.3 µs | 352 µs | 3.5 ms |

### variability

| Function | 100 | 10K | 100K |
|----------|----:|----:|-----:|
| cv_isi | 108 ns | 6.5 µs | 94.7 µs |
| fano_factor | 76 ns | 4.8 µs | 50.7 µs |
| sample_entropy | 32.6 µs | 3.23 ms | O(n²) — not benchmarked |

### correlation

| Function | 100 | 10K |
|----------|----:|----:|
| cross_correlation | 1.46 µs | 213 µs |
| event_synchronization | 142 ns | 63.1 µs |

### distance

| Function | 100 | 5K |
|----------|----:|---:|
| van_rossum | 1.07 µs | 314 µs |
| victor_purpura | 162 ns | 274 µs |
| isi_distance | 165 ns | 5.95 µs |

### information

| Function | 100 | 10K |
|----------|----:|----:|
| mutual_information | 714 ns | 48.7 µs |
| transfer_entropy | 1.03 µs | 60.8 µs |

### causality

No current Rust causality benchmark is dispatched. Use the Python reference for
Granger causality, partial directed coherence, and directed transfer functions.

### decoding

| Function | Latency |
|----------|--------:|
| population_vector_decode | 14.9 µs |
| bayesian_decode | 11.0 µs |

### network

| Function | Latency |
|----------|--------:|
| functional_connectivity (10n) | 814 µs |

### surrogates

| Function | 1K | 100K |
|----------|---:|-----:|
| isi_shuffle | 865 ns | 103 µs |
| homogeneous_poisson | 2.55 µs | 242 µs |

### temporal

| Function | 1K | 100K |
|----------|---:|-----:|
| burst_detection | 708 ns | 91.4 µs |
| change_point_detection | 296 ns | 31.8 µs |

### patterns

| Function | Latency |
|----------|--------:|
| spike_directionality (5K) | 3.85 µs |
| cubic_higher_order (5K) | 409 µs |

### spectral

| Function | 256 | 10K | 100K |
|----------|----:|----:|-----:|
| power_spectrum | 3.81 µs | 177 µs | 4.91 ms |

### waveform (50 samples)

| Function | Latency |
|----------|--------:|
| waveform_width | 64 ns |
| waveform_amplitude | 38 ns |
| waveform_repolarization_slope | 72 ns |
| waveform_halfwidth | 168 ns |
| waveform_pt_ratio | 58 ns |

### point_process

| Function | 1K | 100K |
|----------|---:|-----:|
| conditional_intensity | 22.9 µs | 2.33 ms |
| isi_hazard | 1.07 µs | 103 µs |

### statistics

| Function | Latency |
|----------|--------:|
| significance_bootstrap (200 surr) | 661 µs |

### stimulus

| Function | 1K | 50K |
|----------|---:|----:|
| STA | 1.18 µs | 67.8 µs |
| spatial_information | 3.89 µs | 205 µs |

### lfp

| Function | 500 | 10K |
|----------|----:|----:|
| phase_locking_value | 26.2 µs | 540 µs |
| spike_field_coherence | 11.9 µs | 237 µs |

### sorting_quality

| Function | Latency |
|----------|--------:|
| isolation_distance (50 pts, 4D) | 6.82 µs |
| isolation_distance (200 pts, 4D) | 27.0 µs |
| silhouette_score (50 pts, 4D) | 35.8 µs |
| silhouette_score (200 pts, 4D) | 536 µs |
| isi_violation_rate (5K) | 2.93 µs |

### dimensionality

| Function | Latency |
|----------|--------:|
| PCA (10n, 2000t) | 43.4 µs |
| factor_analysis (10n, 2000t) | 408 µs |

### gpfa

| Function | Latency |
|----------|--------:|
| GPFA (4n, 500t, 5 EM iter) | 4.84 ms |

### spade

| Function | Latency |
|----------|--------:|
| SPADE detect (3n, 500t, 50 surr) | 748 µs |

## Performance Notes

All functions are single-threaded pure CPU. Benchmarks run via
`cargo bench --bench analysis_bench`. Parallelisation is left to the
Python caller via `concurrent.futures` or joblib.

Full benchmark results with scaling analysis are maintained as internal release
evidence.  Public performance claims on this page are limited to the curated
benchmark tables above until the corresponding public benchmark packet is
published.
