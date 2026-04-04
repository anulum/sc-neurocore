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
  `dimensionality.rs`, `causality.rs`): partial-pivot elimination on
  augmented [A|I] matrix.
- **Gaussian elimination solver** (`gpfa.rs`, `causality.rs`,
  `decoding.rs`): solves Ax = b with partial pivoting.
- **Complex matrix ops** (`causality.rs`): `C64` type with inverse,
  determinant, multiply for spectral Granger.

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

Measured with Criterion 0.8 on mining rig (i7-11700, DDR4-3200).
Values are median latency in nanoseconds. Hardware: 5x RX 6600 XT +
GTX 1060 (GPUs unused — pure CPU benchmarks).

### basic

| Function | 100 | 10K | 100K |
|----------|----:|----:|-----:|
| spike_times | 142 ns | 7.9 us | 102 us |
| isi | 176 ns | 8.5 us | 100 us |
| firing_rate | 24 ns | 1.6 us | 23 us |
| bin_spike_train | 62 ns | 3.6 us | 49 us |

### rate

| Function | 100 | 10K | 100K |
|----------|----:|----:|-----:|
| instantaneous_rate | 5.7 us | 446 us | 4.1 ms |

### variability

| Function | 100 | 10K | 100K |
|----------|----:|----:|-----:|
| cv_isi | 146 ns | 9.8 us | 125 us |
| fano_factor | 98 ns | 6.5 us | 75 us |
| sample_entropy | 46 us | 4.3 ms | O(n^2) — not benchmarked |

### correlation

| Function | 100 | 10K |
|----------|----:|----:|
| cross_correlation | 8.8 us | 1.5 ms |
| event_synchronization | 212 ns | 73 us |

### distance

| Function | 100 | 5K |
|----------|----:|---:|
| van_rossum | 1.4 us | 388 us |
| victor_purpura | 258 ns | 339 us |
| isi_distance | 254 ns | 8.3 us |

### information

| Function | 100 | 10K |
|----------|----:|----:|
| mutual_information | 1.0 us | 70 us |
| transfer_entropy | 1.3 us | 91 us |

### causality

| Function | 100 | 5K |
|----------|----:|---:|
| pairwise_granger | 157 ns | 113 us |

### decoding

| Function | Latency |
|----------|--------:|
| population_vector_decode (20n, 1000t) | 10.7 us |
| bayesian_decode (20n, 8 stim) | 982 ns |

### network

| Function | Latency |
|----------|--------:|
| functional_connectivity (10n, 2000t) | 4.3 ms |

### surrogates

| Function | 1K | 100K |
|----------|---:|-----:|
| isi_shuffle | 1.2 us | 139 us |
| homogeneous_poisson | 3.5 us | 316 us |

### temporal

| Function | 1K | 100K |
|----------|---:|-----:|
| burst_detection | 1.0 us | 131 us |
| change_point_detection | 347 ns | 39 us |

### patterns

| Function | Latency |
|----------|--------:|
| spike_directionality (5K) | 68 us |
| cubic_higher_order (5K, lag=20) | 2.2 ms |

### spectral

| Function | 256 | 10K | 100K |
|----------|----:|----:|-----:|
| power_spectrum | 4.0 us | 194 us | 5.7 ms |

### waveform (64 samples)

| Function | Latency |
|----------|--------:|
| waveform_width | 63 ns |
| waveform_amplitude | 39 ns |
| waveform_repolarization_slope | 75 ns |
| waveform_halfwidth | 216 ns |
| waveform_pt_ratio | 61 ns |

### point_process

| Function | 1K | 100K |
|----------|---:|-----:|
| conditional_intensity | 25 us | 3.0 ms |
| isi_hazard | 1.5 us | 126 us |

### statistics

| Function | Latency |
|----------|--------:|
| significance_bootstrap (200 surr) | 727 us |

### stimulus

| Function | 1K | 50K |
|----------|---:|----:|
| spike_triggered_average | 1.4 us | 85 us |
| spatial_information | 4.7 us | 224 us |

### lfp

| Function | 500 | 10K |
|----------|----:|----:|
| phase_locking_value | 34 us | 628 us |
| spike_field_coherence | 13 us | 330 us |

### sorting_quality

| Function | Latency |
|----------|--------:|
| isolation_distance (50 pts, 4D) | 8.9 us |
| isolation_distance (200 pts, 4D) | 33 us |
| silhouette_score (100 pts, 4D) | 40 us |
| silhouette_score (400 pts, 4D) | 544 us |
| isi_violation_rate (5K) | 4.2 us |

### dimensionality

| Function | Latency |
|----------|--------:|
| pca (10n, 2000t) | 58 us |
| factor_analysis (10n, 2000t, 20 iter) | 531 us |

### gpfa

| Function | Latency |
|----------|--------:|
| gpfa (4n, 500t, 5 EM iter) | 5.8 ms |

### spade

| Function | Latency |
|----------|--------:|
| spade_detect (3n, 500t, 50 surr) | 811 us |

## Performance Notes

All functions are single-threaded pure CPU. Benchmarks run via
`cargo bench --bench analysis_bench`. Parallelisation is left to the
Python caller via `concurrent.futures` or joblib.
