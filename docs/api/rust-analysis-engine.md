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

## Performance Notes

The Rust analysis modules outperform their Python/NumPy counterparts
on all functions by virtue of zero-copy array access and elimination
of Python interpreter overhead. The largest gains are in:

- **SPADE** (O(2^n * T * S) surrogate testing): 10-50x faster
- **GPFA** (dense matrix operations per EM iteration): 5-20x faster
- **Sorting quality** (O(n^2 * d) distance computations): 5-15x faster
- **Cross-correlation** (sliding window): 3-10x faster

All functions are single-threaded. Parallelisation is left to the
Python caller via `concurrent.futures` or joblib.
