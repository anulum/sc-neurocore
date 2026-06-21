# Spike Train Analysis -- 124 Functions (23 Modules)

SC-NeuroCore includes a complete spike train analysis toolkit covering
every metric from Elephant, PySpike, SpikeInterface, and NeuroTools.
127 functions across 23 modules (22 spike_stats + 1 explainability).
Pure NumPy, zero external dependencies.

**Rust acceleration:** All 22 spike_stats modules have native Rust
implementations with 96 PyO3 bindings (597 tests). See
[Rust Analysis Engine](rust-analysis-engine.md) for the Rust API reference.

## Quick Start

```python
from sc_neurocore.analysis import (
    firing_rate, cv_isi, fano_factor, cross_correlation,
    power_spectrum, burst_detection, victor_purpura_distance,
    phase_locking_value, spade_detect, gpfa,
)
import numpy as np

# Generate a Poisson spike train
from sc_neurocore.analysis import homogeneous_poisson
train = homogeneous_poisson(rate_hz=50.0, duration_s=10.0)

# Basic statistics
print(f"Rate: {firing_rate(train):.1f} Hz")
print(f"CV(ISI): {cv_isi(train):.2f}")
print(f"Fano factor: {fano_factor(train):.2f}")
```

## Module Reference

### Basic (`spike_stats.basic`)

Spike extraction, ISI computation, rate, count, binning.

| Function | Description |
|----------|-------------|
| `spike_times(train, dt)` | Extract spike times (seconds) from a binary 0/1 array |
| `isi(train, dt)` | Inter-spike intervals (seconds) |
| `firing_rate(train, dt)` | Mean firing rate (Hz) |
| `spike_count(train)` | Total spike count |
| `bin_spike_train(train, bin_size)` | Bin a binary spike train into spike counts per bin |

### Variability (`spike_stats.variability`)

Spike train regularity, complexity, and fractal measures.

| Function | Description |
|----------|-------------|
| `cv_isi(train, dt)` | Coefficient of variation of ISI (CV=1 for Poisson, <1 regular) |
| `cv2(train, dt)` | Local CV2, less sensitive to rate changes (Holt et al. 1996) |
| `local_variation(train, dt)` | Local variation LV (Shinomoto et al. 2003) |
| `lvr(train, dt, refractoriness_ms)` | Revised local variation LvR corrected for refractoriness (Shinomoto et al. 2009) |
| `fano_factor(train, window_ms, dt)` | Fano factor: variance/mean of windowed spike counts |
| `isi_entropy(train, dt, bins)` | Shannon entropy of ISI distribution (bits) |
| `lempel_ziv_complexity(train)` | Lempel-Ziv 1976 complexity, normalised by N/log2(N) |
| `approximate_entropy(train, m, r_factor)` | Approximate entropy ApEn (Pincus 1991) |
| `sample_entropy(train, m, r_factor)` | Sample entropy SampEn, lower bias (Richman & Moorman 2000) |
| `permutation_entropy(train, order, delay)` | Bandt-Pompe permutation entropy, normalised to [0,1] |
| `hurst_exponent(train, min_window)` | Hurst exponent via detrended fluctuation analysis (Peng et al. 1994) |
| `allan_factor(train, dt, n_scales)` | Allan factor vs window size for fractal clustering detection |
| `rescaled_range(train, min_window)` | Hurst exponent via R/S analysis (Hurst 1951) |
| `complexity_pdf(train, dt, bins)` | ISI probability density function (Abeles 1982) |
| `optimal_bin_width(train, dt)` | Shimazaki-Shinomoto 2007 optimal histogram bin width |
| `optimal_kernel_bandwidth(train, dt)` | Silverman's rule-of-thumb bandwidth for ISI KDE |

### Rate Estimation (`spike_stats.rate`)

Instantaneous and population firing rate estimation.

| Function | Description |
|----------|-------------|
| `instantaneous_rate(train, dt, kernel, sigma_ms)` | Kernel-smoothed instantaneous rate (gaussian/exponential/rectangular) |
| `population_rate(trains, dt, sigma_ms)` | Population-level instantaneous rate across multiple neurons |
| `psth(trials, bin_ms, dt)` | Peri-stimulus time histogram across trials |

### Distance Metrics (`spike_stats.distance`)

Spike train distance and similarity measures.

| Function | Description |
|----------|-------------|
| `van_rossum_distance(a, b, dt, tau_ms)` | Van Rossum 2001 exponential-kernel distance |
| `victor_purpura_distance(times_a, times_b, cost_per_s)` | Victor-Purpura 1996 edit distance |
| `isi_distance(a, b, dt)` | ISI-distance ratio comparison (Kreuz et al. 2007) |
| `spike_distance(times_a, times_b, t_start, t_end)` | SPIKE-distance (Kreuz et al. 2013) |
| `spike_sync(times_a, times_b, t_start, t_end)` | SPIKE-synchronization, normalised to [0,1] (Kreuz et al. 2015) |
| `spike_sync_profile(times_a, times_b, n_bins, ...)` | Binned SPIKE-synchronization profile |
| `spike_profile(times_a, times_b, n_bins, ...)` | Binned SPIKE-distance profile |
| `isi_profile(a, b, dt, n_bins)` | Binned ISI-distance profile |
| `adaptive_spike_distance(times_a, times_b, ...)` | Adaptive SPIKE-distance with cost interpolation (Kreuz et al. 2013) |
| `schreiber_similarity(a, b, dt, sigma_ms)` | Smoothed Pearson correlation similarity (Schreiber et al. 2003) |
| `hunter_milton_similarity(times_a, times_b, dt_max)` | Nearest-neighbour coincidence fraction (Hunter-Milton 2003) |
| `earth_movers_distance(times_a, times_b, ...)` | EMD between spike time distributions (Rubner et al. 1998) |
| `multi_neuron_victor_purpura(times_list, cost)` | All-pairs Victor-Purpura distance matrix |
| `generalized_victor_purpura(times_a, times_b, cost_func)` | Victor-Purpura with arbitrary cost function |
| `spike_distance_matrix(times_list, metric, ...)` | All-pairs distance matrix (spike_distance/spike_sync/victor_purpura) |

### Correlation (`spike_stats.correlation`)

Cross-correlation, synchrony, and covariance measures.

| Function | Description |
|----------|-------------|
| `cross_correlation(a, b, max_lag_ms, dt)` | Cross-correlogram between two binary trains |
| `pairwise_correlation(trains, dt)` | Pairwise Pearson correlation matrix across neurons |
| `event_synchronization(a, b, dt, tau_ms)` | Quian Quiroga et al. 2002 event synchronization |
| `spike_train_coherence(a, b, dt)` | Magnitude-squared coherence between spike trains |
| `spike_time_tiling_coefficient(a, b, dt, delta_ms)` | STTC, corrects for rate bias (Cutts & Eglen 2014) |
| `covariance_matrix(trains, bin_size)` | Spike count covariance matrix (de la Rocha et al. 2007) |
| `autocorrelation_time(train, dt, max_lag_ms)` | Autocorrelation time until first zero crossing |
| `noise_correlation(trains, bin_size)` | Trial-to-trial variability correlation (Averbeck & Lee 2006) |
| `signal_correlation(trains, bin_size)` | Tuning similarity via Pearson correlation of mean responses |
| `spike_count_covariance(trains, window)` | Windowed spike count covariance (Kohn & Smith 2005) |
| `joint_psth(a, b, bin_size)` | Joint PSTH matrix (Aertsen et al. 1989) |
| `coincidence_index(a, b, dt, delta_ms)` | Rate-corrected coincidence index kappa (Joris et al. 2006) |

### Spectral (`spike_stats.spectral`)

Spectral analysis of spike trains.

| Function | Description |
|----------|-------------|
| `power_spectrum(train, dt)` | Power spectral density of a binary spike train |

### Temporal Patterns (`spike_stats.temporal`)

Burst detection, latency, onset, and change point analysis.

| Function | Description |
|----------|-------------|
| `burst_detection(train, dt, max_isi_ms, min_spikes)` | Detect bursts as consecutive short-ISI spikes |
| `first_spike_latency(train, dt)` | Time to first spike (seconds) |
| `response_onset(train, baseline_steps, dt, threshold_sigma)` | Detect response onset exceeding baseline + threshold |
| `change_point_detection(train, bin_size, threshold)` | CUSUM-based firing rate change points (Page 1954) |

### Stimulus (`spike_stats.stimulus`)

Spike-triggered analysis and receptive field estimation.

| Function | Description |
|----------|-------------|
| `spike_triggered_average(stimulus, train, window_steps)` | STA: average stimulus preceding each spike |
| `spike_triggered_covariance(stimulus, train, window_steps)` | STC: covariance of pre-spike stimulus (Schwartz et al. 2006) |
| `spatial_information(train, positions, n_bins, dt)` | Spatial information in bits/spike (Skaggs et al. 1993) |
| `place_field_detection(train, positions, ...)` | Detect place fields (O'Keefe & Dostrovsky 1971) |
| `tuning_curve(train, stimulus_values, n_bins, dt)` | Firing rate vs stimulus value (Dayan & Abbott 2001) |

### LFP Coupling (`spike_stats.lfp`)

Spike-LFP phase locking, coherence, and phase histograms.

| Function | Description |
|----------|-------------|
| `phase_locking_value(train, lfp)` | PLV via Hilbert phase at spike times |
| `spike_field_coherence(train, lfp, dt)` | Spike-field coherence SFC in frequency domain |
| `spike_phase_histogram(train, lfp, n_bins)` | Histogram of LFP phase at spike times |

### Surrogates (`spike_stats.surrogates`)

Surrogate spike train generation for significance testing.

| Function | Description |
|----------|-------------|
| `surrogate_isi_shuffle(train, seed)` | ISI-shuffled surrogate preserving rate + ISI distribution |
| `surrogate_dither(train, dither_ms, dt, seed)` | Spike jittering surrogate (+/- dither window) |
| `surrogate_trial_shuffle(trains, seed)` | Trial-order shuffling destroying trial-to-trial correlation |
| `homogeneous_poisson(rate_hz, duration_s, dt, seed)` | Generate homogeneous Poisson spike train |
| `inhomogeneous_poisson(rate_func, duration_s, dt, seed)` | Time-varying Poisson via thinning (Lewis & Shedler 1979) |
| `gamma_process(rate_hz, shape, duration_s, dt, seed)` | Gamma-renewal process (shape=1 Poisson, >1 regular) |
| `compound_poisson_process(rate_hz, burst_mean, ...)` | Compound Poisson with burst events (Snyder & Miller 1991) |
| `surrogate_joint_isi(train, seed)` | Joint-ISI surrogate preserving serial correlations (Louis et al. 2010) |
| `surrogate_bin_shuffling(train, bin_size, seed)` | Within-bin spike shuffling (Hatsopoulos et al. 2003) |
| `surrogate_spike_train_shifting(train, max_shift, seed)` | Circular shifting surrogate |

### Information Theory (`spike_stats.information`)

Information-theoretic measures for spike trains.

| Function | Description |
|----------|-------------|
| `mutual_information(a, b, bin_size)` | Mutual information between binned spike trains (bits) |
| `transfer_entropy(source, target, bin_size, lag)` | Transfer entropy from source to target (bits) |
| `spike_train_entropy(train, bin_size, word_length)` | Binary word entropy (Strong et al. 1998) |
| `noise_entropy(train, n_trials, bin_size, word_length)` | Noise entropy via pseudo-trials (de Ruyter van Steveninck et al. 1997) |
| `stimulus_specific_information(counts, stim_ids)` | SSI in bits (Butts 2003) |
| `kozachenko_leonenko_mi(x, y, k)` | k-NN mutual information estimator in nats (Kraskov et al. 2004) |
| `time_rescaling_ks_test(times, rate_func, ...)` | KS goodness-of-fit for point processes (Brown et al. 2002) |

### Causality (`spike_stats.causality`)

Granger causality and directed connectivity.

| Function | Description |
|----------|-------------|
| `pairwise_granger_causality(source, target, ...)` | Pairwise Granger causality log-likelihood ratio (Granger 1969) |
| `conditional_granger_causality(source, target, cond, ...)` | Conditional GC controlling for a third signal (Geweke 1984) |
| `spectral_granger_causality(trains, bin_size, order, ...)` | Frequency-domain GC via VAR model (Geweke 1982) |
| `partial_directed_coherence(trains, ...)` | PDC: normalised directed connectivity (Baccala & Sameshima 2001) |
| `directed_transfer_function(trains, ...)` | DTF: normalised transfer function (Kaminski & Blinowska 1991) |

### Dimensionality Reduction (`spike_stats.dimensionality`)

Low-dimensional projections of population activity.

| Function | Description |
|----------|-------------|
| `spike_train_pca(trains, n_components, bin_size)` | PCA on binned spike count matrix |
| `demixed_pca(trains_by_condition, n_components, ...)` | Demixed PCA separating condition variance (Kobak et al. 2016) |
| `factor_analysis(trains, n_factors, bin_size, n_iter)` | Factor analysis via EM (Rubin & Thayer 1982) |

### Decoding (`spike_stats.decoding`)

Neural population decoding algorithms.

| Function | Description |
|----------|-------------|
| `population_vector_decode(trains, directions, window)` | Georgopoulos population vector decoding |
| `bayesian_decode(counts, tuning_rates, prior)` | Bayesian MAP decoder (Dayan & Abbott 2001) |
| `maximum_likelihood_decode(counts, tuning_rates)` | Maximum likelihood Poisson decoder |
| `linear_discriminant_decode(data, labels, test)` | Fisher LDA decoder (Fisher 1936) |
| `naive_bayes_decode(data, labels, test)` | Gaussian naive Bayes decoder |

### Network (`spike_stats.network`)

Network-level analysis: connectivity, assemblies, synfire chains.

| Function | Description |
|----------|-------------|
| `functional_connectivity(trains, max_lag_ms, dt)` | Infer connectivity from peak cross-correlation |
| `unitary_events(trains, bin_size, alpha)` | Detect significant synchronous bins (Gruen et al. 2002) |
| `cell_assembly_detection(trains, bin_size, threshold)` | PCA-based cell assembly detection (Lopes-dos-Santos et al. 2013) |
| `synfire_chain_detection(trains, dt, max_delay_ms, ...)` | Cross-correlation peak ordering (Abeles 1991) |

### Point Process (`spike_stats.point_process`)

Point process models and hazard functions.

| Function | Description |
|----------|-------------|
| `conditional_intensity(train, dt, window_ms)` | Moving-window MLE conditional intensity (Brown et al. 2004) |
| `isi_hazard_function(train, dt, bins)` | ISI hazard function h(t) = f(t)/S(t) (Tuckwell 1988) |
| `isi_survivor_function(train, dt, bins)` | ISI survivor function S(t) = P(ISI > t) |
| `renewal_density(train, dt, bins)` | Renewal density normalised by mean rate (Cox 1962) |

### Sorting Quality (`spike_stats.sorting_quality`)

Spike sorting quality metrics.

| Function | Description |
|----------|-------------|
| `isolation_distance(cluster, noise, backend)` | Mahalanobis isolation distance (Harris et al. 2001) |
| `l_ratio(cluster, noise, backend)` | L-ratio cluster quality (Schmitzer-Torbert et al. 2005) |
| `silhouette_score(features, labels)` | Mean silhouette score (Rousseeuw 1987) |
| `d_prime(cluster_a, cluster_b)` | Sensitivity index between two clusters (Green & Swets 1966) |
| `isi_violation_rate(train, dt, refractory_ms)` | Fraction of ISIs below refractory period (Hill et al. 2011) |
| `presence_ratio(train, n_bins)` | Fraction of time bins with at least one spike (IBL 2019) |
| `amplitude_cutoff(amplitudes, bins)` | Estimated missing spike fraction (Hill et al. 2011) |
| `snr(waveforms)` | Signal-to-noise ratio of spike waveforms (Suner et al. 2005) |
| `nn_hit_rate(cluster, noise, k)` | k-NN cluster purity (Chung et al. 2017) |
| `drift_metric(waveforms, timestamps, n_bins)` | Waveform amplitude drift over time (IBL 2019) |

**Mahalanobis kernel.** `isolation_distance` and `l_ratio` share one numerically
optimal kernel for the squared Mahalanobis distance
`(x-μ)ᵀ Σ⁻¹ (x-μ)`. The regularised cluster covariance `Σ = L Lᵀ` is Cholesky-
factorised once and the quadratic form is obtained by a triangular solve
`L z = x-μ` followed by `Σ z²` — the covariance is never inverted explicitly,
which is both more accurate for ill-conditioned cluster covariances and cheaper
than forming `Σ⁻¹` and multiplying.

**Polyglot chain.** `isolation_distance(..., backend=...)` and
`l_ratio(..., backend=...)` run the same Cholesky kernel across five backends
(NumPy / Rust / Julia / Go / Mojo), agreeing with the NumPy reference to
floating-point round-off (~1e-13). `backend="auto"` prefers the Rust engine (the
always-available compiled path) and otherwise falls back to the NumPy reference;
`"python"`, `"rust"`, `"julia"`, `"go"` and `"mojo"` force a specific path. On
the reference workload (`benchmarks/bench_sorting_quality.py`, i5-11600K,
shielded cores 10–11; 64-point cluster, 256 noise points, 12 features) the
compiled backends beat NumPy — Mojo ~4.6×, Go ~3.8×, Rust ~1.9× — while the
Julia path is dominated by per-call interop overhead on this small workload
(~0.5×). Degenerate inputs (`n_cluster < 2`, fewer noise points than the cluster
size for the isolation distance, or empty noise for the L-ratio) return `nan`
before any backend runs.

### Waveform (`spike_stats.waveform`)

Spike waveform shape analysis.

| Function | Description |
|----------|-------------|
| `waveform_width(waveform, dt)` | Trough-to-peak width in seconds (Bartho et al. 2004) |
| `waveform_amplitude(waveform)` | Peak-to-trough amplitude |
| `waveform_repolarization_slope(waveform, dt)` | Max dV/dt after trough (Bean 2007) |
| `waveform_recovery_slope(waveform, dt)` | dV/dt during return to baseline (Bean 2007) |
| `waveform_halfwidth(waveform, dt)` | Duration at half-minimum amplitude (Bartho et al. 2004) |
| `waveform_pt_ratio(waveform)` | Peak-to-trough ratio (Bartho et al. 2004) |

### Statistics (`spike_stats.statistics`)

Significance testing.

| Function | Description |
|----------|-------------|
| `significance_bootstrap(func, a, b, n_surrogates, seed)` | Bootstrap permutation test for pairwise statistics |

### Patterns (`spike_stats.patterns`)

Spike directionality, ordering, and higher-order patterns.

| Function | Description |
|----------|-------------|
| `spike_directionality(times_a, times_b, ...)` | Asymmetry in [-1,1]: positive means A leads B (Kreuz et al. 2015) |
| `spike_train_order(times_list, ...)` | Pairwise directionality matrix (Kreuz et al. 2017) |
| `cubic_higher_order(train, dt, max_lag)` | Third-order cumulant C3(tau1, tau2) (Nikias & Petropulu 1993) |

### SPADE (`spike_stats.spade`)

Spike Pattern Detection and Evaluation (Torre et al. 2013).

| Function | Description |
|----------|-------------|
| `spade_detect(trains, bin_ms, dt, min_support, ...)` | Detect significant spatiotemporal spike patterns via frequent itemset mining + surrogate testing |

### GPFA (`spike_stats.gpfa`)

Gaussian Process Factor Analysis (Yu et al. 2009). Deterministic PCA
initialisation and a five-language EM chain (NumPy / Rust / Julia / Go / Mojo)
that agree to floating-point round-off. See **[GPFA -- deterministic init and
polyglot EM](gpfa.md)** for the algorithm, the backend contract and the
benchmark.

| Function | Description |
|----------|-------------|
| `gpfa(trains, n_latents, bin_ms, dt, max_iter, ..., backend)` | Extract smooth latent trajectories via EM with SE-GP priors |
| `gpfa_transform(new_trains, params, bin_ms, dt)` | Project new data using learned GPFA parameters |
| `gpfa_pca_init(Y, n_latents, bin_ms)` | Deterministic PCA initialisation shared by every backend |

## Explainability

::: sc_neurocore.analysis.explainability

## Integrated Information (Phi*)

`phi_star(...)` estimates geometric integrated information following
Barrett and Seth (2011). It compares Gaussian mutual information between
past and future whole-system states against contiguous bipartitions and
returns a non-negative reducibility estimate in nats. This is a tractable
analysis metric, not a claim about consciousness or intrinsic causal power.

Each Gaussian mutual information is a difference of covariance
**log-determinants** taken from Cholesky factors
(`MI = 0.5 (log|Cov_X| + log|Cov_Y| - log|Cov_XY|)`). Summing log-determinants is
the numerically stable form of the determinant ratio: the naive product of raw
determinants underflows for larger channel counts, where the log-determinant form
stays exact.

`phi_from_spike_trains(...)` bins binary spike trains into spike counts and
then applies `phi_star(...)` at the requested lag. The estimator is suitable
for small analysis windows and coarse-grained spike-train diagnostics; it is
not a full IIT minimum-information-partition search.

**Polyglot chain.** `phi_star(..., backend=...)` runs the same estimator across
five backends (NumPy / Rust / Julia / Go / Mojo), agreeing to floating-point
round-off (Rust, Julia, Go ~1e-15; Mojo ~1e-10). `backend="auto"` prefers the
Rust engine when present, otherwise the NumPy reference. On the reference
workload (`benchmarks/bench_phi.py`, i5-11600K, shielded cores) every compiled
backend beats NumPy: Rust ~7x, Julia ~4.4x, Mojo ~3.5x, Go ~3.0x.

The public contract is tested as follows:

- independent continuous channels produce near-zero Phi* within finite-sample
  tolerance;
- correlated channels produce positive Phi*;
- two-channel results are invariant to channel ordering;
- single-channel and too-short inputs return `0.0`;
- Phi* is always non-negative;
- correlated spike trains pass through the binning adapter;
- independent random spike trains remain low-Phi under the maintained bound;
- the Rust, Julia, Go and Mojo backends match the NumPy reference (gated parity
  tests).

::: sc_neurocore.analysis.phi_estimation.phi_star

::: sc_neurocore.analysis.phi_estimation.phi_from_spike_trains
