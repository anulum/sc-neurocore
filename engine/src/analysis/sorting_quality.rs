// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike sorting quality metrics

use super::basic;
use nalgebra::DMatrix;

// ── helpers ─────────────────────────────────────────────────────────

/// Squared Mahalanobis distances `(x-μ)ᵀ Σ⁻¹ (x-μ)` of each row of `points`
/// (`n_pts × d`) from the cluster mean, using the regularised cluster
/// covariance `Σ`.
///
/// `Σ` is symmetric positive-definite (jitter-regularised in
/// [`covariance_matrix`]), so the quadratic form is evaluated through its
/// Cholesky factorisation — `Σ = L Lᵀ`, then `Σ X = D` is solved for the
/// centred points `D` and the distance is the column-wise dot product
/// `Dᵢᵀ Xᵢ`. The covariance is never inverted explicitly, which is both more
/// accurate and cheaper than forming `Σ⁻¹` and multiplying.
fn cluster_mahalanobis_sq(
    cluster: &[f64],
    n_cluster: usize,
    points: &[f64],
    n_pts: usize,
    d: usize,
) -> Vec<f64> {
    let mu = col_mean(cluster, n_cluster, d);
    let cov = covariance_matrix(cluster, n_cluster, d);
    let chol = DMatrix::<f64>::from_row_slice(d, d, &cov)
        .cholesky()
        .expect("cluster covariance must be symmetric positive-definite");

    // Centred points as columns of a `d × n_pts` matrix.
    let mut diffs = DMatrix::<f64>::zeros(d, n_pts);
    for i in 0..n_pts {
        for j in 0..d {
            diffs[(j, i)] = points[i * d + j] - mu[j];
        }
    }
    let solved = chol.solve(&diffs); // Σ⁻¹ · diffs, column-wise

    (0..n_pts)
        .map(|i| (0..d).map(|j| diffs[(j, i)] * solved[(j, i)]).sum::<f64>())
        .collect()
}

/// Covariance matrix (d x d) of row-major data (n x d), with regularisation.
fn covariance_matrix(data: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut mu = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            mu[j] += data[i * d + j];
        }
    }
    for v in &mut mu {
        *v /= n as f64;
    }
    let mut cov = vec![0.0f64; d * d];
    for i in 0..n {
        for j in 0..d {
            let dj = data[i * d + j] - mu[j];
            for k in j..d {
                let dk = data[i * d + k] - mu[k];
                cov[j * d + k] += dj * dk;
            }
        }
    }
    let denom = (n - 1).max(1) as f64;
    for j in 0..d {
        for k in j..d {
            cov[j * d + k] /= denom;
            cov[k * d + j] = cov[j * d + k];
        }
        // Regularise diagonal
        cov[j * d + j] += 1e-8;
    }
    cov
}

/// Mean of each column.
fn col_mean(data: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut mu = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            mu[j] += data[i * d + j];
        }
    }
    for v in &mut mu {
        *v /= n as f64;
    }
    mu
}

// ── public functions ──────────���─────────────────────────────────────

/// Isolation distance. Harris et al. 2001.
///
/// `cluster`: row-major `(n_cluster, n_features)`.
/// `noise`: row-major `(n_noise, n_features)`.
pub fn isolation_distance(
    cluster: &[f64],
    n_cluster: usize,
    noise: &[f64],
    n_noise: usize,
    n_features: usize,
) -> f64 {
    if n_cluster < 2 || n_noise < n_cluster {
        return f64::NAN;
    }
    let mut mah = cluster_mahalanobis_sq(cluster, n_cluster, noise, n_noise, n_features);
    mah.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if n_cluster - 1 < mah.len() {
        mah[n_cluster - 1]
    } else {
        *mah.last().unwrap()
    }
}

/// L-ratio. Schmitzer-Torbert et al. 2005.
pub fn l_ratio(
    cluster: &[f64],
    n_cluster: usize,
    noise: &[f64],
    n_noise: usize,
    n_features: usize,
) -> f64 {
    if n_cluster < 2 || n_noise == 0 {
        return f64::NAN;
    }
    let mah = cluster_mahalanobis_sq(cluster, n_cluster, noise, n_noise, n_features);
    let d = n_features as f64;
    let l_sum: f64 = mah
        .iter()
        .map(|&m| (-0.5 * (m.max(1e-10) - d)).exp().clamp(0.0, 1.0))
        .sum();
    l_sum / n_cluster as f64
}

/// Mean silhouette score. Rousseeuw 1987.
///
/// `features`: row-major `(n, d)`. `labels`: `(n,)`.
pub fn silhouette_score(features: &[f64], n: usize, d: usize, labels: &[i64]) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let mut classes: Vec<i64> = labels.to_vec();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return 0.0;
    }

    let mut scores = vec![0.0f64; n];
    for i in 0..n {
        let own = labels[i];
        // a_i: mean distance to own cluster
        let mut own_sum = 0.0;
        let mut own_count = 0usize;
        for j in 0..n {
            if labels[j] == own && j != i {
                own_sum +=
                    euclidean_dist(&features[i * d..(i + 1) * d], &features[j * d..(j + 1) * d]);
                own_count += 1;
            }
        }
        let a_i = if own_count > 0 {
            own_sum / own_count as f64
        } else {
            0.0
        };

        // b_i: min mean distance to any other cluster
        let mut b_i = f64::INFINITY;
        for &c in &classes {
            if c == own {
                continue;
            }
            let mut c_sum = 0.0;
            let mut c_count = 0usize;
            for j in 0..n {
                if labels[j] == c {
                    c_sum += euclidean_dist(
                        &features[i * d..(i + 1) * d],
                        &features[j * d..(j + 1) * d],
                    );
                    c_count += 1;
                }
            }
            if c_count > 0 {
                b_i = b_i.min(c_sum / c_count as f64);
            }
        }
        let denom = a_i.max(b_i).max(1e-30);
        scores[i] = (b_i - a_i) / denom;
    }
    scores.iter().sum::<f64>() / n as f64
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// d-prime (sensitivity index) between two clusters. Green & Swets 1966.
///
/// Both are row-major `(n, d)`.
pub fn d_prime(cluster_a: &[f64], n_a: usize, cluster_b: &[f64], n_b: usize, d: usize) -> f64 {
    if n_a == 0 || n_b == 0 || d == 0 {
        return 0.0;
    }
    let mu_a = col_mean(cluster_a, n_a, d);
    let mu_b = col_mean(cluster_b, n_b, d);
    let mut direction = vec![0.0f64; d];
    for j in 0..d {
        direction[j] = mu_b[j] - mu_a[j];
    }
    let norm: f64 = direction.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm < 1e-30 {
        return 0.0;
    }
    for v in &mut direction {
        *v /= norm;
    }
    // Project
    let proj_a: Vec<f64> = (0..n_a)
        .map(|i| {
            (0..d)
                .map(|j| cluster_a[i * d + j] * direction[j])
                .sum::<f64>()
        })
        .collect();
    let proj_b: Vec<f64> = (0..n_b)
        .map(|i| {
            (0..d)
                .map(|j| cluster_b[i * d + j] * direction[j])
                .sum::<f64>()
        })
        .collect();
    let mean_a: f64 = proj_a.iter().sum::<f64>() / n_a as f64;
    let mean_b: f64 = proj_b.iter().sum::<f64>() / n_b as f64;
    let var_a: f64 = proj_a.iter().map(|&v| (v - mean_a).powi(2)).sum::<f64>() / n_a as f64;
    let var_b: f64 = proj_b.iter().map(|&v| (v - mean_b).powi(2)).sum::<f64>() / n_b as f64;
    let pooled_std = (0.5 * (var_a + var_b)).sqrt();
    if pooled_std < 1e-30 {
        return 0.0;
    }
    (mean_a - mean_b).abs() / pooled_std
}

/// ISI violation rate: fraction of ISIs below refractory period. Hill et al. 2011.
pub fn isi_violation_rate(binary_train: &[i32], dt: f64, refractory_ms: f64) -> f64 {
    let intervals = basic::isi(binary_train, dt);
    if intervals.is_empty() {
        return 0.0;
    }
    let ref_sec = refractory_ms / 1000.0;
    let violations = intervals.iter().filter(|&&i| i < ref_sec).count();
    violations as f64 / intervals.len() as f64
}

/// Presence ratio: fraction of time bins with at least one spike. IBL 2019.
pub fn presence_ratio(binary_train: &[i32], n_bins: usize) -> f64 {
    let counts = basic::bin_spike_train(binary_train, binary_train.len().max(1) / n_bins.max(1));
    if counts.is_empty() {
        return 0.0;
    }
    counts.iter().filter(|&&c| c > 0).count() as f64 / counts.len() as f64
}

/// Amplitude cutoff estimate. Hill et al. 2011.
pub fn amplitude_cutoff(amplitudes: &[f64], bins: usize) -> f64 {
    if amplitudes.len() < 10 {
        return f64::NAN;
    }
    let min = amplitudes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = amplitudes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-30 {
        1.0
    } else {
        max - min
    };
    let mut hist = vec![0usize; bins];
    for &a in amplitudes {
        let mut k = ((a - min) / range * bins as f64) as usize;
        if k >= bins {
            k = bins - 1;
        }
        hist[k] += 1;
    }
    let peak_idx = hist.iter().enumerate().max_by_key(|(_, &c)| c).unwrap().0;
    if peak_idx == 0 {
        return 0.5;
    }
    let left_count: usize = hist[..peak_idx].iter().sum();
    let right_count: usize = hist[peak_idx..].iter().sum();
    let total = left_count + right_count;
    if total == 0 {
        return 0.0;
    }
    let estimated_missing = right_count.saturating_sub(left_count);
    estimated_missing as f64 / (total + estimated_missing) as f64
}

/// Signal-to-noise ratio of spike waveforms. Suner et al. 2005.
///
/// `waveforms`: row-major `(n_spikes, n_samples)`.
pub fn snr(waveforms: &[f64], n_spikes: usize, n_samples: usize) -> f64 {
    if n_spikes < 2 || n_samples == 0 {
        return f64::NAN;
    }
    // Mean waveform
    let mut mean_wf = vec![0.0f64; n_samples];
    for i in 0..n_spikes {
        for j in 0..n_samples {
            mean_wf[j] += waveforms[i * n_samples + j];
        }
    }
    for v in &mut mean_wf {
        *v /= n_spikes as f64;
    }
    let peak = mean_wf.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    // Noise std: mean of per-sample std
    let mut noise_std_sum = 0.0f64;
    for j in 0..n_samples {
        let mean_j = mean_wf[j];
        let var_j: f64 = (0..n_spikes)
            .map(|i| (waveforms[i * n_samples + j] - mean_j).powi(2))
            .sum::<f64>()
            / n_spikes as f64;
        noise_std_sum += var_j.sqrt();
    }
    let noise_std = noise_std_sum / n_samples as f64;
    if noise_std < 1e-30 {
        return f64::INFINITY;
    }
    peak / noise_std
}

/// Nearest-neighbour hit rate. Chung et al. 2017.
///
/// `cluster` and `noise` are row-major `(n, d)`.
pub fn nn_hit_rate(
    cluster: &[f64],
    n_c: usize,
    noise: &[f64],
    n_noise: usize,
    d: usize,
    k: usize,
) -> f64 {
    if n_c < k + 1 {
        return f64::NAN;
    }
    let n_total = n_c + n_noise;
    // Build combined points + labels
    let mut all_points = Vec::with_capacity(n_total * d);
    all_points.extend_from_slice(cluster);
    all_points.extend_from_slice(noise);

    let mut hits = 0usize;
    for i in 0..n_c {
        // Compute distances to all points
        let mut dists: Vec<(usize, f64)> = (0..n_total)
            .map(|j| {
                if j == i {
                    (j, f64::INFINITY)
                } else {
                    let dist = (0..d)
                        .map(|f| (all_points[i * d + f] - all_points[j * d + f]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (j, dist)
                }
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let all_in_cluster = dists[..k].iter().all(|(idx, _)| *idx < n_c);
        if all_in_cluster {
            hits += 1;
        }
    }
    hits as f64 / n_c as f64
}

/// Waveform drift metric. IBL 2019.
///
/// `waveforms`: row-major `(n_spikes, n_samples)`.
/// `timestamps`: `(n_spikes,)`.
pub fn drift_metric(
    waveforms: &[f64],
    n_spikes: usize,
    n_samples: usize,
    timestamps: &[f64],
    n_bins: usize,
) -> f64 {
    if n_spikes < n_bins || n_samples == 0 {
        return f64::NAN;
    }
    // Amplitudes
    let amplitudes: Vec<f64> = (0..n_spikes)
        .map(|i| {
            waveforms[i * n_samples..(i + 1) * n_samples]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f64, f64::max)
        })
        .collect();
    // Sort by timestamp
    let mut sorted_idx: Vec<usize> = (0..n_spikes).collect();
    sorted_idx.sort_by(|&a, &b| timestamps[a].partial_cmp(&timestamps[b]).unwrap());
    let sorted_amp: Vec<f64> = sorted_idx.iter().map(|&i| amplitudes[i]).collect();

    let bin_size = sorted_amp.len() / n_bins;
    if bin_size == 0 {
        return f64::NAN;
    }
    let mut means = vec![0.0f64; n_bins];
    for i in 0..n_bins {
        let chunk = &sorted_amp[i * bin_size..(i + 1) * bin_size];
        means[i] = chunk.iter().sum::<f64>() / chunk.len() as f64;
    }
    let mean_of_means: f64 = means.iter().sum::<f64>() / n_bins as f64;
    if mean_of_means.abs() < 1e-30 {
        return 0.0;
    }
    let max_m = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_m = means.iter().cloned().fold(f64::INFINITY, f64::min);
    (max_m - min_m) / mean_of_means
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster(centre: &[f64], n: usize, spread: f64, seed: u64) -> Vec<f64> {
        let d = centre.len();
        let mut data = vec![0.0f64; n * d];
        let mut rng = seed;
        for i in 0..n {
            for j in 0..d {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
                data[i * d + j] = centre[j] + u * spread;
            }
        }
        data
    }

    #[test]
    fn test_isolation_distance_well_separated() {
        let c = make_cluster(&[0.0, 0.0], 30, 1.0, 42);
        let n = make_cluster(&[10.0, 10.0], 50, 1.0, 99);
        let id = isolation_distance(&c, 30, &n, 50, 2);
        assert!(
            id > 10.0,
            "ID={id} should be large for well-separated clusters"
        );
    }

    #[test]
    fn test_isolation_distance_small_cluster() {
        assert!(isolation_distance(&[1.0], 1, &[2.0], 1, 1).is_nan());
    }

    #[test]
    fn test_l_ratio_well_separated() {
        let c = make_cluster(&[0.0, 0.0], 20, 0.5, 42);
        let n = make_cluster(&[10.0, 10.0], 30, 0.5, 99);
        let lr = l_ratio(&c, 20, &n, 30, 2);
        assert!(
            lr < 0.1,
            "L-ratio={lr} should be low for well-separated clusters"
        );
    }

    #[test]
    fn test_silhouette_two_clusters() {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for i in 0..10 {
            features.push(i as f64);
            features.push(0.0);
            labels.push(0i64);
        }
        for i in 0..10 {
            features.push(100.0 + i as f64);
            features.push(0.0);
            labels.push(1i64);
        }
        let s = silhouette_score(&features, 20, 2, &labels);
        assert!(
            s > 0.8,
            "Silhouette={s} should be high for well-separated clusters"
        );
    }

    #[test]
    fn test_silhouette_single_cluster() {
        let features = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![0i64, 0i64];
        assert_eq!(silhouette_score(&features, 2, 2, &labels), 0.0);
    }

    #[test]
    fn test_d_prime_separated() {
        let a = make_cluster(&[0.0, 0.0], 20, 1.0, 42);
        let b = make_cluster(&[10.0, 10.0], 20, 1.0, 99);
        let dp = d_prime(&a, 20, &b, 20, 2);
        assert!(dp > 5.0, "d'={dp} should be large for separated clusters");
    }

    #[test]
    fn test_d_prime_identical() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(d_prime(&a, 2, &b, 2, 2), 0.0);
    }

    #[test]
    fn test_isi_violation_rate() {
        // Regular ISI of 10ms, refractory 1.5ms -> no violations
        let mut train = vec![0i32; 200];
        for i in (0..200).step_by(10) {
            train[i] = 1;
        }
        let rate = isi_violation_rate(&train, 0.001, 1.5);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_isi_violation_rate_violations() {
        // ISI of 1ms, refractory 1.5ms -> all violations
        let train = vec![1i32; 10];
        let rate = isi_violation_rate(&train, 0.001, 1.5);
        assert_eq!(rate, 1.0);
    }

    #[test]
    fn test_presence_ratio() {
        let mut train = vec![0i32; 1000];
        for i in (0..1000).step_by(5) {
            train[i] = 1;
        }
        let pr = presence_ratio(&train, 100);
        assert!(pr > 0.9, "Presence ratio {pr} should be high");
    }

    #[test]
    fn test_amplitude_cutoff_basic() {
        // Gaussian-like amplitudes
        let amps: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64 - 50.0) * 0.5).collect();
        let ac = amplitude_cutoff(&amps, 50);
        assert!((0.0..=1.0).contains(&ac));
    }

    #[test]
    fn test_amplitude_cutoff_few() {
        assert!(amplitude_cutoff(&[1.0, 2.0], 10).is_nan());
    }

    #[test]
    fn test_snr_basic() {
        // 5 identical waveforms -> infinite SNR
        let wf = vec![
            0.0, -1.0, 0.5, 0.0, 0.0, -1.0, 0.5, 0.0, 0.0, -1.0, 0.5, 0.0, 0.0, -1.0, 0.5, 0.0,
            0.0, -1.0, 0.5, 0.0,
        ];
        let s = snr(&wf, 5, 4);
        assert!(s.is_infinite() || s > 100.0);
    }

    #[test]
    fn test_snr_few() {
        assert!(snr(&[1.0], 1, 1).is_nan());
    }

    #[test]
    fn test_nn_hit_rate_perfect() {
        let c = make_cluster(&[0.0, 0.0], 20, 0.1, 42);
        let n = make_cluster(&[100.0, 100.0], 20, 0.1, 99);
        let hr = nn_hit_rate(&c, 20, &n, 20, 2, 4);
        assert!((hr - 1.0).abs() < 1e-12, "Hit rate={hr} should be 1.0");
    }

    #[test]
    fn test_nn_hit_rate_too_few() {
        assert!(nn_hit_rate(&[1.0, 2.0], 1, &[3.0, 4.0], 1, 2, 4).is_nan());
    }

    #[test]
    fn test_drift_metric_stable() {
        // Constant amplitude -> drift = 0
        let n = 100;
        let wf: Vec<f64> = (0..n).flat_map(|_| vec![0.0, 1.0, 0.0]).collect();
        let ts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let dm = drift_metric(&wf, n, 3, &ts, 10);
        assert!(
            (dm).abs() < 1e-10,
            "Drift={dm} should be ~0 for stable waveforms"
        );
    }

    #[test]
    fn test_drift_metric_drifting() {
        // Increasing amplitude -> positive drift
        let n = 100;
        let wf: Vec<f64> = (0..n).flat_map(|i| vec![0.0, i as f64, 0.0]).collect();
        let ts: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let dm = drift_metric(&wf, n, 3, &ts, 10);
        assert!(
            dm > 0.5,
            "Drift={dm} should be positive for drifting waveforms"
        );
    }

    #[test]
    fn test_cluster_mahalanobis_sq_matches_dense() {
        // 4-point 2-D cluster; the Cholesky-solve helper must equal the
        // closed-form `diffᵀ Σ⁻¹ diff` using the 2×2 analytic inverse.
        let cluster = vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0];
        let point = vec![5.0, 3.0];
        let mah = cluster_mahalanobis_sq(&cluster, 4, &point, 1, 2);

        let cov = covariance_matrix(&cluster, 4, 2);
        let (a, b, c, dd) = (cov[0], cov[1], cov[2], cov[3]);
        let det = a * dd - b * c;
        let mu = col_mean(&cluster, 4, 2);
        let (dx, dy) = (point[0] - mu[0], point[1] - mu[1]);
        // Σ⁻¹ = (1/det) [[dd, -b], [-c, a]]
        let ref_mah = (dx * (dd * dx - b * dy) + dy * (-c * dx + a * dy)) / det;
        assert!(
            (mah[0] - ref_mah).abs() < 1e-9,
            "mah={} ref={}",
            mah[0],
            ref_mah
        );
    }

    #[test]
    fn test_cluster_mahalanobis_sq_centre_is_zero() {
        // The cluster mean has zero Mahalanobis distance from itself.
        let cluster = vec![1.0, 4.0, 3.0, 4.0, 1.0, 8.0, 3.0, 8.0];
        let centre = vec![2.0, 6.0]; // column means
        let mah = cluster_mahalanobis_sq(&cluster, 4, &centre, 1, 2);
        assert!(
            mah[0].abs() < 1e-9,
            "centre distance {} should be ~0",
            mah[0]
        );
    }
}
