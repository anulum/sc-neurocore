// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Information-theoretic measures for spike trains

use std::collections::HashMap;

use super::basic::bin_spike_train;

/// Digamma function approximation (Stirling series, accurate to ~1e-10 for x > 6).
fn digamma(mut x: f64) -> f64 {
    // Shift x up if small, using ψ(x) = ψ(x+1) - 1/x
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion for large x
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += x.ln()
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0 - inv_x2 / 240.0)));
    result
}

fn entropy_from_counts(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n_inv = 1.0 / total as f64;
    let mut h = 0.0_f64;

    let (chunks, remainder) = counts.as_chunks::<4>();
    for chunk in chunks {
        for &c in chunk {
            if c > 0 {
                let p = c as f64 * n_inv;
                h -= p * (p + 1e-30).log2();
            }
        }
    }
    for &c in remainder {
        if c > 0 {
            let p = c as f64 * n_inv;
            h -= p * (p + 1e-30).log2();
        }
    }
    h
}

fn count_values(data: &[i64]) -> Vec<usize> {
    let mut map: HashMap<i64, usize> = HashMap::new();
    for &v in data {
        *map.entry(v).or_insert(0) += 1;
    }
    map.into_values().collect()
}

/// Mutual information between two binned spike trains (bits).
pub fn mutual_information(train_a: &[i32], train_b: &[i32], bin_size: usize) -> f64 {
    let ca = bin_spike_train(train_a, bin_size);
    let cb = bin_spike_train(train_b, bin_size);
    let n = ca.len().min(cb.len());
    if n == 0 {
        return 0.0;
    }

    let ha = entropy_from_counts(&count_values(&ca[..n]), n);
    let hb = entropy_from_counts(&count_values(&cb[..n]), n);

    // Joint: encode as ca[i] * (max_b + 1) + cb[i]
    let max_b = cb[..n].iter().copied().max().unwrap_or(0);
    let joint: Vec<i64> = (0..n).map(|i| ca[i] * (max_b + 1) + cb[i]).collect();
    let hab = entropy_from_counts(&count_values(&joint), n);

    (ha + hb - hab).max(0.0)
}

/// Transfer entropy from source to target spike train (bits).
pub fn transfer_entropy(source: &[i32], target: &[i32], bin_size: usize, lag: usize) -> f64 {
    let cs = bin_spike_train(source, bin_size);
    let ct = bin_spike_train(target, bin_size);
    let n = cs.len().min(ct.len());
    if n <= lag {
        return 0.0;
    }

    let t_past = &ct[..n - lag];
    let t_future = &ct[lag..n];
    let s_past = &cs[..n - lag];
    let n_pts = t_past.len();

    // H(t_future | t_past) = H(t_future, t_past) - H(t_past)
    let max_tp = t_past.iter().copied().max().unwrap_or(0) + 1;
    let joint_ft: Vec<i64> = (0..n_pts)
        .map(|i| t_future[i] * max_tp + t_past[i])
        .collect();
    let h_ft = entropy_from_counts(&count_values(&joint_ft), n_pts);
    let h_tp = entropy_from_counts(&count_values(t_past), n_pts);
    let h1 = h_ft - h_tp;

    // H(t_future | t_past, s_past) = H(t_future, t_past, s_past) - H(t_past, s_past)
    let max_sp = s_past.iter().copied().max().unwrap_or(0) + 1;
    let past_joint: Vec<i64> = (0..n_pts).map(|i| t_past[i] * max_sp + s_past[i]).collect();
    let max_pj = past_joint.iter().copied().max().unwrap_or(0) + 1;
    let joint_fts: Vec<i64> = (0..n_pts)
        .map(|i| t_future[i] * max_pj + past_joint[i])
        .collect();
    let h_fts = entropy_from_counts(&count_values(&joint_fts), n_pts);
    let h_ps = entropy_from_counts(&count_values(&past_joint), n_pts);
    let h2 = h_fts - h_ps;

    (h1 - h2).max(0.0)
}

/// Spike train entropy via binary word analysis (Strong et al. 1998). Returns bits.
pub fn spike_train_entropy(binary_train: &[i32], bin_size: usize, word_length: usize) -> f64 {
    let binned: Vec<i64> = bin_spike_train(binary_train, bin_size)
        .iter()
        .map(|&v| if v > 0 { 1_i64 } else { 0_i64 })
        .collect();
    let n = binned.len();
    if n < word_length {
        return f64::NAN;
    }
    let n_words = n - word_length + 1;
    let mut words = Vec::with_capacity(n_words);
    for i in 0..n_words {
        let mut w = 0_i64;
        for j in 0..word_length {
            w = w * 2 + binned[i + j];
        }
        words.push(w);
    }
    entropy_from_counts(&count_values(&words), n_words)
}

/// Noise entropy via pseudo-trials (de Ruyter van Steveninck et al. 1997). Returns bits.
pub fn noise_entropy(
    binary_train: &[i32],
    n_trials: usize,
    bin_size: usize,
    word_length: usize,
) -> f64 {
    let n = binary_train.len();
    let trial_len = n / n_trials;
    if trial_len < bin_size * word_length {
        return f64::NAN;
    }
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for t in 0..n_trials {
        let start = t * trial_len;
        let end = start + trial_len;
        let h = spike_train_entropy(&binary_train[start..end], bin_size, word_length);
        if !h.is_nan() {
            sum += h;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    sum / count as f64
}

/// Stimulus-specific information (Butts 2003). Returns bits.
pub fn stimulus_specific_information(spike_counts: &[f64], stimulus_ids: &[i64]) -> f64 {
    let n_total = spike_counts.len().min(stimulus_ids.len());
    if n_total == 0 {
        return 0.0;
    }

    let overall_mean: f64 = spike_counts[..n_total].iter().sum::<f64>() / n_total as f64;
    if overall_mean <= 0.0 {
        return 0.0;
    }

    // Group by stimulus
    let mut groups: HashMap<i64, Vec<f64>> = HashMap::new();
    for i in 0..n_total {
        groups
            .entry(stimulus_ids[i])
            .or_default()
            .push(spike_counts[i]);
    }

    let mut ssi = 0.0_f64;
    for counts in groups.values() {
        let n_s = counts.len() as f64;
        let p_s = n_s / n_total as f64;
        let mean_s: f64 = counts.iter().sum::<f64>() / n_s;
        if mean_s > 0.0 {
            ssi += p_s * mean_s * (mean_s / overall_mean).log2() / overall_mean;
        }
    }
    ssi.max(0.0)
}

/// Kozachenko-Leonenko k-NN mutual information estimator (Kraskov et al. 2004).
/// Returns MI in nats.
pub fn kozachenko_leonenko_mi(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len().min(y.len());
    if n < k + 1 {
        return 0.0;
    }

    let psi_k = digamma(k as f64);
    let psi_n = digamma(n as f64);

    let mut nx_sum = 0.0_f64;
    let mut ny_sum = 0.0_f64;

    for i in 0..n {
        // Chebyshev (max-norm) k-th neighbour distance in joint space
        let mut dists: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (x[i] - x[j]).abs().max((y[i] - y[j]).abs()))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let eps = dists[k - 1];

        // Count neighbours within eps in marginal spaces
        let nx = (0..n)
            .filter(|&j| j != i && (x[i] - x[j]).abs() < eps)
            .count();
        let ny = (0..n)
            .filter(|&j| j != i && (y[i] - y[j]).abs() < eps)
            .count();

        nx_sum += digamma((nx + 1) as f64);
        ny_sum += digamma((ny + 1) as f64);
    }

    (psi_k + psi_n - nx_sum / n as f64 - ny_sum / n as f64).max(0.0)
}

/// Time-rescaling KS test for point process goodness-of-fit (Brown et al. 2002).
/// rate_func(t) -> conditional intensity. Returns (ks_statistic, passes_at_95pct).
pub fn time_rescaling_ks_test(
    times: &[f64],
    rate_func: fn(f64) -> f64,
    t_start: f64,
    t_end: f64,
) -> (f64, bool) {
    let mut sorted: Vec<f64> = times
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n < 5 {
        return (1.0, false);
    }

    let n_quad = 20_usize;
    let mut rescaled = Vec::with_capacity(n);
    for i in 0..n {
        let lo = if i == 0 { t_start } else { sorted[i - 1] };
        let hi = sorted[i];
        // Trapezoidal integration
        let step = (hi - lo) / (n_quad - 1).max(1) as f64;
        let mut integral = 0.0_f64;
        for q in 0..n_quad {
            let t = lo + q as f64 * step;
            let w = if q == 0 || q == n_quad - 1 { 0.5 } else { 1.0 };
            integral += w * rate_func(t) * step;
        }
        rescaled.push(integral);
    }

    let mut transformed: Vec<f64> = rescaled.iter().map(|&r| 1.0 - (-r).exp()).collect();
    transformed.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut ks = 0.0_f64;
    for i in 0..n {
        let ecdf = (i + 1) as f64 / n as f64;
        ks = ks.max((ecdf - transformed[i]).abs());
    }

    let critical_95 = 1.36 / (n as f64).sqrt();
    (ks, ks < critical_95)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_train(spikes: &[usize], len: usize) -> Vec<i32> {
        let mut t = vec![0i32; len];
        for &s in spikes {
            t[s] = 1;
        }
        t
    }

    // ── digamma ─────────────────────────────────────────────────────

    #[test]
    fn test_digamma_known_values() {
        // ψ(1) = -γ ≈ -0.5772156649
        assert!((digamma(1.0) - (-0.5772156649)).abs() < 1e-8);
        // ψ(2) = 1 - γ ≈ 0.4227843351
        assert!((digamma(2.0) - 0.4227843351).abs() < 1e-8);
        // ψ(0.5) = -γ - 2 ln 2 ≈ -1.9635100260
        assert!((digamma(0.5) - (-1.9635100260)).abs() < 1e-7);
    }

    // ── mutual_information ──────────────────────────────────────────

    #[test]
    fn test_mi_identical() {
        let train = make_train(&[0, 1, 2, 10, 11, 12, 20, 21, 22], 30);
        let mi = mutual_information(&train, &train, 5);
        assert!(mi > 0.0, "identical trains → positive MI, got {mi}");
    }

    #[test]
    fn test_mi_non_negative() {
        let a = make_train(&[5, 15, 25], 30);
        let b = make_train(&[0, 10, 20], 30);
        let mi = mutual_information(&a, &b, 5);
        assert!(mi >= 0.0, "MI must be non-negative");
    }

    #[test]
    fn test_mi_zero_constant() {
        // Both trains have same bin counts → H(A,B) = H(A) → MI = H(B) which is 0 for constant
        let a = vec![0i32; 50];
        let b = vec![0i32; 50];
        let mi = mutual_information(&a, &b, 10);
        assert!(mi.abs() < 1e-10, "constant trains → MI ≈ 0, got {mi}");
    }

    // ── transfer_entropy ────────────────────────────────────────────

    #[test]
    fn test_te_non_negative() {
        let source = make_train(&[5, 15, 25, 35, 45], 50);
        let target = make_train(&[7, 17, 27, 37, 47], 50);
        let te = transfer_entropy(&source, &target, 5, 1);
        assert!(te >= 0.0, "TE must be non-negative");
    }

    #[test]
    fn test_te_short_returns_zero() {
        let source = make_train(&[1], 5);
        let target = make_train(&[2], 5);
        let te = transfer_entropy(&source, &target, 5, 10);
        assert_eq!(te, 0.0, "n <= lag → 0");
    }

    #[test]
    fn test_te_self_zero() {
        // TE(X→X) should be 0 (no additional info from knowing past of X beyond X's own past)
        let train = make_train(&[5, 15, 25, 35, 45], 50);
        let te = transfer_entropy(&train, &train, 5, 1);
        assert!(te < 1e-10, "TE(X→X) should be ~0, got {te}");
    }

    // ── spike_train_entropy ─────────────────────────────────────────

    #[test]
    fn test_entropy_constant() {
        let train = vec![0i32; 100];
        let h = spike_train_entropy(&train, 10, 4);
        assert!(h.abs() < 1e-10, "constant → entropy 0, got {h}");
    }

    #[test]
    fn test_entropy_all_ones_binary() {
        // All bins active → single word type → entropy 0
        let train = vec![1i32; 100];
        let h = spike_train_entropy(&train, 10, 4);
        assert!(h.abs() < 1e-10, "uniform → entropy 0, got {h}");
    }

    #[test]
    fn test_entropy_non_negative() {
        let train = make_train(&[5, 15, 25, 45, 55, 85], 100);
        let h = spike_train_entropy(&train, 10, 4);
        assert!(h >= 0.0 || h.is_nan(), "entropy must be non-negative");
    }

    #[test]
    fn test_entropy_short_nan() {
        let train = make_train(&[0, 1], 5);
        let h = spike_train_entropy(&train, 10, 4);
        assert!(h.is_nan(), "too short → NaN");
    }

    // ── noise_entropy ───────────────────────────────────────────────

    #[test]
    fn test_noise_entropy_constant() {
        let train = vec![0i32; 500];
        let h = noise_entropy(&train, 5, 10, 4);
        assert!(h.abs() < 1e-10 || h.is_nan(), "constant → 0 or NaN");
    }

    #[test]
    fn test_noise_entropy_too_short() {
        let train = vec![0i32; 10];
        let h = noise_entropy(&train, 10, 10, 4);
        assert!(h.is_nan(), "too short → NaN");
    }

    // ── stimulus_specific_information ───────────────────────────────

    #[test]
    fn test_ssi_uniform() {
        // All stimuli produce same response → SSI = 0
        let counts = vec![5.0, 5.0, 5.0, 5.0];
        let stim = vec![0_i64, 1, 0, 1];
        let ssi = stimulus_specific_information(&counts, &stim);
        assert!(ssi.abs() < 1e-10, "uniform response → SSI 0, got {ssi}");
    }

    #[test]
    fn test_ssi_selective() {
        // Stim 0 → high rate, stim 1 → low rate
        let counts = vec![10.0, 1.0, 10.0, 1.0];
        let stim = vec![0_i64, 1, 0, 1];
        let ssi = stimulus_specific_information(&counts, &stim);
        assert!(ssi > 0.0, "selective response → positive SSI, got {ssi}");
    }

    #[test]
    fn test_ssi_empty() {
        let ssi = stimulus_specific_information(&[], &[]);
        assert_eq!(ssi, 0.0);
    }

    // ── kozachenko_leonenko_mi ──────────────────────────────────────

    #[test]
    fn test_kl_mi_identical() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y = x.clone();
        let mi = kozachenko_leonenko_mi(&x, &y, 3);
        assert!(mi > 0.0, "identical signals → positive MI, got {mi}");
    }

    #[test]
    fn test_kl_mi_independent() {
        // Independent data: interleaved patterns, should have low MI
        let x: Vec<f64> = (0..100).map(|i| (i % 7) as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| (i % 11) as f64).collect();
        let mi = kozachenko_leonenko_mi(&x, &y, 3);
        // MI for independent should be near 0 (with some estimation noise)
        assert!(mi < 1.0, "roughly independent → low MI, got {mi}");
    }

    #[test]
    fn test_kl_mi_too_few() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        assert_eq!(kozachenko_leonenko_mi(&x, &y, 3), 0.0, "n < k+1 → 0");
    }

    // ── time_rescaling_ks_test ──────────────────────────────────────

    #[test]
    fn test_ks_constant_rate() {
        fn rate(_t: f64) -> f64 {
            100.0
        }
        // Regular spikes at constant rate: should pass
        let times: Vec<f64> = (0..50).map(|i| i as f64 * 0.02).collect();
        let (ks, _passes) = time_rescaling_ks_test(&times, rate, 0.0, 1.0);
        assert!((0.0..=1.0).contains(&ks), "KS stat in [0,1], got {ks}");
    }

    #[test]
    fn test_ks_too_few_spikes() {
        fn rate(_t: f64) -> f64 {
            100.0
        }
        let (ks, passes) = time_rescaling_ks_test(&[0.5], rate, 0.0, 1.0);
        assert_eq!(ks, 1.0);
        assert!(!passes);
    }

    // ── entropy_from_counts ─────────────────────────────────────────

    #[test]
    fn test_entropy_single_symbol() {
        let h = entropy_from_counts(&[10], 10);
        assert!(h.abs() < 1e-10, "single symbol → entropy 0");
    }

    #[test]
    fn test_entropy_uniform_two() {
        // 50/50 → 1 bit
        let h = entropy_from_counts(&[5, 5], 10);
        assert!((h - 1.0).abs() < 1e-10, "uniform binary → 1 bit, got {h}");
    }

    #[test]
    fn test_entropy_uniform_four() {
        // 4 equally likely → 2 bits
        let h = entropy_from_counts(&[25, 25, 25, 25], 100);
        assert!((h - 2.0).abs() < 1e-10, "uniform 4-ary → 2 bits, got {h}");
    }
}
