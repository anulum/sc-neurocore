// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SPADE: Spike Pattern Detection and Evaluation
//
// Torre, Canova, Denker, Gerstein, Helias, Gruen (2013)
// Front. Comput. Neurosci. 7:132.

use std::collections::HashSet;

/// A detected spatiotemporal pattern.
#[derive(Debug, Clone)]
pub struct SpadePattern {
    /// Neuron indices participating in the pattern.
    pub neurons: Vec<usize>,
    /// Lag (in bins) for each neuron relative to the reference.
    pub lags: Vec<i64>,
    /// Number of occurrences.
    pub count: usize,
    /// p-value from surrogate testing.
    pub p_value: f64,
}

// ── helpers ─────────────────────────────────────────────────────────

/// Build binary matrix (n_neurons x n_bins) from spike trains.
fn build_binary_matrix(trains: &[&[i32]], bin_steps: usize, n_bins: usize) -> Vec<Vec<u8>> {
    trains
        .iter()
        .map(|t| {
            let mut row = vec![0u8; n_bins];
            for b in 0..n_bins {
                let start = b * bin_steps;
                let end = ((b + 1) * bin_steps).min(t.len());
                if start < t.len() && t[start..end].iter().any(|&v| v > 0) {
                    row[b] = 1;
                }
            }
            row
        })
        .collect()
}

/// Apriori-style frequent itemset mining.
fn find_frequent_itemsets(
    binary_matrix: &[Vec<u8>],
    min_support: usize,
    max_size: usize,
) -> Vec<(Vec<usize>, usize)> {
    let n_neurons = binary_matrix.len();
    let n_bins = if n_neurons > 0 {
        binary_matrix[0].len()
    } else {
        return vec![];
    };

    let mut freq: Vec<(Vec<usize>, usize)> = Vec::new();
    let mut candidates_k: Vec<Vec<usize>> = Vec::new();

    // Level 1
    for nid in 0..n_neurons {
        let cnt: usize = binary_matrix[nid].iter().map(|&v| v as usize).sum();
        if cnt >= min_support {
            let s = vec![nid];
            freq.push((s.clone(), cnt));
            candidates_k.push(s);
        }
    }

    // Level k >= 2
    for k in 2..=max_size {
        if candidates_k.len() < 2 {
            break;
        }
        let mut new_candidates: HashSet<Vec<usize>> = HashSet::new();
        let prev = &candidates_k;
        for i in 0..prev.len() {
            for j in i + 1..prev.len() {
                let mut union: Vec<usize> = prev[i].clone();
                for &v in &prev[j] {
                    if !union.contains(&v) {
                        union.push(v);
                    }
                }
                union.sort();
                if union.len() == k {
                    new_candidates.insert(union);
                }
            }
        }

        candidates_k = Vec::new();
        for s in new_candidates {
            // Count co-active bins
            let mut cnt = 0usize;
            for b in 0..n_bins {
                if s.iter().all(|&nid| binary_matrix[nid][b] > 0) {
                    cnt += 1;
                }
            }
            if cnt >= min_support {
                freq.push((s.clone(), cnt));
                candidates_k.push(s);
            }
        }
    }

    freq
}

/// Extend synchronous itemsets to spatiotemporal patterns with lags.
fn extend_to_spatiotemporal(
    trains: &[&[i32]],
    itemsets: &[(Vec<usize>, usize)],
    bin_steps: usize,
    n_bins: usize,
    max_lag_bins: usize,
) -> Vec<(Vec<usize>, Vec<i64>, usize)> {
    let mut patterns = Vec::new();

    for (neurons, _sync_count) in itemsets {
        if neurons.len() < 2 {
            continue;
        }
        let ref_id = neurons[0];

        // Reference neuron binned
        let mut ref_bins = vec![0u8; n_bins];
        for b in 0..n_bins {
            let start = b * bin_steps;
            let end = ((b + 1) * bin_steps).min(trains[ref_id].len());
            if start < trains[ref_id].len() && trains[ref_id][start..end].iter().any(|&v| v > 0) {
                ref_bins[b] = 1;
            }
        }

        let mut best_lags: Vec<(usize, i64)> = vec![(ref_id, 0)];
        let mut coincidence = ref_bins.clone();

        for &nid in &neurons[1..] {
            let mut best_lag: i64 = 0;
            let mut best_overlap = 0usize;

            for lag in 0..=max_lag_bins {
                let mut shifted = vec![0u8; n_bins];
                for b in 0..n_bins {
                    let src_b = b as i64 - lag as i64;
                    if src_b >= 0 && (src_b as usize) < n_bins {
                        let start = src_b as usize * bin_steps;
                        let end = ((src_b as usize + 1) * bin_steps).min(trains[nid].len());
                        if start < trains[nid].len()
                            && trains[nid][start..end].iter().any(|&v| v > 0)
                        {
                            shifted[b] = 1;
                        }
                    }
                }
                let overlap: usize = coincidence
                    .iter()
                    .zip(shifted.iter())
                    .map(|(&a, &b)| (a & b) as usize)
                    .sum();
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_lag = lag as i64;
                }
            }

            best_lags.push((nid, best_lag));

            // Update coincidence
            let mut nbins_best = vec![0u8; n_bins];
            for b in 0..n_bins {
                let src_b = b as i64 - best_lag;
                if src_b >= 0 && (src_b as usize) < n_bins {
                    let start = src_b as usize * bin_steps;
                    let end = ((src_b as usize + 1) * bin_steps).min(trains[nid].len());
                    if start < trains[nid].len() && trains[nid][start..end].iter().any(|&v| v > 0) {
                        nbins_best[b] = 1;
                    }
                }
            }
            for b in 0..n_bins {
                coincidence[b] &= nbins_best[b];
            }
        }

        let best_count: usize = coincidence.iter().map(|&v| v as usize).sum();
        if best_count > 0 {
            let neuron_list: Vec<usize> = best_lags.iter().map(|&(n, _)| n).collect();
            let lag_list: Vec<i64> = best_lags.iter().map(|&(_, l)| l).collect();
            patterns.push((neuron_list, lag_list, best_count));
        }
    }

    patterns
}

// ── public API ──────────────────────────────────────────────────────

/// Detect repeated spatiotemporal spike patterns with significance testing.
pub fn spade_detect(
    trains: &[&[i32]],
    bin_ms: f64,
    dt: f64,
    min_support: usize,
    max_pattern_size: usize,
    n_surrogates: usize,
    alpha: f64,
    seed: u64,
) -> Vec<SpadePattern> {
    let n_neurons = trains.len();
    if n_neurons < 2 {
        return vec![];
    }
    let bin_steps = (bin_ms / (dt * 1000.0)).round().max(1.0) as usize;
    let duration = trains.iter().map(|t| t.len()).max().unwrap_or(0);
    let n_bins = duration / bin_steps;
    if n_bins == 0 {
        return vec![];
    }

    let binary_matrix = build_binary_matrix(trains, bin_steps, n_bins);
    let itemsets = find_frequent_itemsets(&binary_matrix, min_support, max_pattern_size);
    if itemsets.is_empty() {
        return vec![];
    }

    let patterns = extend_to_spatiotemporal(trains, &itemsets, bin_steps, n_bins, 10);
    if patterns.is_empty() {
        return vec![];
    }

    // Surrogate testing
    let mut rng = seed;
    let mut results = Vec::new();

    for (neuron_list, lag_list, count) in &patterns {
        let mut surr_counts = vec![0usize; n_surrogates];

        for s in 0..n_surrogates {
            // Circular shift each train
            let surr_trains: Vec<Vec<i32>> = (0..n_neurons)
                .map(|i| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let shift = (rng % (bin_steps as u64 * 10 + 1)) as i64 - (bin_steps as i64 * 5);
                    let n = trains[i].len();
                    if n == 0 {
                        return vec![];
                    }
                    let mut shifted = vec![0i32; n];
                    for j in 0..n {
                        let src = ((j as i64 - shift).rem_euclid(n as i64)) as usize;
                        shifted[j] = trains[i][src];
                    }
                    shifted
                })
                .collect();

            let surr_binary = build_binary_matrix(
                &surr_trains.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                bin_steps,
                n_bins,
            );

            // Count coincidences for this pattern
            let mut coincidence = vec![1u8; n_bins];
            for (idx, (&nid, &lag)) in neuron_list.iter().zip(lag_list.iter()).enumerate() {
                let _ = idx;
                for b in 0..n_bins {
                    let src_b = b as i64 - lag;
                    if src_b >= 0 && (src_b as usize) < n_bins {
                        coincidence[b] &= surr_binary[nid][src_b as usize];
                    } else {
                        coincidence[b] = 0;
                    }
                }
            }
            surr_counts[s] = coincidence.iter().map(|&v| v as usize).sum();
        }

        let p_value = (surr_counts.iter().filter(|&&c| c >= *count).count() + 1) as f64
            / (n_surrogates + 1) as f64;
        if p_value <= alpha {
            results.push(SpadePattern {
                neurons: neuron_list.clone(),
                lags: lag_list.clone(),
                count: *count,
                p_value,
            });
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_correlated_trains() -> Vec<Vec<i32>> {
        // 3 neurons with synchronous activity
        let n = 500;
        let mut trains = vec![vec![0i32; n]; 3];
        // Synchronous spikes every 20 steps
        for i in (0..n).step_by(20) {
            trains[0][i] = 1;
            trains[1][i] = 1;
            if i + 2 < n {
                trains[2][i + 2] = 1; // lagged
            }
        }
        // Add some noise spikes
        let mut rng = 42u64;
        for t in &mut trains {
            for j in 0..n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if rng % 50 == 0 && t[j] == 0 {
                    t[j] = 1;
                }
            }
        }
        trains
    }

    #[test]
    fn test_spade_detects_patterns() {
        let trains = make_correlated_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let results = spade_detect(&refs, 5.0, 0.001, 3, 3, 50, 0.05, 42);
        // Should detect at least one pattern
        assert!(
            !results.is_empty(),
            "SPADE should detect synchronous patterns"
        );
    }

    #[test]
    fn test_spade_pattern_fields() {
        let trains = make_correlated_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let results = spade_detect(&refs, 5.0, 0.001, 3, 3, 50, 0.05, 42);
        for pat in &results {
            assert!(pat.neurons.len() >= 2);
            assert_eq!(pat.neurons.len(), pat.lags.len());
            assert!(pat.count > 0);
            assert!(pat.p_value <= 0.05);
            assert!(pat.p_value > 0.0);
        }
    }

    #[test]
    fn test_spade_empty() {
        let results = spade_detect(&[], 5.0, 0.001, 3, 3, 50, 0.05, 42);
        assert!(results.is_empty());
    }

    #[test]
    fn test_spade_single_neuron() {
        let train = vec![1, 0, 1, 0, 1];
        let results = spade_detect(&[&train], 5.0, 0.001, 1, 3, 50, 0.05, 42);
        assert!(results.is_empty());
    }

    #[test]
    fn test_spade_no_pattern() {
        // Independent random trains -> likely no significant patterns
        let n = 200;
        let mut trains = Vec::new();
        let mut rng = 42u64;
        for _ in 0..3 {
            let mut t = vec![0i32; n];
            for j in 0..n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if rng % 20 == 0 {
                    t[j] = 1;
                }
            }
            trains.push(t);
        }
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let results = spade_detect(&refs, 5.0, 0.001, 5, 3, 100, 0.01, 42);
        // May or may not find patterns, but shouldn't crash
        let _ = results;
    }

    #[test]
    fn test_find_frequent_itemsets() {
        let matrix = vec![
            vec![1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            vec![1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            vec![0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        ];
        let itemsets = find_frequent_itemsets(&matrix, 3, 3);
        // Neurons 0 and 1 are co-active in 7 bins
        let has_pair = itemsets
            .iter()
            .any(|(s, _)| s.len() == 2 && s.contains(&0) && s.contains(&1));
        assert!(has_pair, "Should find {{0,1}} as frequent pair");
    }

    #[test]
    fn test_build_binary_matrix() {
        let train = vec![0, 0, 1, 0, 0, 1, 0, 0, 0, 0];
        let mat = build_binary_matrix(&[&train], 5, 2);
        assert_eq!(mat.len(), 1);
        assert_eq!(mat[0], vec![1, 1]); // both bins have spikes
    }

    #[test]
    fn test_spade_deterministic() {
        let trains = make_correlated_trains();
        let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
        let r1 = spade_detect(&refs, 5.0, 0.001, 3, 3, 30, 0.05, 42);
        let r2 = spade_detect(&refs, 5.0, 0.001, 3, 3, 30, 0.05, 42);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.neurons, b.neurons);
            assert_eq!(a.count, b.count);
            assert!((a.p_value - b.p_value).abs() < 1e-12);
        }
    }
}
