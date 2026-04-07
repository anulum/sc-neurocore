// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike directionality, ordering, and higher-order patterns

/// Spike directionality (Kreuz et al. 2015).
/// Returns asymmetry in [-1, 1]. Positive: A leads B.
use rayon::prelude::*;

pub fn spike_directionality(times_a: &[f64], times_b: &[f64], t_start: f64, t_end: f64) -> f64 {
    let mut ta: Vec<f64> = times_a
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    let tb: Vec<f64> = times_b
        .iter()
        .copied()
        .filter(|&t| t >= t_start && t <= t_end)
        .collect();
    ta.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }

    let mut lead_a = 0_usize;
    let mut lead_b = 0_usize;

    // Ensure tb is sorted for binary search
    let mut tb_sorted = tb.clone();
    tb_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for &t in &ta {
        let idx = tb_sorted.partition_point(|&x| x < t);

        let nearest_after = if idx < tb_sorted.len() {
            Some(tb_sorted[idx] - t)
        } else {
            None
        };

        let nearest_before = if idx > 0 {
            Some(t - tb_sorted[idx - 1])
        } else {
            None
        };

        if let (Some(nb), Some(na)) = (nearest_before, nearest_after) {
            if nb < na {
                lead_b += 1;
            } else {
                lead_a += 1;
            }
        }
    }

    let total = lead_a + lead_b;
    if total == 0 {
        return 0.0;
    }
    (lead_a as f64 - lead_b as f64) / total as f64
}

/// Spike train order matrix (Kreuz et al. 2017).
/// Returns n×n matrix (flat, row-major) of pairwise directionality.
pub fn spike_train_order(times_list: &[&[f64]], t_start: f64, t_end: f64) -> Vec<f64> {
    let n = times_list.len();
    let mut mat = vec![0.0_f64; n * n];
    mat.par_chunks_exact_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Note: full matrix computation simplifies parallel dispatch
                // even though directionality is antisymmetric.
                if j > i {
                    let d = spike_directionality(times_list[i], times_list[j], t_start, t_end);
                    row[j] = d;
                } else {
                    let d = spike_directionality(times_list[j], times_list[i], t_start, t_end);
                    row[j] = -d;
                }
            }
        });
    mat
}

/// Third-order cumulant C3(tau1, tau2) (Nikias & Petropulu 1993).
/// Returns max_lag × max_lag matrix (flat, row-major).
pub fn cubic_higher_order(binary_train: &[i32], dt: f64, max_lag: usize) -> Vec<f64> {
    let _ = dt;
    let n = binary_train.len();
    let mean: f64 = binary_train.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let x: Vec<f64> = binary_train.iter().map(|&v| v as f64 - mean).collect();

    let mut c3 = vec![0.0_f64; max_lag * max_lag];
    c3.par_chunks_exact_mut(max_lag)
        .enumerate()
        .for_each(|(t1, row)| {
            for t2 in 0..max_lag {
                let valid_n = n.saturating_sub(t1.max(t2));
                if valid_n == 0 {
                    continue;
                }
                let mut sum = 0.0_f64;
                let mut k = 0;
                while k + 3 < valid_n {
                    sum += x[k] * x[k + t1] * x[k + t2];
                    sum += x[k + 1] * x[k + 1 + t1] * x[k + 1 + t2];
                    sum += x[k + 2] * x[k + 2 + t1] * x[k + 2 + t2];
                    sum += x[k + 3] * x[k + 3 + t1] * x[k + 3 + t2];
                    k += 4;
                }
                while k < valid_n {
                    sum += x[k] * x[k + t1] * x[k + t2];
                    k += 1;
                }
                row[t2] = sum / valid_n as f64;
            }
        });
    c3
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── spike_directionality ────────────────────────────────────────

    #[test]
    fn test_directionality_a_leads() {
        // A fires before B consistently
        let ta = vec![0.1, 0.3, 0.5, 0.7];
        let tb = vec![0.15, 0.35, 0.55, 0.75];
        let d = spike_directionality(&ta, &tb, 0.0, 1.0);
        assert!(d > 0.0, "A leads B → positive, got {d}");
    }

    #[test]
    fn test_directionality_b_leads() {
        let ta = vec![0.15, 0.35, 0.55, 0.75];
        let tb = vec![0.1, 0.3, 0.5, 0.7];
        let d = spike_directionality(&ta, &tb, 0.0, 1.0);
        assert!(d < 0.0, "B leads A → negative, got {d}");
    }

    #[test]
    fn test_directionality_empty() {
        assert_eq!(spike_directionality(&[], &[0.5], 0.0, 1.0), 0.0);
    }

    #[test]
    fn test_directionality_antisymmetric() {
        let ta = vec![0.1, 0.3, 0.5];
        let tb = vec![0.15, 0.35, 0.55];
        let d_ab = spike_directionality(&ta, &tb, 0.0, 1.0);
        let d_ba = spike_directionality(&tb, &ta, 0.0, 1.0);
        // Not exactly antisymmetric due to algorithm, but should have opposite signs
        assert!(d_ab > 0.0 && d_ba < 0.0, "should have opposite signs");
    }

    // ── spike_train_order ───────────────────────────────────────────

    #[test]
    fn test_order_antisymmetric() {
        let ta = vec![0.1, 0.3, 0.5];
        let tb = vec![0.15, 0.35, 0.55];
        let trains: Vec<&[f64]> = vec![&ta, &tb];
        let mat = spike_train_order(&trains, 0.0, 1.0);
        assert!((mat[1] + mat[2]).abs() < 1e-10, "antisymmetric");
        assert_eq!(mat[0], 0.0, "diagonal = 0");
    }

    #[test]
    fn test_order_shape() {
        let ta = vec![0.1];
        let tb = vec![0.2];
        let tc = vec![0.3];
        let trains: Vec<&[f64]> = vec![&ta, &tb, &tc];
        let mat = spike_train_order(&trains, 0.0, 1.0);
        assert_eq!(mat.len(), 9);
    }

    // ── cubic_higher_order ──────────────────────────────────────────

    #[test]
    fn test_c3_shape() {
        let train = vec![0i32, 1, 0, 1, 0, 0, 1, 0, 1, 0];
        let c3 = cubic_higher_order(&train, 0.001, 5);
        assert_eq!(c3.len(), 25);
    }

    #[test]
    fn test_c3_zero_lag_positive() {
        // C3(0,0) = E[x^3] which is positive for sparse spikes (skewed)
        let train = vec![0i32, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let c3 = cubic_higher_order(&train, 0.001, 3);
        assert!(
            c3[0] > 0.0,
            "C3(0,0) should be positive for skewed data, got {}",
            c3[0]
        );
    }

    #[test]
    fn test_c3_constant_zero() {
        let train = vec![0i32; 100];
        let c3 = cubic_higher_order(&train, 0.001, 5);
        assert!(c3.iter().all(|&v| v.abs() < 1e-10), "constant → C3 = 0");
    }
}
