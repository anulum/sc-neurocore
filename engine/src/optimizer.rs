// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust SC-Optimizer Acceleration

//! Accelerates SC-Optimizer hot paths:
//! - Simulated annealing design-space search
//! - Pareto frontier extraction (O(N²) dominance check)
//! - Resource estimation batch evaluation

use rayon::prelude::*;

// ── Resource estimation ──────────────────────────────────────────────

/// Pre-computed candidate config for one layer.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub bitstream_length: u32,
    pub decorrelator: u8, // 0=None,1=LFSR,2=Sobol,3=Halton,4=SCC
    pub mode: u8,         // 0=SC, 1=Deterministic, 2=Hybrid
    pub luts: i64,
    pub power: f64,
    pub accuracy: f64,
    pub latency: i64,
}

/// Estimate resources for a single (mac_count, length, decorr, mode) tuple.
pub fn estimate_resources(mac_count: i64, length: u32, decorr: u8, mode: u8) -> Candidate {
    let len = length as f64;
    let log2_len = (len.log2()).floor() as i64;

    if mode == 1 {
        // Deterministic
        return Candidate {
            bitstream_length: length,
            decorrelator: decorr,
            mode,
            luts: mac_count * 120,
            power: mac_count as f64 * 0.5,
            accuracy: 1.0,
            latency: 1,
        };
    }

    if mode == 2 {
        // Hybrid
        let sc_frac = 0.7_f64;
        let det_frac = 0.3_f64;
        let sc_macs = (mac_count as f64 * sc_frac) as i64;
        let det_macs = (mac_count as f64 * det_frac) as i64;
        let mut luts = sc_macs * 2 + log2_len * 5 + det_macs * 120;
        let power = sc_macs as f64 * 0.01 * (len / 256.0) + det_macs as f64 * 0.5;
        let mut accuracy = 0.95_f64;

        match decorr {
            2 => {
                luts += (sc_macs as f64 * 15.0) as i64;
                accuracy = 0.97;
            }
            1 => {
                luts += 16;
                accuracy = 0.96;
            }
            _ => {}
        }

        return Candidate {
            bitstream_length: length,
            decorrelator: decorr,
            mode,
            luts,
            power,
            accuracy: accuracy.min(1.0),
            latency: length as i64,
        };
    }

    // SC mode
    let mut luts = mac_count * 2 + log2_len * 5;
    let power = mac_count as f64 * 0.01 * (len / 256.0);

    let accuracy = match decorr {
        2 => {
            luts += mac_count * 15;
            1.0 - 1.0 / len
        } // Sobol
        3 => {
            luts += mac_count * 12;
            1.0 - 1.2 / len
        } // Halton
        4 => {
            luts += mac_count * 8;
            1.0 - 1.5 / len
        } // SCC
        1 => {
            luts += 16;
            1.0 - 1.0 / len.sqrt()
        } // LFSR
        _ => 1.0 - 2.0 / len.sqrt(), // None
    };

    Candidate {
        bitstream_length: length,
        decorrelator: decorr,
        mode,
        luts,
        power,
        accuracy: accuracy.clamp(0.1, 1.0),
        latency: length as i64,
    }
}

/// Batch-generate all candidates for a layer.
pub fn generate_candidates(mac_count: i64) -> Vec<Candidate> {
    let lengths: &[u32] = &[64, 128, 256, 512, 1024, 2048];
    let decorrs: &[u8] = &[0, 1, 2, 3, 4];
    let mut result = Vec::with_capacity(1 + lengths.len() * decorrs.len() * 2);

    // Deterministic
    result.push(estimate_resources(mac_count, 1, 0, 1));

    // SC + Hybrid
    for &mode in &[0u8, 2] {
        for &length in lengths {
            for &decorr in decorrs {
                result.push(estimate_resources(mac_count, length, decorr, mode));
            }
        }
    }
    result
}

// ── Simulated annealing ──────────────────────────────────────────────

/// PRNG (xoshiro256++)
struct Xoshiro256pp([u64; 4]);

impl Xoshiro256pp {
    fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        for slot in s.iter_mut() {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = z ^ (z >> 31);
        }
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.0[0].wrapping_add(self.0[3]))
            .rotate_left(23)
            .wrapping_add(self.0[0]);
        let t = self.0[1] << 17;
        self.0[2] ^= self.0[0];
        self.0[3] ^= self.0[1];
        self.0[1] ^= self.0[2];
        self.0[0] ^= self.0[3];
        self.0[2] ^= t;
        self.0[3] = self.0[3].rotate_left(45);
        result
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// SA result.
#[derive(Clone, Debug)]
pub struct SAResult {
    pub best_config: Vec<usize>, // index into candidates per layer
    pub best_score: f64,
    pub pareto_luts: Vec<i64>,
    pub pareto_power: Vec<f64>,
    pub pareto_score: Vec<f64>,
}

/// Run SA design-space search.
///
/// * `candidates` — Vec of candidate lists, one per layer
/// * `weights` — per-layer scoring weight (2.0 for critical, 1.0 otherwise)
/// * `max_luts`, `max_power`, `max_latency` — budget constraints
pub fn simulated_annealing(
    candidates: &[Vec<Candidate>],
    weights: &[f64],
    max_luts: i64,
    max_power: f64,
    max_latency: i64,
    t_init: f64,
    t_min: f64,
    alpha: f64,
    max_iter: usize,
    seed: u64,
) -> Option<SAResult> {
    let n = candidates.len();
    let mut rng = Xoshiro256pp::new(seed);

    // Start from cheapest
    let mut current: Vec<usize> = (0..n)
        .map(|i| {
            candidates[i]
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.luts)
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();

    if !is_feasible(candidates, &current, max_luts, max_power, max_latency) {
        return None;
    }

    let weight_sum: f64 = weights.iter().sum();
    let score_fn = |cfg: &[usize]| -> f64 {
        let mut total = 0.0;
        for i in 0..n {
            total += candidates[i][cfg[i]].accuracy * weights[i];
        }
        total / weight_sum
    };

    let mut best = current.clone();
    let mut best_score = score_fn(&best);
    let mut current_score = best_score;
    let mut t = t_init;

    let mut pareto_luts = Vec::new();
    let mut pareto_power = Vec::new();
    let mut pareto_score = Vec::new();

    for _ in 0..max_iter {
        if t <= t_min {
            break;
        }

        let layer = rng.next_usize(n);
        let cand_idx = rng.next_usize(candidates[layer].len());

        let old = current[layer];
        current[layer] = cand_idx;

        if !is_feasible(candidates, &current, max_luts, max_power, max_latency) {
            current[layer] = old;
            t *= alpha;
            continue;
        }

        let trial_score = score_fn(&current);
        let delta = trial_score - current_score;

        if delta > 0.0 || rng.next_f64() < (delta / t).exp() {
            current_score = trial_score;

            if current_score > best_score {
                best = current.clone();
                best_score = current_score;
            }

            let luts: i64 = (0..n).map(|i| candidates[i][current[i]].luts).sum();
            let power: f64 = (0..n).map(|i| candidates[i][current[i]].power).sum();
            pareto_luts.push(luts);
            pareto_power.push(power);
            pareto_score.push(current_score);
        } else {
            current[layer] = old;
        }

        t *= alpha;
    }

    Some(SAResult {
        best_config: best,
        best_score,
        pareto_luts,
        pareto_power,
        pareto_score,
    })
}

fn is_feasible(
    candidates: &[Vec<Candidate>],
    config: &[usize],
    max_luts: i64,
    max_power: f64,
    max_latency: i64,
) -> bool {
    let n = candidates.len();
    let mut total_luts: i64 = 0;
    let mut total_power: f64 = 0.0;
    let mut max_lat: i64 = 0;
    for i in 0..n {
        let c = &candidates[i][config[i]];
        total_luts += c.luts;
        total_power += c.power;
        if c.latency > max_lat {
            max_lat = c.latency;
        }
    }
    total_luts <= max_luts
        && total_power <= max_power
        && (max_latency <= 0 || max_lat <= max_latency)
}

// ── Pareto extraction ────────────────────────────────────────────────

/// Extract non-dominated Pareto frontier (parallelized for large N).
pub fn extract_pareto(luts: &[i64], power: &[f64], score: &[f64]) -> Vec<usize> {
    let n = luts.len();
    if n == 0 {
        return vec![];
    }

    let indices: Vec<usize> = (0..n)
        .into_par_iter()
        .filter(|&i| {
            for j in 0..n {
                if j == i {
                    continue;
                }
                if luts[j] <= luts[i]
                    && power[j] <= power[i]
                    && score[j] >= score[i]
                    && (luts[j] < luts[i] || power[j] < power[i] || score[j] > score[i])
                {
                    return false;
                }
            }
            true
        })
        .collect();

    indices
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_deterministic() {
        let c = estimate_resources(100, 1, 0, 1);
        assert_eq!(c.luts, 12000);
        assert!((c.accuracy - 1.0).abs() < 1e-10);
        assert_eq!(c.latency, 1);
    }

    #[test]
    fn test_estimate_sc_sobol() {
        let c = estimate_resources(10, 256, 2, 0);
        assert!(c.luts > 0);
        assert!(c.accuracy > 0.99);
    }

    #[test]
    fn test_estimate_hybrid() {
        let c = estimate_resources(10, 256, 2, 2);
        assert!((c.accuracy - 0.97).abs() < 1e-10);
    }

    #[test]
    fn test_generate_candidates_count() {
        let cands = generate_candidates(10);
        // 1 deterministic + 6 lengths × 5 decorrs × 2 modes = 61
        assert_eq!(cands.len(), 61);
    }

    #[test]
    fn test_sa_finds_solution() {
        let cands: Vec<Vec<Candidate>> = (0..3).map(|_| generate_candidates(10)).collect();
        let weights = vec![1.0, 1.0, 2.0];
        let result = simulated_annealing(
            &cands, &weights, 100_000, 1000.0, 0, 1.0, 0.001, 0.95, 500, 42,
        );
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.best_score > 0.5);
    }

    #[test]
    fn test_sa_infeasible() {
        let cands: Vec<Vec<Candidate>> = (0..3).map(|_| generate_candidates(10)).collect();
        let weights = vec![1.0, 1.0, 1.0];
        // Budget too tight for even cheapest
        let result = simulated_annealing(&cands, &weights, 1, 0.001, 0, 1.0, 0.001, 0.95, 100, 42);
        assert!(result.is_none());
    }

    #[test]
    fn test_pareto_single_point() {
        let indices = extract_pareto(&[100], &[1.0], &[0.9]);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_pareto_dominated() {
        // Point 1 dominates point 0 (lower luts+power, higher score)
        let indices = extract_pareto(&[200, 100], &[2.0, 1.0], &[0.8, 0.9]);
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn test_pareto_both_nondominated() {
        // Neither dominates: lower luts but lower score
        let indices = extract_pareto(&[100, 200], &[1.0, 2.0], &[0.8, 0.95]);
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_xoshiro_reproducible() {
        let mut rng1 = Xoshiro256pp::new(42);
        let mut rng2 = Xoshiro256pp::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
