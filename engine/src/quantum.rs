// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust Quantum Annealing Acceleration

//! High-performance quantum annealing primitives.
//!
//! Accelerates the hot paths in the Python `quantum_annealing` bridge:
//! - **Simulated annealing**: Metropolis-Hastings with exponential schedule
//! - **Ising energy**: Vectorized energy evaluation
//! - **Gauge transform**: Batch gauge generation and application
//! - **Problem decomposition**: Greedy graph partitioning

pub(crate) mod bindings;

use rayon::prelude::*;
use std::collections::HashMap;

// ── Ising energy ─────────────────────────────────────────────────────

/// Compute Ising energy for a spin configuration.
///
/// H = Σ h_i·s_i + Σ J_ij·s_i·s_j + offset
pub fn ising_energy(
    h: &[(usize, f64)],
    j: &[((usize, usize), f64)],
    spins: &[i8],
    offset: f64,
) -> f64 {
    let mut e = offset;
    for &(i, hi) in h {
        if i < spins.len() {
            e += hi * spins[i] as f64;
        }
    }
    for &((i, j_idx), jij) in j {
        if i < spins.len() && j_idx < spins.len() {
            e += jij * spins[i] as f64 * spins[j_idx] as f64;
        }
    }
    e
}

/// Batch evaluate energies for many configurations (rayon-parallelized).
pub fn batch_ising_energy(
    h: &[(usize, f64)],
    j: &[((usize, usize), f64)],
    configs: &[Vec<i8>],
    offset: f64,
) -> Vec<f64> {
    configs
        .par_iter()
        .map(|spins| ising_energy(h, j, spins, offset))
        .collect()
}

// ── Simulated annealing ──────────────────────────────────────────────

/// Delta energy ΔE = E_after - E_before for flipping spin `qubit`
/// from `s_q` to `-s_q` in the Ising model H = Σ h_i s_i + Σ J_ij s_i s_j.
///
/// ΔE = −2·s_q·(h_q + Σ_k J_qk·s_k).
#[inline]
fn delta_energy(
    h: &[(usize, f64)],
    j_by_qubit: &[Vec<(usize, f64)>],
    spins: &[i8],
    qubit: usize,
) -> f64 {
    let s = spins[qubit] as f64;
    let h_q = h
        .iter()
        .find(|&&(i, _)| i == qubit)
        .map_or(0.0, |&(_, hi)| hi);
    let mut local_field = h_q;

    for &(other, jij) in &j_by_qubit[qubit] {
        local_field += jij * spins[other] as f64;
    }
    -2.0 * s * local_field
}

/// Build adjacency index for fast delta-energy lookup.
fn build_j_index(j: &[((usize, usize), f64)], n: usize) -> Vec<Vec<(usize, f64)>> {
    let mut idx = vec![Vec::new(); n];
    for &((i, j_idx), jij) in j {
        if i < n {
            idx[i].push((j_idx, jij));
        }
        if j_idx < n {
            idx[j_idx].push((i, jij));
        }
    }
    idx
}

/// Run simulated annealing on an Ising model.
///
/// Returns (best_spins, best_energy, all_energies, all_samples).
pub fn simulated_annealing(
    h: &[(usize, f64)],
    j: &[((usize, usize), f64)],
    n_qubits: usize,
    offset: f64,
    n_sweeps: usize,
    num_reads: usize,
    beta_start: f64,
    beta_end: f64,
    seed: u64,
) -> (Vec<i8>, f64, Vec<f64>, Vec<Vec<i8>>) {
    let j_index = build_j_index(j, n_qubits);

    // Parallelized multi-read SA
    let results: Vec<(Vec<i8>, f64)> = (0..num_reads)
        .into_par_iter()
        .map(|read_idx| {
            let mut rng = Xoshiro256pp::new(seed.wrapping_add(read_idx as u64));

            // Random initial config
            let mut spins: Vec<i8> = (0..n_qubits)
                .map(|_| if rng.next_bit() { 1 } else { -1 })
                .collect();
            let mut energy = ising_energy(h, j, &spins, offset);

            for sweep in 0..n_sweeps {
                let beta = beta_start
                    * ((beta_end / beta_start).powf(sweep as f64 / (n_sweeps - 1).max(1) as f64));

                for qubit in 0..n_qubits {
                    let de = delta_energy(h, &j_index, &spins, qubit);

                    if de < 0.0 || rng.next_f64() < (-beta * de).exp() {
                        spins[qubit] *= -1;
                        energy += de;
                    }
                }
            }

            (spins, energy)
        })
        .collect();

    let mut best_energy = f64::INFINITY;
    let mut best_spins = vec![1i8; n_qubits];
    let mut all_energies = Vec::with_capacity(num_reads);
    let mut all_samples = Vec::with_capacity(num_reads);

    for (spins, energy) in results {
        all_energies.push(energy);
        if energy < best_energy {
            best_energy = energy;
            best_spins = spins.clone();
        }
        all_samples.push(spins);
    }

    (best_spins, best_energy, all_energies, all_samples)
}

// ── Gauge transform ──────────────────────────────────────────────────

/// Apply a gauge transform to Ising biases and couplings.
///
/// h'_i = g_i · h_i, J'_ij = g_i · g_j · J_ij
#[allow(clippy::type_complexity)] // tuple shape mirrors Python's QUBO format
pub fn gauge_transform(
    h: &[(usize, f64)],
    j: &[((usize, usize), f64)],
    gauge: &[i8],
) -> (Vec<(usize, f64)>, Vec<((usize, usize), f64)>) {
    let h_new: Vec<(usize, f64)> = h
        .iter()
        .map(|&(i, hi)| {
            let g = if i < gauge.len() {
                gauge[i] as f64
            } else {
                1.0
            };
            (i, g * hi)
        })
        .collect();

    let j_new: Vec<((usize, usize), f64)> = j
        .iter()
        .map(|&((i, j_idx), jij)| {
            let gi = if i < gauge.len() {
                gauge[i] as f64
            } else {
                1.0
            };
            let gj = if j_idx < gauge.len() {
                gauge[j_idx] as f64
            } else {
                1.0
            };
            ((i, j_idx), gi * gj * jij)
        })
        .collect();

    (h_new, j_new)
}

/// Generate n_gauges random gauge vectors.
pub fn generate_gauges(n_qubits: usize, n_gauges: usize, seed: u64) -> Vec<Vec<i8>> {
    (0..n_gauges)
        .into_par_iter()
        .map(|g| {
            let mut rng = Xoshiro256pp::new(seed.wrapping_add(g as u64 * 7919));
            (0..n_qubits)
                .map(|_| if rng.next_bit() { 1 } else { -1 })
                .collect()
        })
        .collect()
}

// ── Graph partitioning ───────────────────────────────────────────────

/// Greedy graph partitioning for problem decomposition.
///
/// Returns list of partitions, each a Vec of qubit indices.
pub fn greedy_partition(
    n_qubits: usize,
    j: &[((usize, usize), f64)],
    max_partition_size: usize,
) -> Vec<Vec<usize>> {
    // Build adjacency
    let mut neighbors: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for &((i, j_idx), jij) in j {
        neighbors.entry(i).or_default().push((j_idx, jij.abs()));
        neighbors.entry(j_idx).or_default().push((i, jij.abs()));
    }

    let mut remaining: Vec<bool> = vec![true; n_qubits];
    let mut partitions: Vec<Vec<usize>> = Vec::new();

    let mut n_remaining = n_qubits;
    while n_remaining > 0 {
        // Find first remaining
        let seed = remaining.iter().position(|&r| r).unwrap();
        let mut partition = vec![seed];
        remaining[seed] = false;
        n_remaining -= 1;

        while partition.len() < max_partition_size && n_remaining > 0 {
            let mut best: Option<usize> = None;
            let mut best_score = -1.0f64;

            for &q in &partition {
                if let Some(nbrs) = neighbors.get(&q) {
                    for &(n, score) in nbrs {
                        if n < n_qubits && remaining[n] && score > best_score {
                            best = Some(n);
                            best_score = score;
                        }
                    }
                }
            }

            let chosen = best.unwrap_or_else(|| remaining.iter().position(|&r| r).unwrap());

            partition.push(chosen);
            remaining[chosen] = false;
            n_remaining -= 1;
        }

        partitions.push(partition);
    }

    partitions
}

// ── Minimal xoshiro256++ PRNG ────────────────────────────────────────

struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    fn new(seed: u64) -> Self {
        // SplitMix64 seeding
        let mut s = [0u64; 4];
        let mut z = seed;
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    #[inline]
    fn next_bit(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    type HTerms = Vec<(usize, f64)>;
    type JTerms = Vec<((usize, usize), f64)>;

    fn simple_model() -> (HTerms, JTerms) {
        let h = vec![(0, 0.1), (1, -0.2), (2, 0.0)];
        let j = vec![((0, 1), -1.0), ((1, 2), 0.5)];
        (h, j)
    }

    #[test]
    fn test_ising_energy_all_up() {
        let (h, j) = simple_model();
        let spins = vec![1, 1, 1];
        let e = ising_energy(&h, &j, &spins, 0.0);
        // 0.1 + (-0.2) + 0.0 + (-1.0)*1*1 + 0.5*1*1 = -0.6
        assert!((e - (-0.6)).abs() < 1e-10);
    }

    #[test]
    fn test_ising_energy_mixed() {
        let (h, j) = simple_model();
        let spins = vec![1, -1, 1];
        let e = ising_energy(&h, &j, &spins, 0.0);
        // 0.1 + 0.2 + 0 + (-1.0)*1*(-1) + 0.5*(-1)*1 = 0.3 + 1.0 - 0.5 = 0.8
        assert!((e - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_batch_energy() {
        let (h, j) = simple_model();
        let configs = vec![vec![1, 1, 1], vec![1, -1, 1], vec![-1, -1, -1]];
        let energies = batch_ising_energy(&h, &j, &configs, 0.0);
        assert_eq!(energies.len(), 3);
        assert!((energies[0] - (-0.6)).abs() < 1e-10);
    }

    #[test]
    fn test_sa_finds_ground_state() {
        // Simple ferromagnetic 2-qubit: J < 0
        let h = vec![(0, 0.0), (1, 0.0)];
        let j = vec![((0, 1), -1.0)];
        let (best_spins, best_energy, _, _) =
            simulated_annealing(&h, &j, 2, 0.0, 5000, 100, 0.1, 20.0, 42);
        // Ground state: aligned → energy = -1.0
        assert!(best_energy <= -0.99, "energy = {}", best_energy);
        // Energy tracker must match recomputed energy of the
        // returned spin configuration. This catches sign bugs in
        // delta_energy that previously let the tracker diverge from
        // the true energy.
        let recomputed = ising_energy(&h, &j, &best_spins, 0.0);
        assert!(
            (best_energy - recomputed).abs() < 1e-9,
            "tracker {} != recomputed {}",
            best_energy,
            recomputed
        );
    }

    #[test]
    fn test_sa_planted_ferromagnetic_8q() {
        // 8-qubit fully-ferromagnetic (h_i = -1, J_ij = -1 ∀ edges).
        // Planted GS = all +1; energy = -n - n_edges = -8 - 28 = -36.
        // With the wrong-sign delta bug, the tracker drifts below
        // -36 (impossible) while the returned spins have higher
        // real energy. The recomputed-vs-tracker check pins this.
        let n = 8;
        let h: Vec<(usize, f64)> = (0..n).map(|i| (i, -1.0)).collect();
        let mut j: Vec<((usize, usize), f64)> = Vec::new();
        for i in 0..n {
            for k in (i + 1)..n {
                j.push(((i, k), -1.0));
            }
        }
        let (best_spins, best_energy, _, _) =
            simulated_annealing(&h, &j, n, 0.0, 2000, 50, 0.1, 10.0, 42);
        let recomputed = ising_energy(&h, &j, &best_spins, 0.0);
        assert!(
            (best_energy - recomputed).abs() < 1e-9,
            "tracker {} != recomputed {}",
            best_energy,
            recomputed
        );
        // Best energy must be ≥ planted GS (cannot be lower).
        assert!(
            best_energy >= -36.0 - 1e-9,
            "best_energy={} below planted GS=-36",
            best_energy
        );
        // SA should land at or near the planted GS.
        assert!(
            best_energy <= -35.0,
            "SA failed to reach planted GS: {}",
            best_energy
        );
    }

    #[test]
    fn test_sa_returns_correct_counts() {
        let h = vec![(0, 0.0)];
        let j = vec![];
        let (_, _, energies, samples) = simulated_annealing(&h, &j, 1, 0.0, 100, 5, 0.1, 5.0, 42);
        assert_eq!(energies.len(), 5);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_gauge_transform_preserves_magnitudes() {
        let h = vec![(0, 1.0), (1, -2.0)];
        let j = vec![((0, 1), 0.5)];
        let gauge = vec![1, -1];
        let (h_new, j_new) = gauge_transform(&h, &j, &gauge);
        assert!((h_new[0].1 - 1.0).abs() < 1e-10);
        assert!((h_new[1].1 - 2.0).abs() < 1e-10); // -1 * -2 = 2
        assert!((j_new[0].1 - (-0.5)).abs() < 1e-10); // 1 * -1 * 0.5 = -0.5
    }

    #[test]
    fn test_generate_gauges() {
        let gauges = generate_gauges(5, 10, 42);
        assert_eq!(gauges.len(), 10);
        for g in &gauges {
            assert_eq!(g.len(), 5);
            for &v in g {
                assert!(v == 1 || v == -1);
            }
        }
    }

    #[test]
    fn test_greedy_partition_small() {
        let j = vec![((0, 1), -1.0), ((1, 2), 0.5)];
        let parts = greedy_partition(3, &j, 100);
        assert_eq!(parts.len(), 1); // fits in one partition
    }

    #[test]
    fn test_greedy_partition_forced_split() {
        let j = vec![((0, 1), -1.0), ((1, 2), 0.5), ((2, 3), 0.8), ((3, 4), 0.3)];
        let parts = greedy_partition(5, &j, 2);
        assert!(parts.len() >= 3);
        for p in &parts {
            assert!(p.len() <= 2);
        }
    }

    #[test]
    fn test_xoshiro_deterministic() {
        let mut rng1 = Xoshiro256pp::new(42);
        let mut rng2 = Xoshiro256pp::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xoshiro_f64_range() {
        let mut rng = Xoshiro256pp::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
