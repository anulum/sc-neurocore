// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust Evolutionary Substrate Acceleration

//! Accelerates evolutionary substrate hot paths:
//! - Batch fitness evaluation (population-parallel)
//! - Genome mutation (weight perturbation + topology)
//! - Crossover (aligned gene matching)
//! - Novelty/diversity metrics
//! - Pareto front maintenance

use rayon::prelude::*;

// ── PRNG ─────────────────────────────────────────────────────────────

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

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn next_gaussian(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Batch weight mutation ────────────────────────────────────────────

/// Mutate weights in-place with Gaussian noise (parallelized for large populations).
///
/// Each genome is a flat f64 slice. mutation_rate is probability per weight.
pub fn batch_mutate_weights(
    genomes: &mut [Vec<f64>],
    mutation_rate: f64,
    mutation_scale: f64,
    seed: u64,
) {
    genomes
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, genome)| {
            let mut rng = Xoshiro256pp::new(seed.wrapping_add(i as u64));
            for w in genome.iter_mut() {
                if rng.next_f64() < mutation_rate {
                    *w += rng.next_gaussian() * mutation_scale;
                    *w = w.clamp(-10.0, 10.0);
                }
            }
        });
}

// ── Batch fitness evaluation ─────────────────────────────────────────

/// Evaluate fitness for a population of weight vectors against a target.
///
/// Fitness = -MSE(genome_weights, target_outputs) for each genome
/// applied to a simple linear evaluation: output = sum(w_i * input_i).
pub fn batch_evaluate_fitness(
    genomes: &[Vec<f64>],
    inputs: &[f64],
    target: f64,
) -> Vec<f64> {
    genomes
        .par_iter()
        .map(|genome| {
            let output: f64 = genome
                .iter()
                .zip(inputs.iter())
                .map(|(w, x)| w * x)
                .sum();
            let error = output - target;
            -(error * error) // negative MSE as fitness
        })
        .collect()
}

// ── Crossover ────────────────────────────────────────────────────────

/// Uniform crossover between two parent genomes (parallelized batch).
pub fn batch_crossover(
    parents_a: &[Vec<f64>],
    parents_b: &[Vec<f64>],
    seed: u64,
) -> Vec<Vec<f64>> {
    parents_a
        .par_iter()
        .zip(parents_b.par_iter())
        .enumerate()
        .map(|(i, (a, b))| {
            let mut rng = Xoshiro256pp::new(seed.wrapping_add(i as u64));
            let len = a.len().min(b.len());
            let mut child = Vec::with_capacity(len);
            for j in 0..len {
                if rng.next_f64() < 0.5 {
                    child.push(a[j]);
                } else {
                    child.push(b[j]);
                }
            }
            child
        })
        .collect()
}

// ── Diversity / Novelty ──────────────────────────────────────────────

/// Compute pairwise diversity (mean L2 distance) for a population.
pub fn population_diversity(genomes: &[Vec<f64>]) -> f64 {
    let n = genomes.len();
    if n < 2 {
        return 0.0;
    }

    let total: f64 = (0..n)
        .into_par_iter()
        .flat_map(|i| ((i + 1)..n).into_par_iter().map(move |j| (i, j)))
        .map(|(i, j)| {
            let len = genomes[i].len().min(genomes[j].len());
            let dist_sq: f64 = (0..len)
                .map(|k| {
                    let d = genomes[i][k] - genomes[j][k];
                    d * d
                })
                .sum();
            dist_sq.sqrt()
        })
        .sum();

    let pairs = (n * (n - 1)) / 2;
    total / pairs as f64
}

/// Compute novelty score for each genome against an archive.
pub fn novelty_scores(
    genomes: &[Vec<f64>],
    archive: &[Vec<f64>],
    k_nearest: usize,
) -> Vec<f64> {
    let combined: Vec<&Vec<f64>> = archive.iter().chain(genomes.iter()).collect();

    genomes
        .par_iter()
        .map(|genome| {
            let mut distances: Vec<f64> = combined
                .iter()
                .filter(|&&other| !std::ptr::eq(other, genome))
                .map(|other| {
                    let len = genome.len().min(other.len());
                    let d: f64 = (0..len)
                        .map(|k| {
                            let diff = genome[k] - other[k];
                            diff * diff
                        })
                        .sum();
                    d.sqrt()
                })
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k = k_nearest.min(distances.len());
            if k == 0 {
                return 0.0;
            }
            distances[..k].iter().sum::<f64>() / k as f64
        })
        .collect()
}

// ── Tournament selection ─────────────────────────────────────────────

/// Tournament selection: select n winners from population by fitness.
pub fn tournament_select(
    fitness: &[f64],
    n_select: usize,
    tournament_size: usize,
    seed: u64,
) -> Vec<usize> {
    let pop_size = fitness.len();
    let mut rng = Xoshiro256pp::new(seed);
    let mut selected = Vec::with_capacity(n_select);

    for _ in 0..n_select {
        let mut best_idx = (rng.next_u64() % pop_size as u64) as usize;
        let mut best_fit = fitness[best_idx];
        for _ in 1..tournament_size {
            let idx = (rng.next_u64() % pop_size as u64) as usize;
            if fitness[idx] > best_fit {
                best_idx = idx;
                best_fit = fitness[idx];
            }
        }
        selected.push(best_idx);
    }

    selected
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_mutate() {
        let mut pop = vec![vec![0.0; 10]; 50];
        batch_mutate_weights(&mut pop, 0.5, 0.1, 42);
        let changed = pop.iter().any(|g| g.iter().any(|&w| w != 0.0));
        assert!(changed);
    }

    #[test]
    fn test_mutate_bounded() {
        let mut pop = vec![vec![100.0; 5]; 10];
        batch_mutate_weights(&mut pop, 1.0, 1.0, 42);
        for g in &pop {
            for &w in g {
                assert!(w >= -10.0 && w <= 10.0);
            }
        }
    }

    #[test]
    fn test_batch_fitness() {
        let genomes = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let inputs = vec![1.0, 0.0];
        let fitness = batch_evaluate_fitness(&genomes, &inputs, 1.0);
        assert!((fitness[0] - 0.0).abs() < 1e-10); // perfect
        assert!(fitness[1] < 0.0); // error
    }

    #[test]
    fn test_crossover_length() {
        let a = vec![vec![1.0; 10]; 20];
        let b = vec![vec![2.0; 10]; 20];
        let children = batch_crossover(&a, &b, 42);
        assert_eq!(children.len(), 20);
        assert_eq!(children[0].len(), 10);
    }

    #[test]
    fn test_crossover_mix() {
        let a = vec![vec![0.0; 100]];
        let b = vec![vec![1.0; 100]];
        let children = batch_crossover(&a, &b, 42);
        let has_zero = children[0].iter().any(|&v| v == 0.0);
        let has_one = children[0].iter().any(|&v| v == 1.0);
        assert!(has_zero && has_one);
    }

    #[test]
    fn test_diversity_identical() {
        let pop = vec![vec![1.0; 5]; 10];
        assert!((population_diversity(&pop) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_diversity_distinct() {
        let pop = vec![vec![0.0; 5], vec![1.0; 5]];
        assert!(population_diversity(&pop) > 0.0);
    }

    #[test]
    fn test_novelty_scores() {
        let genomes = vec![vec![0.0; 5], vec![10.0; 5]];
        let archive = vec![vec![0.0; 5]];
        let scores = novelty_scores(&genomes, &archive, 1);
        assert!(scores[1] > scores[0]); // second is more novel
    }

    #[test]
    fn test_tournament_select() {
        let fitness = vec![0.1, 0.9, 0.5, 0.3];
        let selected = tournament_select(&fitness, 10, 3, 42);
        assert_eq!(selected.len(), 10);
        // Best-fitness individual should appear frequently
        let count_best = selected.iter().filter(|&&i| i == 1).count();
        assert!(count_best > 0);
    }

    #[test]
    fn test_single_genome_diversity() {
        let pop = vec![vec![1.0; 5]];
        assert!((population_diversity(&pop) - 0.0).abs() < 1e-10);
    }
}
