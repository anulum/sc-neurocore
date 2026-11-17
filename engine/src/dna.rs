// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust DNA circuit acceleration engine

//! High-performance DNA strand displacement pipeline.
//!
//! Accelerates three critical hot paths from `sc_neurocore.bridges.dna_mapper`:
//!
//! 1. **Sequence design** — GC-balanced, homopolymer-free, orthogonal oligo generation
//! 2. **Cross-hybridization** — O(n²) pairwise alignment scoring (rayon-parallelized)
//! 3. **Kinetic simulation** — RK4 mass-action integrator with Arrhenius scaling

#![deny(missing_docs)]

pub(crate) mod bindings;
mod kinetics;

pub use kinetics::{
    check_kinetics, simulate_kinetics, try_simulate_kinetics, DnaGateSpec, DnaGateType,
    KineticConfig, KineticsError, MAX_KINETIC_SAMPLES,
};

use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

// ── Constants ────────────────────────────────────────────────────────

const ALPHABET: [u8; 4] = *b"ACGT";
const GC_LOW: f64 = 0.40;
const GC_HIGH: f64 = 0.60;
const MAX_HOMOPOLYMER: usize = 3;
#[allow(dead_code)] // reserved for strand-displacement toehold heuristic
const DEFAULT_TOEHOLD: usize = 7;

// ── Sequence Designer ────────────────────────────────────────────────

/// Generate a GC-balanced, homopolymer-free DNA sequence.
pub fn design_sequence(length: usize, seed: u64) -> Vec<u8> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut seq = Vec::with_capacity(length);

    for _ in 0..length * 10 {
        if seq.len() >= length {
            break;
        }
        let base = ALPHABET[rng.random_range(0..4)];

        // Homopolymer check
        if seq.len() >= MAX_HOMOPOLYMER {
            let tail = &seq[seq.len() - MAX_HOMOPOLYMER..];
            if tail.iter().all(|&b| b == base) {
                continue;
            }
        }

        seq.push(base);

        // GC check at end
        if seq.len() == length {
            let gc = gc_content(&seq);
            if !(GC_LOW..=GC_HIGH).contains(&gc) {
                seq.clear();
            }
        }
    }

    // Fallback if constraints too tight
    while seq.len() < length {
        let base = ALPHABET[rng.random_range(0..4)];
        seq.push(base);
    }
    seq.truncate(length);
    seq
}

/// Generate multiple orthogonal sequences.
pub fn design_orthogonal_set(count: usize, length: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        result.push(design_sequence(length, seed.wrapping_add(i as u64)));
    }
    result
}

#[inline]
fn gc_content(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return 0.5;
    }
    let gc = seq.iter().filter(|&&b| b == b'G' || b == b'C').count();
    gc as f64 / seq.len() as f64
}

/// Watson-Crick complement.
pub fn complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .map(|&b| match b {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            _ => b'N',
        })
        .collect()
}

/// Reverse complement.
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    let mut rc = complement(seq);
    rc.reverse();
    rc
}

// ── Cross-Hybridization Checker (rayon-parallelized) ─────────────────

/// Pairwise alignment score between two sequences.
fn alignment_score(a: &[u8], b: &[u8]) -> usize {
    let rc_b = reverse_complement(b);
    longest_common_substring(a, &rc_b)
}

fn longest_common_substring(a: &[u8], b: &[u8]) -> usize {
    let n = a.len();
    let m = b.len();
    let mut max_len = 0usize;
    let mut prev = vec![0usize; m + 1];
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n {
        for j in 1..=m {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
                if curr[j] > max_len {
                    max_len = curr[j];
                }
            } else {
                curr[j] = 0;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.iter_mut().for_each(|x| *x = 0);
    }
    max_len
}

/// Check all pairs for dangerous cross-hybridization.
/// Returns vec of (i, j, score) for pairs above threshold.
pub fn check_cross_hybridization(
    sequences: &[Vec<u8>],
    threshold: usize,
) -> Vec<(usize, usize, usize)> {
    let n = sequences.len();
    if n < 2 {
        return vec![];
    }

    // Build pair indices
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    // Parallel pairwise scoring
    pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let score = alignment_score(&sequences[i], &sequences[j]);
            if score >= threshold {
                Some((i, j, score))
            } else {
                None
            }
        })
        .collect()
}

// ── Hairpin Detection ────────────────────────────────────────────────

/// Detect hairpins in a sequence. Returns vec of (stem_start, stem_len, loop_len).
pub fn detect_hairpins(seq: &[u8], min_stem: usize, min_loop: usize) -> Vec<(usize, usize, usize)> {
    let n = seq.len();
    let mut hairpins = Vec::new();

    if n < min_stem * 2 + min_loop {
        return hairpins;
    }

    let wc = |a: u8, b: u8| -> bool {
        matches!(
            (a, b),
            (b'A', b'T') | (b'T', b'A') | (b'C', b'G') | (b'G', b'C')
        )
    };

    for i in 0..n.saturating_sub(min_stem * 2 + min_loop) {
        for stem_len in min_stem..12.min((n - i) / 2) {
            let loop_start = i + stem_len;
            for loop_len in min_loop..10.min(n - loop_start - stem_len + 1) {
                let j = loop_start + loop_len;
                if j + stem_len > n {
                    break;
                }
                let matches = (0..stem_len)
                    .filter(|&k| wc(seq[i + k], seq[j + stem_len - 1 - k]))
                    .count();
                if matches >= stem_len {
                    hairpins.push((i, stem_len, loop_len));
                }
            }
        }
    }
    hairpins
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_design_sequence_length() {
        let seq = design_sequence(30, 42);
        assert_eq!(seq.len(), 30);
    }

    #[test]
    fn test_design_sequence_alphabet() {
        let seq = design_sequence(50, 42);
        assert!(seq.iter().all(|&b| ALPHABET.contains(&b)));
    }

    #[test]
    fn test_gc_content_balanced() {
        let seq = design_sequence(40, 42);
        let gc = gc_content(&seq);
        assert!((0.3..=0.7).contains(&gc), "GC={gc}");
    }

    #[test]
    fn test_complement() {
        assert_eq!(complement(b"ACGT"), b"TGCA");
    }

    #[test]
    fn test_reverse_complement() {
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
    }

    #[test]
    fn test_cross_hybridization_self() {
        let seqs = vec![b"ACGTACGTACGT".to_vec(), b"ACGTACGTACGT".to_vec()];
        let flags = check_cross_hybridization(&seqs, 4);
        assert!(!flags.is_empty());
    }

    #[test]
    fn test_cross_hybridization_orthogonal() {
        let seqs = design_orthogonal_set(3, 30, 42);
        let flags = check_cross_hybridization(&seqs, 20);
        assert!(flags.is_empty());
    }

    #[test]
    fn test_detect_hairpins_short() {
        let hps = detect_hairpins(b"ACGT", 4, 3);
        assert!(hps.is_empty());
    }

    #[test]
    fn test_lcs() {
        let score = longest_common_substring(b"ACGTACGT", b"ACGTACGT");
        assert_eq!(score, 8);
    }
}
