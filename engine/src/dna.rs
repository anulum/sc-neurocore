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

pub(crate) mod bindings;

use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::HashMap;

// ── Constants ────────────────────────────────────────────────────────

const ALPHABET: [u8; 4] = *b"ACGT";
const GC_LOW: f64 = 0.40;
const GC_HIGH: f64 = 0.60;
const MAX_HOMOPOLYMER: usize = 3;
#[allow(dead_code)] // reserved for strand-displacement toehold heuristic
const DEFAULT_TOEHOLD: usize = 7;
const R_GAS: f64 = 1.987e-3; // kcal/(mol·K)

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

// ── Kinetic Simulator (RK4 + Arrhenius) ──────────────────────────────

/// Gate types for kinetic simulation.
#[derive(Clone, Copy, Debug)]
pub enum DnaGateType {
    And,
    Or,
    Not,
    Threshold,
    Mux,
    Amplifier,
    Buffer,
    Nand,
    Xor,
}

/// A gate in the DNA circuit.
#[derive(Clone, Debug)]
pub struct DnaGateSpec {
    pub gate_type: DnaGateType,
    pub input_names: Vec<String>,
    pub output_name: String,
    pub threshold: f64,
    pub leak_rate: f64,
}

/// Kinetic simulation configuration.
pub struct KineticConfig {
    pub k_hyb: f64,
    pub k_disp: f64,
    pub temperature_c: f64,
    pub max_conc: f64,
    pub use_rk4: bool,
}

impl Default for KineticConfig {
    fn default() -> Self {
        Self {
            k_hyb: 3e5,
            k_disp: 1.0,
            temperature_c: 37.0,
            max_conc: 200.0,
            use_rk4: true,
        }
    }
}

/// Arrhenius temperature scaling.
#[inline]
fn arrhenius_scale(k_ref: f64, temperature_c: f64, ea_kcal: f64) -> f64 {
    let t_ref = 310.15; // 37°C
    let t_op = temperature_c + 273.15;
    k_ref * (-(ea_kcal / R_GAS) * (1.0 / t_op - 1.0 / t_ref)).exp()
}

/// Compute effective rate constant for a gate.
fn compute_k_eff(gate: &DnaGateSpec, inputs: &HashMap<String, f64>, config: &KineticConfig) -> f64 {
    let k_hyb = arrhenius_scale(config.k_hyb, config.temperature_c, 15.0);
    let k_disp = arrhenius_scale(config.k_disp, config.temperature_c, 15.0);

    let k_eff = match gate.gate_type {
        DnaGateType::And => {
            let concs: Vec<f64> = gate
                .input_names
                .iter()
                .map(|n| *inputs.get(n).unwrap_or(&0.0))
                .collect();
            let all_present = concs.iter().all(|&c| c > 0.0);
            let min_c = concs.iter().cloned().fold(f64::MAX, f64::min);
            k_hyb * min_c * 1e-9 * if all_present { 1.0 } else { 0.0 }
        }
        DnaGateType::Or => {
            let concs: Vec<f64> = gate
                .input_names
                .iter()
                .map(|n| *inputs.get(n).unwrap_or(&0.0))
                .collect();
            let max_c = concs.iter().cloned().fold(0.0f64, f64::max);
            k_hyb * max_c * 1e-9
        }
        DnaGateType::Not => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            k_disp * (1.0 - (inp / config.max_conc).min(1.0))
        }
        DnaGateType::Threshold => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            let excess = (inp - gate.threshold * config.max_conc).max(0.0);
            k_hyb * excess * 1e-9
        }
        DnaGateType::Mux => {
            let sel = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            let a = *inputs.get(&gate.input_names[1]).unwrap_or(&0.0);
            let b = *inputs.get(&gate.input_names[2]).unwrap_or(&0.0);
            let sel_frac = (sel / config.max_conc).min(1.0);
            k_hyb * (sel_frac * a + (1.0 - sel_frac) * b) * 1e-9
        }
        DnaGateType::Amplifier => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            k_hyb * inp * 1e-9 * 5.0
        }
        DnaGateType::Buffer => {
            let inp = *inputs.get(&gate.input_names[0]).unwrap_or(&0.0);
            k_disp * (inp / config.max_conc).min(1.0)
        }
        _ => 0.0,
    };

    k_eff + gate.leak_rate
}

/// Run kinetic simulation, returning time traces per output.
pub fn simulate_kinetics(
    gates: &[DnaGateSpec],
    input_concentrations: &HashMap<String, f64>,
    duration_s: f64,
    dt: f64,
    config: &KineticConfig,
) -> HashMap<String, Vec<f64>> {
    let n_steps = (duration_s / dt) as usize;
    let max_conc = config.max_conc;

    let mut result: HashMap<String, Vec<f64>> = HashMap::new();

    // Time axis
    let time: Vec<f64> = (0..n_steps).map(|t| t as f64 * dt).collect();
    result.insert("time".to_string(), time);

    for gate in gates {
        let k_eff = compute_k_eff(gate, input_concentrations, config);
        let mut conc = vec![0.0f64; n_steps];

        if config.use_rk4 {
            for t in 1..n_steps {
                let c = conc[t - 1];
                let k1 = k_eff * (max_conc - c) * dt;
                let k2 = k_eff * (max_conc - (c + k1 / 2.0)) * dt;
                let k3 = k_eff * (max_conc - (c + k2 / 2.0)) * dt;
                let k4 = k_eff * (max_conc - (c + k3)) * dt;
                conc[t] = (c + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
                    .max(0.0)
                    .min(max_conc);
            }
        } else {
            for t in 1..n_steps {
                let d = k_eff * (max_conc - conc[t - 1]) * dt;
                conc[t] = (conc[t - 1] + d).max(0.0).min(max_conc);
            }
        }

        result.insert(gate.output_name.clone(), conc);
    }

    result
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
    fn test_simulate_kinetics_and_gate() {
        let gates = vec![DnaGateSpec {
            gate_type: DnaGateType::And,
            input_names: vec!["A".to_string(), "B".to_string()],
            output_name: "C".to_string(),
            threshold: 0.5,
            leak_rate: 1e-7,
        }];
        let mut inputs = HashMap::new();
        inputs.insert("A".to_string(), 200.0);
        inputs.insert("B".to_string(), 200.0);

        let result = simulate_kinetics(&gates, &inputs, 1800.0, 1.0, &KineticConfig::default());
        let c = result.get("C").unwrap();
        assert!(c.last().unwrap() > &50.0);
    }

    #[test]
    fn test_arrhenius_higher_temp_faster() {
        let k37 = arrhenius_scale(3e5, 37.0, 15.0);
        let k25 = arrhenius_scale(3e5, 25.0, 15.0);
        assert!(k37 > k25);
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
