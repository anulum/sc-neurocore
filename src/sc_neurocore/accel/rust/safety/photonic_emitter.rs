// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed photonic crosstalk reference kernel

//! Stand-alone Rust implementation of the public coupled-waveguide contract.
//!
//! This file deliberately mirrors only the numeric crosstalk hot path. FDTD,
//! Meep orchestration, netlist emission, and filesystem export remain in their
//! canonical Python responsibility modules.

use std::fmt;

const ISOLATION_CEILING_DB: f64 = 300.0;
const ISOLATION_RATIO_FLOOR: f64 = 1.0e-15;

/// Invalid physical input supplied to a crosstalk kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrosstalkError(&'static str);

impl fmt::Display for CrosstalkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for CrosstalkError {}

/// One arbitrary waveguide-pair request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairSpec {
    pub index_a: usize,
    pub index_b: usize,
    pub gap_nm: f64,
    pub coupling_length_um: f64,
}

/// Coupled-mode result for one waveguide pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairMetrics {
    pub coupling_coefficient_per_um: f64,
    pub coupling_ratio: f64,
    pub isolation_db: f64,
}

/// Aggregate metrics for a uniform bank of parallel waveguides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BankMetrics {
    pub num_waveguides: usize,
    pub num_near_pairs: usize,
    pub num_far_pairs: usize,
    pub adjacent: PairMetrics,
    pub next_nearest: PairMetrics,
    pub worst_isolation_db: f64,
    pub mean_coupling_ratio: f64,
    pub max_coupling_ratio: f64,
    pub crosstalk_safe: bool,
}

fn require_non_negative(value: f64, message: &'static str) -> Result<(), CrosstalkError> {
    if !value.is_finite() || value < 0.0 {
        return Err(CrosstalkError(message));
    }
    Ok(())
}

fn validate_material(
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> Result<(), CrosstalkError> {
    if !wavelength_nm.is_finite() || wavelength_nm <= 0.0 {
        return Err(CrosstalkError("wavelength_nm must be finite and positive"));
    }
    if !core_index.is_finite() || core_index <= 0.0 {
        return Err(CrosstalkError("core_index must be finite and positive"));
    }
    if !cladding_index.is_finite() || cladding_index <= 0.0 {
        return Err(CrosstalkError("cladding_index must be finite and positive"));
    }
    if core_index <= cladding_index {
        return Err(CrosstalkError(
            "core_index must be greater than cladding_index",
        ));
    }
    Ok(())
}

/// Evaluate the Marcatili-form coupled-mode contract for one pair.
pub fn analyze_pair(
    gap_nm: f64,
    coupling_length_um: f64,
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> Result<PairMetrics, CrosstalkError> {
    require_non_negative(gap_nm, "gap_nm must be finite and non-negative")?;
    require_non_negative(
        coupling_length_um,
        "coupling_length_um must be finite and non-negative",
    )?;
    validate_material(wavelength_nm, core_index, cladding_index)?;

    let index_contrast = (core_index * core_index - cladding_index * cladding_index).sqrt();
    let decay_length_nm = wavelength_nm / (2.0 * std::f64::consts::PI * index_contrast);
    let effective_index_difference = 0.1 * (-gap_nm / decay_length_nm).exp();
    let coupling_coefficient_per_um =
        std::f64::consts::PI * effective_index_difference / (wavelength_nm * 1.0e-3);
    let phase = coupling_coefficient_per_um * coupling_length_um;
    let coupling_ratio = phase.sin().powi(2);
    let isolation_db = if coupling_ratio < ISOLATION_RATIO_FLOOR {
        ISOLATION_CEILING_DB
    } else {
        -10.0 * coupling_ratio.log10()
    };

    Ok(PairMetrics {
        coupling_coefficient_per_um,
        coupling_ratio,
        isolation_db,
    })
}

/// Evaluate adjacent and next-nearest coupling in a uniform waveguide bank.
pub fn analyze_bank(
    num_waveguides: usize,
    gap_nm: f64,
    coupling_length_um: f64,
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> Result<BankMetrics, CrosstalkError> {
    if num_waveguides == 0 {
        return Err(CrosstalkError("num_waveguides must be at least one"));
    }
    let adjacent = analyze_pair(
        gap_nm,
        coupling_length_um,
        wavelength_nm,
        core_index,
        cladding_index,
    )?;
    let next_nearest = analyze_pair(
        2.0 * gap_nm,
        coupling_length_um,
        wavelength_nm,
        core_index,
        cladding_index,
    )?;
    let num_near_pairs = num_waveguides.saturating_sub(1);
    let num_far_pairs = num_waveguides.saturating_sub(2);
    let pair_count = num_near_pairs + num_far_pairs;
    let (worst_isolation_db, mean_coupling_ratio, max_coupling_ratio) = if pair_count == 0 {
        (f64::INFINITY, 0.0, 0.0)
    } else {
        (
            adjacent.isolation_db.min(next_nearest.isolation_db),
            (num_near_pairs as f64 * adjacent.coupling_ratio
                + num_far_pairs as f64 * next_nearest.coupling_ratio)
                / pair_count as f64,
            adjacent.coupling_ratio.max(next_nearest.coupling_ratio),
        )
    };

    Ok(BankMetrics {
        num_waveguides,
        num_near_pairs,
        num_far_pairs,
        adjacent,
        next_nearest,
        worst_isolation_db,
        mean_coupling_ratio,
        max_coupling_ratio,
        crosstalk_safe: worst_isolation_db > 20.0,
    })
}

/// Evaluate arbitrary pairs without accepting self-pairs or partial output.
pub fn analyze_pairs(
    pairs: &[PairSpec],
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> Result<Vec<PairMetrics>, CrosstalkError> {
    validate_material(wavelength_nm, core_index, cladding_index)?;
    for pair in pairs {
        if pair.index_a == pair.index_b {
            return Err(CrosstalkError("a pair must name distinct waveguides"));
        }
        require_non_negative(pair.gap_nm, "gap_nm must be finite and non-negative")?;
        require_non_negative(
            pair.coupling_length_um,
            "coupling_length_um must be finite and non-negative",
        )?;
    }
    pairs
        .iter()
        .map(|pair| {
            analyze_pair(
                pair.gap_nm,
                pair.coupling_length_um,
                wavelength_nm,
                core_index,
                cladding_index,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1.0e-15;

    #[test]
    fn golden_pair_matches_the_public_python_contract() {
        let metrics = analyze_pair(200.0, 50.0, 1550.0, 3.48, 1.45).unwrap();
        assert!((metrics.coupling_coefficient_per_um - 0.015593714868342372).abs() < TOLERANCE);
        assert!((metrics.coupling_ratio - 0.49428770428966934).abs() < TOLERANCE);
        assert!((metrics.isolation_db - 3.0602019274692).abs() < TOLERANCE);
    }

    #[test]
    fn zero_length_uses_the_shared_isolation_ceiling() {
        let metrics = analyze_pair(200.0, 0.0, 1550.0, 3.48, 1.45).unwrap();
        assert_eq!(metrics.coupling_ratio, 0.0);
        assert_eq!(metrics.isolation_db, 300.0);
    }

    #[test]
    fn bank_counts_and_empty_pair_statistics_are_exact() {
        let single = analyze_bank(1, 200.0, 50.0, 1550.0, 3.48, 1.45).unwrap();
        assert_eq!(single.num_near_pairs, 0);
        assert_eq!(single.num_far_pairs, 0);
        assert!(single.worst_isolation_db.is_infinite());
        assert_eq!(single.mean_coupling_ratio, 0.0);

        let bank = analyze_bank(5, 200.0, 50.0, 1550.0, 3.48, 1.45).unwrap();
        assert_eq!(bank.num_near_pairs, 4);
        assert_eq!(bank.num_far_pairs, 3);
        assert!(bank.adjacent.coupling_ratio >= bank.next_nearest.coupling_ratio);
    }

    #[test]
    fn invalid_inputs_fail_before_returning_results() {
        assert!(analyze_pair(f64::NAN, 10.0, 1550.0, 3.48, 1.45).is_err());
        assert!(analyze_pair(200.0, -1.0, 1550.0, 3.48, 1.45).is_err());
        assert!(analyze_pair(200.0, 10.0, 0.0, 3.48, 1.45).is_err());
        assert!(analyze_pair(200.0, 10.0, 1550.0, 1.45, 1.45).is_err());
        assert!(analyze_bank(0, 200.0, 10.0, 1550.0, 3.48, 1.45).is_err());
        let self_pair = [PairSpec {
            index_a: 1,
            index_b: 1,
            gap_nm: 200.0,
            coupling_length_um: 10.0,
        }];
        assert!(analyze_pairs(&self_pair, 1550.0, 3.48, 1.45).is_err());
    }
}
