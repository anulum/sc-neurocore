// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust Photonic NoC Acceleration

//! High-performance photonic network-on-chip primitives.
//!
//! Accelerates the hot paths in the Python `photonic_noc` bridge:
//! - **Waveguide routing** — O(N²) Manhattan distance on mesh
//! - **Crosstalk analysis** — O(N²) pairwise channel coupling
//! - **MZI transfer matrix** — 2×2 unitary matrix cascade
//! - **Power budget** — path loss accumulation

use rayon::prelude::*;

// ── Constants ────────────────────────────────────────────────────────

const CROSSING_LOSS_DB: f64 = 0.08;
const MZI_INSERTION_LOSS_DB: f64 = 0.5;

// ── Waveguide routing ────────────────────────────────────────────────

/// A routed waveguide segment.
#[derive(Clone, Debug)]
pub struct WaveguideResult {
    pub source: usize,
    pub target: usize,
    pub length_um: f64,
    pub loss_db: f64,
    pub n_crossings: usize,
}

/// Route waveguides on a mesh topology from an adjacency matrix.
///
/// Returns vector of (source, target, length_um, loss_db, n_crossings).
pub fn route_waveguides(
    adjacency: &[f64],
    n: usize,
    pitch_um: f64,
    loss_db_per_cm: f64,
) -> Vec<WaveguideResult> {
    let grid_size = ((n as f64).sqrt().ceil() as usize).max(1);

    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let w = adjacency[i * n + j].abs() + adjacency[j * n + i].abs();
            if w < 1e-12 {
                return None;
            }

            let (ri, ci) = (i / grid_size, i % grid_size);
            let (rj, cj) = (j / grid_size, j % grid_size);
            let manhattan = ((ri as isize - rj as isize).unsigned_abs()
                + (ci as isize - cj as isize).unsigned_abs()) as usize;

            let length_um = manhattan as f64 * pitch_um;
            let mut loss = length_um * 1e-4 * loss_db_per_cm;
            let n_crossings = if manhattan > 0 { manhattan - 1 } else { 0 };
            loss += n_crossings as f64 * CROSSING_LOSS_DB;

            Some(WaveguideResult {
                source: i,
                target: j,
                length_um,
                loss_db: loss,
                n_crossings,
            })
        })
        .collect()
}

// ── MZI transfer matrix ──────────────────────────────────────────────

/// MZI 2×2 transfer matrix elements (complex).
///
/// M = [[cos(φ/2), i·sin(φ/2)], [i·sin(φ/2), cos(φ/2)]]
///
/// Returns (re_00, im_00, re_01, im_01, re_10, im_10, re_11, im_11)
pub fn mzi_transfer_matrix(phase_rad: f64) -> [f64; 8] {
    let half = phase_rad / 2.0;
    let c = half.cos();
    let s = half.sin();
    // [[c, i·s], [i·s, c]]  →  stored as (re, im) pairs
    [c, 0.0, 0.0, s, 0.0, s, c, 0.0]
}

/// Cascade N MZI stages by multiplying transfer matrices.
///
/// Returns the final 2×2 complex matrix as 8 f64s.
pub fn cascade_mzi(phases: &[f64]) -> [f64; 8] {
    let mut result = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // identity

    for &phase in phases {
        let m = mzi_transfer_matrix(phase);
        result = complex_mat_mul(&result, &m);
    }

    result
}

/// Multiply two 2×2 complex matrices (each stored as 8 f64s).
fn complex_mat_mul(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    // a = [[a00, a01], [a10, a11]], b = [[b00, b01], [b10, b11]]
    // Each element is (re, im) at indices (2k, 2k+1)
    let cmul = |ar: f64, ai: f64, br: f64, bi: f64| -> (f64, f64) {
        (ar * br - ai * bi, ar * bi + ai * br)
    };
    let cadd = |a: (f64, f64), b: (f64, f64)| -> (f64, f64) { (a.0 + b.0, a.1 + b.1) };

    // c00 = a00*b00 + a01*b10
    let c00 = cadd(cmul(a[0], a[1], b[0], b[1]), cmul(a[2], a[3], b[4], b[5]));
    // c01 = a00*b01 + a01*b11
    let c01 = cadd(cmul(a[0], a[1], b[2], b[3]), cmul(a[2], a[3], b[6], b[7]));
    // c10 = a10*b00 + a11*b10
    let c10 = cadd(cmul(a[4], a[5], b[0], b[1]), cmul(a[6], a[7], b[4], b[5]));
    // c11 = a10*b01 + a11*b11
    let c11 = cadd(cmul(a[4], a[5], b[2], b[3]), cmul(a[6], a[7], b[6], b[7]));

    [c00.0, c00.1, c01.0, c01.1, c10.0, c10.1, c11.0, c11.1]
}

// ── Crosstalk analysis ───────────────────────────────────────────────

/// Crosstalk result per channel.
#[derive(Clone, Debug)]
pub struct CrosstalkResult {
    pub channel_id: usize,
    pub wavelength_nm: f64,
    pub n_adjacent: usize,
    pub crosstalk_db: f64,
    pub osnr_db: f64,
}

/// Analyze inter-channel crosstalk for WDM channels.
///
/// channels: list of (channel_id, wavelength_nm, bandwidth_nm, power_dbm)
pub fn analyze_crosstalk(
    channels: &[(usize, f64, f64, f64)],
    adjacent_xt_db: f64,
) -> Vec<CrosstalkResult> {
    channels
        .par_iter()
        .map(|&(ch_id, wl, bw, power)| {
            let n_adj = channels
                .iter()
                .filter(|&&(other_id, other_wl, _, _)| {
                    other_id != ch_id && (wl - other_wl).abs() < bw * 3.0
                })
                .count();

            let xt = adjacent_xt_db + 10.0 * (n_adj.max(1) as f64).log10();
            let osnr = power - xt;

            CrosstalkResult {
                channel_id: ch_id,
                wavelength_nm: wl,
                n_adjacent: n_adj,
                crosstalk_db: xt,
                osnr_db: osnr,
            }
        })
        .collect()
}

// ── Power budget ─────────────────────────────────────────────────────

/// Path power budget result.
#[derive(Clone, Debug)]
pub struct PowerBudgetResult {
    pub source: usize,
    pub target: usize,
    pub total_loss_db: f64,
    pub received_power_dbm: f64,
    pub margin_db: f64,
    pub passed: bool,
}

/// Analyze power budget for all waveguide paths.
pub fn analyze_power_budget(
    waveguides: &[(usize, usize, f64)], // (source, target, wg_loss_db)
    mzi_ports: &[(Vec<usize>, usize, f64)], // (input_ports, output_port, insertion_loss)
    laser_power_dbm: f64,
    detector_sensitivity_dbm: f64,
) -> Vec<PowerBudgetResult> {
    waveguides
        .par_iter()
        .map(|&(src, tgt, wg_loss)| {
            let mzi_loss: f64 = mzi_ports
                .iter()
                .filter(|(inputs, output, _)| inputs.contains(&src) || *output == tgt)
                .map(|(_, _, loss)| loss)
                .sum();

            let total_loss = wg_loss + mzi_loss;
            let received = laser_power_dbm - total_loss;
            let margin = received - detector_sensitivity_dbm;

            PowerBudgetResult {
                source: src,
                target: tgt,
                total_loss_db: total_loss,
                received_power_dbm: received,
                margin_db: margin,
                passed: margin >= 0.0,
            }
        })
        .collect()
}

// ── Thermal phase shifter ────────────────────────────────────────────

/// Compute electrical power (mW) needed for a phase shift.
pub fn thermal_power_for_phase(
    phase_rad: f64,
    wavelength_nm: f64,
    heater_length_um: f64,
    dn_dt: f64,
    thermal_resistance_kw: f64,
) -> f64 {
    let wl_m = wavelength_nm * 1e-9;
    let l_m = heater_length_um * 1e-6;
    let delta_t = (phase_rad * wl_m) / (2.0 * std::f64::consts::PI * dn_dt * l_m);
    delta_t.abs() / thermal_resistance_kw
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_simple_2x2() {
        // 2-node full adjacency
        let adj = vec![0.0, 1.0, 1.0, 0.0];
        let result = route_waveguides(&adj, 2, 250.0, 2.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source, 0);
        assert_eq!(result[0].target, 1);
        assert!((result[0].length_um - 250.0).abs() < 1.0);
    }

    #[test]
    fn test_route_sparse() {
        // 3 nodes, only 0↔1 connected
        let adj = vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = route_waveguides(&adj, 3, 250.0, 2.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_route_loss_model() {
        let adj = vec![0.0, 1.0, 1.0, 0.0];
        let result = route_waveguides(&adj, 2, 1000.0, 2.0);
        // 1000 µm = 0.1 cm → 0.2 dB propagation loss
        assert!((result[0].loss_db - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_mzi_identity() {
        // phase=0 → identity
        let m = mzi_transfer_matrix(0.0);
        assert!((m[0] - 1.0).abs() < 1e-10); // M00 real
        assert!((m[6] - 1.0).abs() < 1e-10); // M11 real
        assert!(m[3].abs() < 1e-10); // M01 imag = 0
    }

    #[test]
    fn test_mzi_pi_barstate() {
        // phase=π → bar state (swap)
        let m = mzi_transfer_matrix(std::f64::consts::PI);
        // cos(π/2) ≈ 0, sin(π/2) ≈ 1
        assert!(m[0].abs() < 1e-10); // M00 real ≈ 0
        assert!((m[3] - 1.0).abs() < 1e-10); // M01 imag ≈ 1
    }

    #[test]
    fn test_cascade_identity() {
        // Cascade 0 phases → identity
        let result = cascade_mzi(&[]);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[6] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cascade_two_stages() {
        let phases = vec![std::f64::consts::PI / 4.0, std::f64::consts::PI / 4.0];
        let result = cascade_mzi(&phases);
        // Should be equivalent to single π/2 cascade (non-trivial)
        let mag = (result[0] * result[0] + result[1] * result[1]).sqrt();
        assert!(mag <= 1.0 + 1e-10);
    }

    #[test]
    fn test_crosstalk_single() {
        let channels = vec![(0, 1550.0, 0.8, 0.0)];
        let result = analyze_crosstalk(&channels, -25.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_crosstalk_adjacent() {
        let channels = vec![
            (0, 1550.0, 0.8, 0.0),
            (1, 1550.8, 0.8, 0.0),
            (2, 1551.6, 0.8, 0.0),
        ];
        let result = analyze_crosstalk(&channels, -25.0);
        assert_eq!(result.len(), 3);
        // Middle channel has 2 adjacent
        let mid = &result[1];
        assert_eq!(mid.n_adjacent, 2);
    }

    #[test]
    fn test_power_budget_pass() {
        let wgs = vec![(0, 1, 1.0)]; // 1 dB loss
        let mzis: Vec<(Vec<usize>, usize, f64)> = vec![];
        let result = analyze_power_budget(&wgs, &mzis, 0.0, -20.0);
        assert_eq!(result.len(), 1);
        assert!(result[0].passed); // 0 - 1 = -1 dBm > -20 dBm
        assert!((result[0].margin_db - 19.0).abs() < 0.01);
    }

    #[test]
    fn test_power_budget_fail() {
        let wgs = vec![(0, 1, 25.0)]; // 25 dB loss
        let result = analyze_power_budget(&wgs, &[], 0.0, -20.0);
        assert!(!result[0].passed); // 0 - 25 = -25 dBm < -20 dBm
    }

    #[test]
    fn test_thermal_power() {
        let p = thermal_power_for_phase(
            std::f64::consts::PI, // π phase shift
            1550.0,               // wavelength
            100.0,                // heater length
            1.86e-4,              // dn/dT
            10.0,                 // thermal R
        );
        assert!(p > 0.0);
        assert!(p < 100.0); // reasonable range
    }
}
