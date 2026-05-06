// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Radical Pair Mechanism (Rust safety kernel)

//! Semiclassical Radical Pair Mechanism for ATP hydrolysis gating.
//!
//! Implements the singlet–triplet interconversion dynamics from the
//! hybrid Hamiltonian's H_int interaction term.  Singlet yield Φ_S
//! determines ATP hydrolysis probability in the Fisher-Posner model.
//!
//! # Mathematical Model
//!
//! ```text
//! ω₀       = γ_e · B / (2π × 10⁶)       [MHz, Larmor frequency]
//! ω_eff    = √(ω₀² + a²)                 [effective precession]
//! j_ratio  = J² / (J² + ω_eff²)          [exchange preservation]
//! recomb   = 1 / (1 + (k·τ)⁻²)           [recombination factor]
//! Φ_S      = 0.25 + 0.75 · j_ratio · recomb
//! ```
//!
//! References:
//! - Schulten, K. & Wolynes, P. G. (1978). J. Chem. Phys. 68.
//! - Hore, P. J. & Mouritsen, H. (2016). Ann. Rev. Biophys. 45.
//! - Fisher, M. P. A. (2015). Annals of Physics 362.

#![allow(non_snake_case)]

/// Physical constants
const GAMMA_E: f64 = 1.760_859_630_23e11; // electron gyromagnetic ratio [rad/(s·T)]

/// Parameters for a radical pair system.
#[derive(Debug, Clone)]
pub struct RadicalPairParams {
    /// Isotropic hyperfine coupling constant [MHz]
    pub hyperfine_a: f64,
    /// Exchange coupling [MHz]
    pub exchange_j: f64,
    /// Radical pair recombination rate [µs⁻¹]
    pub recombination_rate: f64,
    /// Radical pair lifetime [µs]
    pub lifetime_us: f64,
}

impl Default for RadicalPairParams {
    fn default() -> Self {
        Self {
            hyperfine_a: 10.0,
            exchange_j: 1.0,
            recombination_rate: 0.1,
            lifetime_us: 100.0,
        }
    }
}

/// Compute single-effective-nucleus singlet yield Φ_S.
///
/// Returns a value in [0, 1] representing the probability of
/// productive ATP hydrolysis (singlet channel).
///
/// Uses the Schulten-Wolynes semiclassical one-effective-nucleus model.
/// Multi-nuclear Posner calculations require the Python exact
/// density-matrix implementation.
pub fn singlet_yield(params: &RadicalPairParams, b_local: f64) -> f64 {
    singlet_yield_multi(params, b_local, 1)
}

/// Compute singlet yield with explicit number of nuclear spins.
///
/// Only n_nuclei=1 is implemented in this safety kernel.
pub fn singlet_yield_multi(params: &RadicalPairParams, b_local: f64, n_nuclei: usize) -> f64 {
    if n_nuclei != 1 {
        panic!("multi-nuclear radical-pair yield requires the Python exact density-matrix path");
    }

    // Larmor frequency (rad/s → MHz)
    let omega_0_mhz = GAMMA_E * b_local / (2.0 * std::f64::consts::PI * 1e6);

    // Effective one-nucleus HF after isotropic orientation averaging.
    let a_eff = params.hyperfine_a / 3.0_f64.sqrt();
    let omega_eff = (omega_0_mhz * omega_0_mhz + a_eff * a_eff).sqrt();

    // Exchange coupling preservation
    let j = params.exchange_j;
    let j_ratio = if omega_eff > 0.0 {
        (j * j) / (j * j + omega_eff * omega_eff)
    } else {
        1.0
    };

    // Recombination dynamics
    let k_tau = params.recombination_rate * params.lifetime_us;
    let k_tau_sq = k_tau * k_tau;
    let recomb_factor = if k_tau_sq > 1e-12 {
        1.0 / (1.0 + 1.0 / k_tau_sq)
    } else {
        0.0
    };

    // Singlet yield: 1/4 statistical baseline + coherent contribution
    let phi_s = 0.25 + 0.75 * j_ratio * recomb_factor;
    phi_s.clamp(0.0, 1.0)
}

/// Compute singlet yield over an array of magnetic field values.
///
/// Writes results into `out`. `b_range` and `out` must have the same length.
pub fn singlet_yield_sweep(params: &RadicalPairParams, b_range: &[f64], out: &mut [f64]) {
    assert_eq!(
        b_range.len(),
        out.len(),
        "b_range and out must have same length"
    );
    for (i, &b) in b_range.iter().enumerate() {
        out[i] = singlet_yield(params, b);
    }
}

/// Compute ATP hydrolysis efficiency from singlet yield only.
///
/// Nonzero entanglement boosts are rejected because the Rust kernel does
/// not compute the exact two-site singlet RDM used by the Python model.
pub fn atp_efficiency(params: &RadicalPairParams, b_local: f64, entanglement_boost: f64) -> f64 {
    if entanglement_boost.abs() > 1e-12 {
        panic!("entanglement_boost requires the Python exact two-site singlet RDM path");
    }
    let phi_s = singlet_yield(params, b_local);
    phi_s.clamp(0.0, 1.0)
}

/// Benchmark: compute singlet yield n_calls times, return final value.
pub fn benchmark_singlet_yield(n_calls: usize) -> f64 {
    let params = RadicalPairParams::default();
    let mut result = 0.0;
    for i in 0..n_calls {
        let b = (i as f64) * 1e-7; // sweep 0 → ~100 µT
        result = singlet_yield(&params, b);
    }
    result
}

/// Benchmark: field sweep over n_points, return mean yield.
pub fn benchmark_field_sweep(n_points: usize) -> f64 {
    let params = RadicalPairParams::default();
    let b_range: Vec<f64> = (0..n_points)
        .map(|i| (i as f64) / (n_points as f64) * 1e-3)
        .collect();
    let mut out = vec![0.0_f64; n_points];
    singlet_yield_sweep(&params, &b_range, &mut out);
    let total: f64 = out.iter().sum();
    total / n_points as f64
}

fn main() {
    let params = RadicalPairParams::default();
    println!(
        "RadicalPairParams: a={} MHz, J={} MHz, k={} µs⁻¹, τ={} µs",
        params.hyperfine_a, params.exchange_j, params.recombination_rate, params.lifetime_us
    );

    // Single-point
    let phi = singlet_yield(&params, 0.0);
    println!("Φ_S(B=0) = {:.6}", phi);

    let phi_earth = singlet_yield(&params, 50e-6);
    println!("Φ_S(B=50µT) = {:.6}", phi_earth);

    // Field sweep
    let n = 1000;
    let b_range: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-6).collect();
    let mut yields = vec![0.0_f64; n];
    singlet_yield_sweep(&params, &b_range, &mut yields);
    println!(
        "Field sweep: {} points, mean Φ_S = {:.6}",
        n,
        yields.iter().sum::<f64>() / n as f64
    );

    // Benchmark
    use std::time::Instant;
    let t0 = Instant::now();
    let calls = 1_000_000;
    let result = benchmark_singlet_yield(calls);
    let dt = t0.elapsed();
    println!(
        "Benchmark: {} calls in {:.3} ms ({:.0} ns/call), final={:.6}",
        calls,
        dt.as_secs_f64() * 1e3,
        dt.as_nanos() as f64 / calls as f64,
        result
    );

    // Field sweep benchmark
    let t1 = Instant::now();
    let sweep_size = 100_000;
    let mean_yield = benchmark_field_sweep(sweep_size);
    let dt1 = t1.elapsed();
    println!(
        "Sweep benchmark: {} points in {:.3} ms, mean={:.6}",
        sweep_size,
        dt1.as_secs_f64() * 1e3,
        mean_yield
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let p = RadicalPairParams::default();
        assert_eq!(p.hyperfine_a, 10.0);
        assert_eq!(p.exchange_j, 1.0);
        assert_eq!(p.lifetime_us, 100.0);
    }

    #[test]
    fn test_singlet_yield_zero_field() {
        let p = RadicalPairParams::default();
        let phi = singlet_yield(&p, 0.0);
        assert!(phi >= 0.0 && phi <= 1.0, "Yield out of range: {}", phi);
    }

    #[test]
    fn test_singlet_yield_bounded() {
        let p = RadicalPairParams::default();
        for b in [0.0, 1e-6, 50e-6, 1e-3, 1.0, 10.0].iter() {
            let phi = singlet_yield(&p, *b);
            assert!(
                phi >= 0.0 && phi <= 1.0,
                "Yield {} out of range at B={}",
                phi,
                b
            );
        }
    }

    #[test]
    fn test_strong_exchange_preserves_singlet() {
        let p = RadicalPairParams {
            exchange_j: 1000.0,
            ..Default::default()
        };
        let phi = singlet_yield(&p, 0.0);
        assert!(phi > 0.9, "Strong J should give high yield: {}", phi);
    }

    #[test]
    fn test_weak_exchange_reduces_singlet() {
        let strong = RadicalPairParams {
            exchange_j: 100.0,
            ..Default::default()
        };
        let weak = RadicalPairParams {
            exchange_j: 0.01,
            ..Default::default()
        };
        let phi_strong = singlet_yield(&strong, 0.0);
        let phi_weak = singlet_yield(&weak, 0.0);
        assert!(
            phi_strong > phi_weak,
            "Strong J ({}) should exceed weak J ({})",
            phi_strong,
            phi_weak
        );
    }

    #[test]
    fn test_field_sweep_length() {
        let p = RadicalPairParams::default();
        let b = vec![0.0, 1e-6, 50e-6, 1e-3];
        let mut out = vec![0.0; 4];
        singlet_yield_sweep(&p, &b, &mut out);
        for val in &out {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_atp_efficiency_range() {
        let p = RadicalPairParams::default();
        let eff = atp_efficiency(&p, 0.0, 0.0);
        assert!((0.0..=1.0).contains(&eff), "ATP eff out of range: {}", eff);
    }

    #[test]
    #[should_panic]
    fn test_atp_efficiency_boost() {
        let p = RadicalPairParams::default();
        atp_efficiency(&p, 0.0, 0.3);
    }

    #[test]
    #[should_panic]
    fn test_atp_efficiency_clipping() {
        let p = RadicalPairParams::default();
        atp_efficiency(&p, 0.0, 100.0);
    }

    #[test]
    fn test_short_lifetime_low_yield() {
        let p = RadicalPairParams {
            lifetime_us: 0.001, // very short lifetime
            recombination_rate: 0.001,
            ..Default::default()
        };
        let phi = singlet_yield(&p, 0.0);
        // Very short k·τ → recomb_factor → 0 → yield → 0.25
        assert!(phi < 0.5, "Short lifetime should give low yield: {}", phi);
    }

    #[test]
    fn test_earth_field_vs_zero() {
        let p = RadicalPairParams::default();
        let phi_zero = singlet_yield(&p, 0.0);
        let phi_earth = singlet_yield(&p, 50e-6);
        // Earth's field (50 µT) is extremely weak — should barely change yield
        assert!(
            (phi_zero - phi_earth).abs() < 0.01,
            "Earth field should barely change yield: {} vs {}",
            phi_zero,
            phi_earth
        );
    }

    #[test]
    fn test_strong_field_reduces_singlet() {
        let p = RadicalPairParams::default();
        let phi_zero = singlet_yield(&p, 0.0);
        let phi_strong = singlet_yield(&p, 10.0); // 10 Tesla!
                                                  // Very strong field → large ω₀ → j_ratio → 0 → yield → 0.25
        assert!(
            phi_strong <= phi_zero,
            "Strong field should not increase yield: {} vs {}",
            phi_strong,
            phi_zero
        );
    }

    #[test]
    fn test_benchmark_runs() {
        let result = benchmark_singlet_yield(100);
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_field_sweep_benchmark() {
        let mean = benchmark_field_sweep(100);
        assert!(mean >= 0.0 && mean <= 1.0);
    }

    #[test]
    fn test_cross_parity_with_python() {
        // Single effective nucleus:
        //   a_eff = 10 / √3 ≈ 5.774 MHz
        //   ω_eff = 5.774 MHz (at B=0)
        //   j_ratio = 1/(1 + 33.33) = 0.0291
        //   k_tau = 10, recomb = 0.9901
        //   Φ_S = 0.25 + 0.75 * 0.0291 * 0.9901 ≈ 0.2716
        let p = RadicalPairParams::default();
        let phi_0 = singlet_yield(&p, 0.0);
        assert!(
            (phi_0 - 0.2716).abs() < 0.01,
            "Cross-parity check: expected ~0.272, got {}",
            phi_0
        );
    }

    #[test]
    fn test_single_nucleus_explicit() {
        let p = RadicalPairParams::default();
        let phi_1 = singlet_yield_multi(&p, 0.0, 1);
        // a_eff = 10·√(1/3) = 5.774, ω_eff = 5.774
        // j_ratio = 1/(1+33.33) = 0.0291
        // Φ_S = 0.25 + 0.75 * 0.0291 * 0.9901 ≈ 0.272
        assert!(
            phi_1 >= 0.26 && phi_1 <= 0.29,
            "Single-nucleus yield out of range: {}",
            phi_1
        );
    }

    #[test]
    #[should_panic]
    fn test_multi_nuclear_rejected() {
        let p = RadicalPairParams::default();
        singlet_yield_multi(&p, 0.0, 6);
    }

    #[test]
    #[should_panic]
    fn test_sweep_length_mismatch() {
        let p = RadicalPairParams::default();
        let b = vec![0.0, 1.0];
        let mut out = vec![0.0; 3]; // wrong length
        singlet_yield_sweep(&p, &b, &mut out);
    }
}
