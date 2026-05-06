// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kane Si:P register mapper (Rust safety kernel)

//! Kane-architecture silicon qubit register layout mapper.
//!
//! Maps abstract spin pool sites to physical ³¹P donor positions in a
//! silicon lattice.  Computes exchange coupling J(d), decoherence budgets,
//! and feasibility constraints.
//!
//! # Exchange Coupling Model
//!
//! ```text
//! J(d) = J₀ · exp(-2d / a_B*)
//!
//! J₀   = 0.1 meV           (exchange prefactor)
//! a_B* = 2.5 nm             (effective Bohr radius in Si)
//! ```
//!
//! # Decoherence Budget
//!
//! ```text
//! T₂(nuclear)  = 30 s        (²⁸Si, Muhonen et al. 2014)
//! T₂(electron) = 2 ms        (1 K, standard)
//! gate_time     = 10 ns       (single-qubit)
//! max_depth     = T₂ / gate_time
//! ```
//!
//! References:
//! - Kane, B. E. (1998). Nature 393, 133–137.
//! - Muhonen, J. T. et al. (2014). Nature Nanotech. 9.

#![allow(non_snake_case, dead_code, unused_variables)]

/// Physical constants
const BOHR_RADIUS_STAR_NM: f64 = 2.5;   // effective Bohr radius in Si [nm]
const J0_MEV: f64 = 0.1;                 // exchange coupling prefactor [meV]
const T2_ELECTRON_MS: f64 = 2.0;         // electron spin T₂ at 1 K [ms]
const T2_NUCLEAR_S: f64 = 30.0;          // nuclear spin T₂ in ²⁸Si [s]
const GATE_TIME_NS: f64 = 10.0;          // single-qubit gate time [ns]

/// Physical register layout.
#[derive(Debug, Clone)]
pub struct KaneRegisterLayout {
    pub n_qubits: usize,
    pub positions: Vec<[f64; 2]>,
    pub coupling_matrix: Vec<Vec<f64>>,
    pub depth_nm: f64,
    pub t2_budget_ms: f64,
    pub max_gate_depth: usize,
}

/// Compute exchange coupling J(d) in meV.
///
/// J(d) = J₀ · exp(-2d / a_B*)
pub fn exchange_coupling(distance_nm: f64) -> f64 {
    if distance_nm <= 0.0 {
        return J0_MEV;
    }
    J0_MEV * (-2.0 * distance_nm / BOHR_RADIUS_STAR_NM).exp()
}

/// Compute qubit positions for a linear (1D) chain.
pub fn linear_positions(n: usize, spacing_nm: f64) -> Vec<[f64; 2]> {
    (0..n).map(|i| [i as f64 * spacing_nm, 0.0]).collect()
}

/// Compute qubit positions for a 2D grid.
pub fn grid_positions(n: usize, spacing_nm: f64) -> Vec<[f64; 2]> {
    let cols = (n as f64).sqrt().ceil() as usize;
    (0..n)
        .map(|i| [(i % cols) as f64 * spacing_nm, (i / cols) as f64 * spacing_nm])
        .collect()
}

/// Compute positions based on topology string.
pub fn compute_positions(n: usize, spacing_nm: f64, topology: &str) -> Vec<[f64; 2]> {
    match topology {
        "grid" => grid_positions(n, spacing_nm),
        _ => linear_positions(n, spacing_nm),
    }
}

/// Compute the full exchange coupling matrix for given positions.
///
/// Returns a symmetric n×n matrix with zero diagonal.
/// Each off-diagonal element J[i][j] = J₀ · exp(-2·d(i,j) / a_B*).
pub fn compute_coupling_matrix(positions: &[[f64; 2]]) -> Vec<Vec<f64>> {
    let n = positions.len();
    let mut matrix = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let d = (dx * dx + dy * dy).sqrt();
            let j_val = exchange_coupling(d);
            matrix[i][j] = j_val;
            matrix[j][i] = j_val;
        }
    }

    matrix
}

/// Build a complete Kane register layout.
pub fn map_pool_to_register(
    n_sites: usize,
    spacing_nm: f64,
    topology: &str,
    depth_nm: f64,
) -> KaneRegisterLayout {
    let positions = compute_positions(n_sites, spacing_nm, topology);
    let coupling_matrix = compute_coupling_matrix(&positions);
    let t2_budget_ms = T2_ELECTRON_MS;
    let max_gate_depth = (t2_budget_ms * 1e6 / GATE_TIME_NS) as usize;

    KaneRegisterLayout {
        n_qubits: n_sites,
        positions,
        coupling_matrix,
        depth_nm,
        t2_budget_ms,
        max_gate_depth,
    }
}

/// Check feasibility: nearest-neighbour coupling must be > threshold.
pub fn is_feasible(spacing_nm: f64, threshold_meV: f64) -> bool {
    exchange_coupling(spacing_nm) > threshold_meV
}

/// Find maximum coupling in a matrix (off-diagonal).
pub fn max_coupling(matrix: &[Vec<f64>]) -> f64 {
    let mut max = 0.0_f64;
    for row in matrix {
        for &val in row {
            if val > max {
                max = val;
            }
        }
    }
    max
}

/// Benchmark: build register + coupling matrix for n_sites.
pub fn benchmark_coupling_matrix(n_sites: usize) -> f64 {
    let layout = map_pool_to_register(n_sites, 20.0, "linear", 20.0);
    max_coupling(&layout.coupling_matrix)
}

/// Benchmark: repeated exchange_coupling calls.
pub fn benchmark_exchange_coupling(n_calls: usize) -> f64 {
    let mut result = 0.0;
    for i in 0..n_calls {
        result = exchange_coupling(i as f64 * 0.1);
    }
    result
}

fn main() {
    println!("Kane Si:P Register Mapper (Rust)");
    println!("================================");

    // Exchange coupling at various distances
    for d in [5.0, 10.0, 15.0, 20.0, 25.0, 50.0] {
        let j = exchange_coupling(d);
        let feasible = is_feasible(d, 1e-6);
        println!("  J({}nm) = {:.4e} meV  feasible={}", d, j, feasible);
    }

    // Build a 16-qubit register
    let layout = map_pool_to_register(16, 10.0, "linear", 20.0);
    println!("\nRegister: {} qubits, linear, spacing=10nm", layout.n_qubits);
    println!("  T₂ budget: {:.1} ms", layout.t2_budget_ms);
    println!("  Max depth: {}", layout.max_gate_depth);
    println!("  Max coupling: {:.4e} meV", max_coupling(&layout.coupling_matrix));

    // Grid register
    let grid = map_pool_to_register(64, 10.0, "grid", 20.0);
    println!("\nGrid register: {} qubits", grid.n_qubits);
    println!("  Max coupling: {:.4e} meV", max_coupling(&grid.coupling_matrix));

    // Benchmark
    use std::time::Instant;
    let t0 = Instant::now();
    let n = 256;
    let max_j = benchmark_coupling_matrix(n);
    let dt = t0.elapsed();
    println!("\nBenchmark: {} sites coupling matrix in {:.3} ms, max_J={:.4e}",
             n, dt.as_secs_f64() * 1e3, max_j);

    let t1 = Instant::now();
    let calls = 1_000_000;
    let j = benchmark_exchange_coupling(calls);
    let dt1 = t1.elapsed();
    println!("Benchmark: {} exchange_coupling calls in {:.3} ms ({:.0} ns/call)",
             calls, dt1.as_secs_f64() * 1e3, dt1.as_nanos() as f64 / calls as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_coupling_zero_distance() {
        let j = exchange_coupling(0.0);
        assert!((j - J0_MEV).abs() < 1e-10, "J(0) should be J₀: {}", j);
    }

    #[test]
    fn test_exchange_coupling_decay() {
        let j5 = exchange_coupling(5.0);
        let j10 = exchange_coupling(10.0);
        let j20 = exchange_coupling(20.0);
        assert!(j5 > j10, "J(5) > J(10): {} vs {}", j5, j10);
        assert!(j10 > j20, "J(10) > J(20): {} vs {}", j10, j20);
    }

    #[test]
    fn test_exchange_coupling_positive() {
        for d in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0] {
            let j = exchange_coupling(d);
            assert!(j >= 0.0, "J({}) should be non-negative: {}", d, j);
        }
    }

    #[test]
    fn test_linear_positions() {
        let pos = linear_positions(4, 20.0);
        assert_eq!(pos.len(), 4);
        assert!((pos[0][0] - 0.0).abs() < 1e-10);
        assert!((pos[1][0] - 20.0).abs() < 1e-10);
        assert!((pos[3][0] - 60.0).abs() < 1e-10);
        for p in &pos {
            assert!((p[1] - 0.0).abs() < 1e-10, "y should be 0 for linear");
        }
    }

    #[test]
    fn test_grid_positions() {
        let pos = grid_positions(9, 25.0);
        assert_eq!(pos.len(), 9);
        // 9 qubits → 3×3 grid
        assert!((pos[0][0] - 0.0).abs() < 1e-10);
        assert!((pos[0][1] - 0.0).abs() < 1e-10);
        assert!((pos[3][0] - 0.0).abs() < 1e-10);  // second row, first col
        assert!((pos[3][1] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_coupling_matrix_symmetry() {
        let pos = linear_positions(8, 20.0);
        let matrix = compute_coupling_matrix(&pos);
        assert_eq!(matrix.len(), 8);
        for i in 0..8 {
            assert_eq!(matrix[i].len(), 8);
            assert!((matrix[i][i] - 0.0).abs() < 1e-10, "Diagonal should be 0");
            for j in 0..8 {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10,
                        "Matrix should be symmetric at [{},{}]", i, j);
            }
        }
    }

    #[test]
    fn test_coupling_matrix_decay_with_distance() {
        let pos = linear_positions(4, 20.0);
        let matrix = compute_coupling_matrix(&pos);
        // Nearest neighbour > next-nearest > furthest
        assert!(matrix[0][1] > matrix[0][2],
                "NN > NNN: {} vs {}", matrix[0][1], matrix[0][2]);
        assert!(matrix[0][2] > matrix[0][3],
                "NNN > far: {} vs {}", matrix[0][2], matrix[0][3]);
    }

    #[test]
    fn test_coupling_matrix_all_positive() {
        let pos = grid_positions(16, 10.0);
        let matrix = compute_coupling_matrix(&pos);
        for row in &matrix {
            for &val in row {
                assert!(val >= 0.0, "Coupling should be non-negative: {}", val);
            }
        }
    }

    #[test]
    fn test_register_layout() {
        let layout = map_pool_to_register(8, 20.0, "linear", 20.0);
        assert_eq!(layout.n_qubits, 8);
        assert_eq!(layout.positions.len(), 8);
        assert_eq!(layout.coupling_matrix.len(), 8);
        assert!(layout.t2_budget_ms > 0.0);
        assert!(layout.max_gate_depth > 0);
    }

    #[test]
    fn test_single_qubit() {
        let layout = map_pool_to_register(1, 20.0, "linear", 20.0);
        assert_eq!(layout.n_qubits, 1);
        assert_eq!(layout.coupling_matrix[0][0], 0.0);
    }

    #[test]
    fn test_feasibility_close_spacing() {
        assert!(is_feasible(5.0, 1e-6), "5nm should be feasible");
        assert!(is_feasible(10.0, 1e-6), "10nm should be feasible");
    }

    #[test]
    fn test_feasibility_wide_spacing() {
        assert!(!is_feasible(50.0, 1e-6), "50nm should be infeasible");
    }

    #[test]
    fn test_large_register() {
        // 256-qubit register should not panic or produce NaN
        let layout = map_pool_to_register(256, 15.0, "grid", 20.0);
        assert_eq!(layout.n_qubits, 256);
        let max_j = max_coupling(&layout.coupling_matrix);
        assert!(max_j.is_finite(), "Max coupling should be finite");
        assert!(max_j > 0.0, "Should have non-zero coupling");
    }

    #[test]
    fn test_cross_parity_with_python() {
        // Must match Python kane_mapper.py to 1e-10
        // J(20nm) = 0.1 * exp(-2*20/2.5) = 0.1 * exp(-16)
        let j20 = exchange_coupling(20.0);
        let expected = 0.1 * (-16.0_f64).exp();
        assert!((j20 - expected).abs() < 1e-15,
                "Cross-parity: expected {:.6e}, got {:.6e}", expected, j20);
    }

    #[test]
    fn test_benchmark_runs() {
        let max_j = benchmark_coupling_matrix(8);
        assert!(max_j >= 0.0);
    }

    #[test]
    fn test_exchange_benchmark() {
        let j = benchmark_exchange_coupling(100);
        assert!(j >= 0.0);
    }
}
