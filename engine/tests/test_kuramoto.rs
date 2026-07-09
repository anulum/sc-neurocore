// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

use sc_neurocore_engine::scpn::KuramotoSolver;

#[test]
fn identical_phases_give_r_equals_one() {
    let n = 16;
    let omega = vec![1.0; n];
    let coupling = vec![0.0; n * n];
    let phases = vec![0.5; n];
    let solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let r = solver.order_parameter();
    assert!(
        (r - 1.0).abs() < 1e-10,
        "R should be 1.0 for identical phases"
    );
}

#[test]
fn uniform_phases_give_r_near_zero() {
    let n = 1000;
    let omega = vec![1.0; n];
    let coupling = vec![0.0; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n as f64))
        .collect();
    let solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let r = solver.order_parameter();
    assert!(r < 0.01, "R should be near 0 for uniform phases, got {r}");
}

#[test]
fn strong_coupling_increases_r() {
    let n = 50;
    let omega = vec![1.0; n];
    // K=100 so effective coupling is K/N = 2.0 per oscillator (Kuramoto 1984)
    let coupling = vec![100.0; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * ((i * 37 % n) as f64) / (n as f64))
        .collect();

    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let r_initial = solver.order_parameter();
    let r_values = solver.run(500, 0.01, 42);
    let r_final = *r_values.last().expect("run should return 500 samples");
    assert!(
        r_final > r_initial + 0.1,
        "Strong coupling should increase R: initial={r_initial}, final={r_final}"
    );
}

#[test]
fn step_preserves_phase_count() {
    let n = 10;
    let omega = vec![1.0; n];
    let coupling = vec![0.1; n * n];
    let phases: Vec<f64> = (0..n).map(|i| i as f64 * 0.3).collect();
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.step(0.01, 0);
    assert_eq!(solver.get_phases().len(), n);
}

#[test]
#[should_panic(expected = "omega values must be finite")]
fn constructor_rejects_non_finite_omega() {
    let omega = vec![1.0, f64::NAN];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let _ = KuramotoSolver::new(omega, coupling, phases, 0.0);
}

#[test]
#[should_panic(expected = "coupling values must be finite")]
fn constructor_rejects_non_finite_coupling() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0, f64::INFINITY, 0.0, 0.0];
    let phases = vec![0.1, 0.2];
    let _ = KuramotoSolver::new(omega, coupling, phases, 0.0);
}

#[test]
#[should_panic(expected = "initial_phases values must be finite")]
fn constructor_rejects_non_finite_initial_phases() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, f64::NAN];
    let _ = KuramotoSolver::new(omega, coupling, phases, 0.0);
}

#[test]
#[should_panic(expected = "noise_amp must be finite and non-negative")]
fn constructor_rejects_negative_noise() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let _ = KuramotoSolver::new(omega, coupling, phases, -0.1);
}

#[test]
#[should_panic(expected = "dt must be finite and positive")]
fn step_rejects_non_positive_dt() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.step(0.0, 0);
}

#[test]
#[should_panic(expected = "phases values must be finite")]
fn set_phases_rejects_non_finite_values() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.set_phases(vec![0.3, f64::NAN]);
}

#[test]
#[should_panic(expected = "field_pressure must be finite")]
fn field_pressure_rejects_non_finite_values() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.set_field_pressure(f64::NAN);
}

#[test]
#[should_panic(expected = "w_flat values must be finite")]
fn step_ssgf_rejects_non_finite_geometry_matrix() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.step_ssgf(0.01, 0, &[0.0, f64::NAN, 0.0, 0.0], 1.0, &[], 0.0);
}

#[test]
#[should_panic(expected = "dt must be finite and positive")]
fn run_rejects_invalid_dt_even_without_steps() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let _ = solver.run(0, f64::NAN, 0);
}

#[test]
#[should_panic(expected = "sigma_g must be finite")]
fn run_ssgf_rejects_non_finite_geometry_gain_even_without_steps() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let _ = solver.run_ssgf(0, 0.01, 0, &[0.0; 4], f64::NAN, &[], 0.0);
}

#[test]
#[should_panic(expected = "h_flat values must be finite")]
fn run_ssgf_rejects_non_finite_pgbo_matrix_even_without_steps() {
    let omega = vec![1.0, 1.1];
    let coupling = vec![0.0; 4];
    let phases = vec![0.1, 0.2];
    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let _ = solver.run_ssgf(0, 0.01, 0, &[], 0.0, &[0.0, f64::NAN, 0.0, 0.0], 1.0);
}
