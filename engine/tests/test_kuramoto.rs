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
    let coupling = vec![2.0; n * n];
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
