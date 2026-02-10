use sc_neurocore_engine::scpn::KuramotoSolver;

#[test]
fn ssgf_step_without_extras_matches_basic_step() {
    let n = 16;
    let omega = vec![1.0; n];
    let coupling = vec![0.3; n * n];
    let phases: Vec<f64> = (0..n).map(|i| 0.3 * i as f64).collect();

    let mut solver_a = KuramotoSolver::new(omega.clone(), coupling.clone(), phases.clone(), 0.0);
    let mut solver_b = KuramotoSolver::new(omega, coupling, phases, 0.0);

    let r_basic = solver_a.step(0.01, 42);
    let r_ssgf = solver_b.step_ssgf(0.01, 42, &[], 0.0, &[], 0.0);

    assert!(
        (r_basic - r_ssgf).abs() < 1e-14,
        "step_ssgf with no extras should match step: basic={r_basic}, ssgf={r_ssgf}",
    );
    assert_eq!(solver_a.get_phases(), solver_b.get_phases());
}

#[test]
fn geometry_coupling_accelerates_synchronization() {
    let n = 50;
    let omega = vec![1.0; n];
    let coupling = vec![0.1; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * ((i * 37 % n) as f64) / (n as f64))
        .collect();
    let w = vec![1.0; n * n];

    let mut solver_no_geo =
        KuramotoSolver::new(omega.clone(), coupling.clone(), phases.clone(), 0.0);
    let mut solver_with_geo = KuramotoSolver::new(omega, coupling, phases, 0.0);

    let r_no_geo = solver_no_geo.run(300, 0.01, 0);
    let r_with_geo = solver_with_geo.run_ssgf(300, 0.01, 0, &w, 1.0, &[], 0.0);

    let r_final_no = *r_no_geo.last().unwrap();
    let r_final_geo = *r_with_geo.last().unwrap();

    assert!(
        r_final_geo > r_final_no + 0.05,
        "Geometry coupling should boost R: no_geo={r_final_no}, geo={r_final_geo}",
    );
}

#[test]
fn field_pressure_creates_preferred_phase() {
    let n = 20;
    let omega = vec![0.0; n];
    let coupling = vec![0.0; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n as f64))
        .collect();

    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    solver.set_field_pressure(1.0);
    solver.run_ssgf(1000, 0.01, 0, &[], 0.0, &[], 0.0);

    let r = solver.order_parameter();
    assert!(
        r > 0.1,
        "Field pressure should create some phase structure, R={r}"
    );
}

#[test]
fn pgbo_coupling_modulates_dynamics() {
    let n = 20;
    let omega = vec![1.0; n];
    let coupling = vec![0.5; n * n];
    let phases: Vec<f64> = (0..n).map(|i| 0.1 * i as f64).collect();
    let mut h = vec![0.0; n * n];
    for i in 0..n / 2 {
        for j in 0..n / 2 {
            h[i * n + j] = 2.0;
        }
    }

    let mut solver = KuramotoSolver::new(omega, coupling, phases, 0.0);
    let r_values = solver.run_ssgf(200, 0.01, 0, &[], 0.0, &h, 1.0);
    let r_final = *r_values.last().unwrap();

    assert!(r_final > 0.0 && r_final <= 1.0);
}
